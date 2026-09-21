package com.avtracker.mobile.fusion

import android.util.Log
import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.gender.PitchGender
import com.avtracker.mobile.voice.SessionSpeakerTracker
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.voice.VerifierResult

data class FusionConfig(
    val language: String = "en",
    /** Expected number of people; limits identities like the Python `num_speakers`. */
    val numSpeakers: Int? = null,
    val verifierConfidenceMin: Float = 0.70f,
    /** Name of the speech recognition model, recorded in the session metrics. */
    val whisperModel: String = "base"
)

data class TranscriptEntry(
    val speakerId: String,
    val speakerName: String?,
    val text: String,
    val decision: String,
    val timeSec: Double,
    /** Wall-clock time, for the transcript file. */
    val wallMillis: Long = System.currentTimeMillis()
) {
    /** "(Name)" for a real identity, otherwise the speaker id, so different unknown people stay distinguishable. */
    val label: String
        get() = if (speakerName != null && !IdentityRegistry.isGeneric(speakerName)) "($speakerName)" else "($speakerId)"
}

/**
 * Audio side of av-tracker's multimodal fusion (RealtimeTranscriber): chunks the microphone stream, finds speech
 * turns, transcribes each with Whisper and attributes it to a person by combining voice biometrics, voice gender, the
 * face tracker and the active-speaker detector; then learns names from the conversation and saves voices.
 *
 * [processChunk] does the work for one window and can be driven directly (tests, file input); [start] runs a
 * worker thread that feeds it from an [AudioTimeline].
 *
 * The LLM speaker analysis ([LlmSpeakerAnalysis]) is off unless an [LlmClient] is given, as run_multimodal_tracker.py
 * constructs the transcriber with `use_ai_analysis=False`.
 */
class FusionEngine(
    private val timeline: AudioTimeline,
    private val turnDetector: TurnDetector,
    private val voice: VoiceEncoder,
    val verifier: SpeakerVerifier,
    private val recognizer: SpeechRecognizer,
    private val asd: SpeakerActivity?,
    private val registry: IdentityRegistry,
    private val config: FusionConfig = FusionConfig(),
    entities: EntityExtractor = NoEntities,
    faceTracker: FaceTrackerBridge? = null,
    val log: SessionLog = SessionLog(),
    private val onTranscript: (TranscriptEntry) -> Unit = {},
    private val onFaceRenamed: (personId: String, oldName: String, newName: String) -> Unit = { _, _, _ -> },
    private val clock: () -> Double = AudioUtils::nowSec,
    private val wallClockMillis: () -> Long = System::currentTimeMillis,
    private val genderDetector: (FloatArray) -> String? = PitchGender::detect,
    /** The LLM behind the periodic speaker analysis; null (the av-tracker default, `use_ai_analysis=False`) turns it off. */
    llm: LlmClient? = null,
    llmInBackground: Boolean = true
) {
    private val face = faceTracker
    val sessionTracker = SessionSpeakerTracker(
        newId = registry::newPersonId,
        numSpeakers = config.numSpeakers,
        otherTrackedIds = { registry.knownPersonIds() }
    )
    val book = SpeakerBook()
    val decider = SegmentDecider(
        registry = registry,
        sessionTracker = sessionTracker,
        book = book,
        numSpeakers = config.numSpeakers,
        verifierThreshold = verifier.threshold,
        verifierConfidenceMin = config.verifierConfidenceMin
    )
    val naming = SpeakerNaming(registry, verifier, book, voice, entities, faceTracker, config.numSpeakers, clock, genderDetector)
    val llmAnalysis = LlmSpeakerAnalysis(
        llm, registry, book, naming, config.numSpeakers,
        transcript = { transcript }, segmentCount = { log.segmentMetrics.size }, background = llmInBackground
    )

    private val entries = ArrayList<TranscriptEntry>()
    private val chunker = AudioChunker()

    @Volatile private var running = false
    private var worker: Thread? = null

    init {
        // Voices already known start out registered, as run_multimodal_tracker.py does before the transcriber starts.
        for (name in verifier.names) registry.registerKnown(name)
    }

    val transcript: List<TranscriptEntry> get() = synchronized(entries) { entries.toList() }

    fun start() {
        if (running) return
        running = true
        worker = Thread({ loop() }, "fusion-engine").also { it.start() }
    }

    fun stop() {
        running = false
        worker?.join(2_000)
        worker = null
    }

    private fun loop() {
        while (running) {
            val window = chunker.next(timeline)
            if (window == null) {
                Thread.sleep(200)
                continue
            }
            val audio = timeline.slice(window.startSec, window.endSec) ?: continue
            try {
                processChunk(audio, window.startSec, window.newZoneStartSamples)
            } catch (t: Throwable) {
                Log.e(TAG, "Chunk processing failed", t)
            }
        }
    }

    /** Port of RealtimeTranscriber._process_chunk. [chunkStartSec] is the stream time of the first sample. */
    fun processChunk(rawAudio: FloatArray, chunkStartSec: Double, newZoneStartSamples: Int = 0) {
        if (AudioUtils.peak(rawAudio) < SILENCE_PEAK) return
        val audio = AudioUtils.normalize(rawAudio)
        val sr = AudioUtils.SAMPLE_RATE

        val turns = turnDetector.turns(audio).filter { it.durationSec >= MIN_TURN_SEC }
        if (turns.isEmpty()) {
            // No diarization: only the new zone, to avoid re-transcribing the context.
            val zone = audio.copyOfRange(minOf(newZoneStartSamples, audio.size), audio.size)
            if (zone.size >= (sr * MIN_TURN_SEC).toInt()) {
                processSegment(zone, null, chunkStartSec + newZoneStartSamples.toDouble() / sr, chunkStartSec + audio.size.toDouble() / sr)
            }
            return
        }

        // With 2+ diarized speakers the Python code also runs SepFormer. Its separated streams are matched to
        // speakers and then thrown away (the result of _match_streams_to_speakers is never used), so the only lasting
        // effect is this: every diarized speaker's audio, concatenated, is registered with the session tracker
        // before the segments are attributed. That effect is reproduced here without running the network.
        if (turns.map { it.slot }.toSet().size >= 2) {
            val perSpeaker = LinkedHashMap<Int, MutableList<FloatArray>>()
            for (turn in turns) {
                val s = (turn.startSec * sr).toInt()
                val e = minOf((turn.endSec * sr).toInt(), audio.size)
                if (e - s >= (sr * MIN_TURN_SEC).toInt()) perSpeaker.getOrPut(turn.slot) { mutableListOf() } += audio.copyOfRange(s, e)
            }
            for (segments in perSpeaker.values) {
                val joined = FloatArray(segments.sumOf { it.size })
                var pos = 0
                for (seg in segments) { System.arraycopy(seg, 0, joined, pos, seg.size); pos += seg.size }
                voice.embedNormalized(joined)?.let { sessionTracker.getOrCreate(it) }
            }
        }

        // Consecutive turns of the same slot merge only if the voice really is the same: diarization sometimes
        // labels two different people with one label.
        class Group(val slot: Int, val start: Int, var end: Int, val embedding: FloatArray?)
        val groups = ArrayList<Group>()
        for (turn in turns) {
            val startS = (turn.startSec * sr).toInt()
            val endS = minOf((turn.endSec * sr).toInt(), audio.size)
            val segStart = maxOf(startS, newZoneStartSamples)
            if (segStart >= endS) continue

            val turnEmbedding = voice.embedNormalized(audio.copyOfRange(segStart, endS))
            val last = groups.lastOrNull()
            var sameVoice = true
            if (last?.embedding != null && turnEmbedding != null) {
                sameVoice = AudioUtils.dot(last.embedding, turnEmbedding) >= SAME_VOICE_SIM
            }
            if (last != null && last.slot == turn.slot && sameVoice) last.end = endS else groups += Group(turn.slot, segStart, endS, turnEmbedding)
        }

        for (group in groups) {
            val segment = audio.copyOfRange(group.start, group.end)
            if (segment.size.toDouble() / sr < MIN_TURN_SEC) continue

            // Very short segments get surrounding audio so Whisper has enough context.
            var forRecognition = segment
            val duration = segment.size.toDouble() / sr
            if (duration < MIN_SEGMENT_SEC) {
                val padNeeded = ((MIN_SEGMENT_SEC - duration) * sr).toInt()
                val padBefore = minOf(group.start - newZoneStartSamples, padNeeded / 2).coerceAtLeast(0)
                val padAfter = minOf(audio.size - group.end, padNeeded - padBefore).coerceAtLeast(0)
                forRecognition = audio.copyOfRange(group.start - padBefore, group.end + padAfter)
            }
            attributeSegment(
                segment, forRecognition,
                chunkStartSec + group.start.toDouble() / sr,
                chunkStartSec + group.end.toDouble() / sr
            )
        }
    }

    /**
     * The per-group step of _process_chunk: a session id from the group's own single-voice audio and pitch (never from the
     * diarization label), resolved to the face it was confirmed to be, then the segment is processed.
     * [voiceAudio] is the group's audio; [recognitionAudio] is the same audio possibly padded with its surroundings.
     */
    fun attributeSegment(voiceAudio: FloatArray, recognitionAudio: FloatArray, wallStartSec: Double, wallEndSec: Double) {
        val groupGender = genderDetector(voiceAudio)
        var sessionId = voice.embedNormalized(voiceAudio)?.let { sessionTracker.getOrCreate(it, emptySet(), groupGender) }
        sessionId?.let { sid ->
            val face = decider.sessionToFace[sid]
            if (face != null && registry.nameOf(face) != null) sessionId = face
        }
        processSegment(recognitionAudio, sessionId, wallStartSec, wallEndSec)
    }

    /** Port of RealtimeTranscriber._process_segment. */
    fun processSegment(audio: FloatArray, sessionHint: String?, wallStartSec: Double, wallEndSec: Double) {
        val segmentStartNanos = System.nanoTime()

        val whisperStart = System.nanoTime()
        val result = recognizer.transcribe(audio, config.language) ?: return
        val whisperMs = (System.nanoTime() - whisperStart) / 1e6
        if (!TranscriptFilter.accept(result.text, result.avgLogProb, result.noSpeechProb, config.language)) return
        val text = result.text

        val verifierStart = System.nanoTime()
        val embedding = voice.embed(audio)
        val verdict = embedding?.let { verifier.identify(it) } ?: VerifierResult("Unknown", 0f, emptyList())
        val verifierMs = (System.nanoTime() - verifierStart) / 1e6
        val unit = embedding?.let { AudioUtils.l2Normalize(it) }
        val audioGender = genderDetector(audio)

        val asdWindowStart = wallStartSec - ASD_PAD_SEC
        val asdWindowEnd = wallEndSec + ASD_PAD_SEC
        val activeFaces = registry.activeFacesSnapshot()
        val asdTrack = if (asd != null && activeFaces.isNotEmpty()) asd.getActiveSpeaker(asdWindowStart, asdWindowEnd) else null

        val decision = decider.decide(
            SegmentEvidence(
                rawBestName = verdict.bestName,
                rawConf = verdict.bestScore,
                candidates = verdict.candidates,
                asdTrack = asdTrack,
                asdGuess = { asd?.getBestGuess(asdWindowStart, asdWindowEnd) },
                asdAvailable = asd != null,
                sessionHint = sessionHint,
                voiceEmbedding = unit,
                audioGender = audioGender
            )
        )
        Log.i(TAG, "Decision: ${decision.label} -> ${decision.verifiedName ?: decision.speakerId}")

        // High-confidence audio enriches the voice bank (>= 4 s only, like the Python auto-enrol).
        if (decision.type == "VERIFIER_HIGH" && audio.size >= 4 * AudioUtils.SAMPLE_RATE && unit != null) {
            decision.verifiedName?.let { verifier.autoEnroll(PyText.stripHomonymSuffix(it), unit) }
        }

        // FALSE NEGATIVE detection: the verifier had a plausible match that was not used.
        val nowMillis = wallClockMillis()
        val fnReasons = log.falseNegativeReasons(
            decision, verdict.bestName, verdict.bestScore, verdict.candidates, audioGender,
            verifier.threshold, config.verifierConfidenceMin
        )
        if (fnReasons.isNotEmpty()) {
            log.recordFalseNegative(
                log.newFalseNegative(
                    fnReasons, decision, decision.verifiedName ?: decision.speakerId, text,
                    verdict.bestName, verdict.bestScore, verdict.candidates, audioGender, activeFaces.size, nowMillis
                )
            )
            fnReasons.forEach { Log.w(TAG, "[FN] $it | decision=${decision.label} | \"${text.take(60)}\"") }
        }

        // Keep the audio of every speaker: it becomes their voice profile once they get a name.
        naming.accumulateAndAutoSave(decision.speakerId, audio)

        // Face identity bootstraps voice identity when ASD says a recognised face is speaking.
        if (decision.asdPersonId != null && decision.asdTrackId != null) {
            naming.tryFaceVoiceBinding(decision.asdTrackId, audio)
        }

        // Names from what is said: self-introductions, then vocatives/mentions.
        naming.updateNamesIncremental(decision.speakerId, text, audio)
        naming.detectContextNames(text, decision.speakerId)
        naming.applyContextNaming(decision.speakerId)

        // Periodic LLM analysis for the speakers that are still unidentified.
        llmAnalysis.maybeRun()

        synchronized(entries) { naming.deduceLastSpeaker(entries.map { it.speakerId }) }

        val finalName = registry.nameOf(decision.speakerId)
        if (decision.label.startsWith("VERIFIER_HIGH") || decision.label.startsWith("VERIFIER_MOD")) {
            syncFaceName(decision, finalName, audioGender)
        }

        // FALSE POSITIVE risk + structured metrics.
        val (fpRisk, fpReasons) = log.falsePositiveRisk(
            decision, finalName, verdict.bestScore, audioGender, verifier.names, verifier.threshold,
            config.numSpeakers, decider.identifiedSpeakers.size
        )
        if (fpReasons.isNotEmpty()) Log.w(TAG, "[FP risk=$fpRisk] ${fpReasons.joinToString(" | ")} | decision=${decision.label} final=$finalName")
        log.recordSegment(
            nowMillis, decision, verdict.bestName, verdict.bestScore, verdict.candidates, finalName, audioGender,
            asdActive = decision.asdPersonId != null, faces = activeFaces.size, fnReasons = fnReasons,
            fpRisk = fpRisk, fpReasons = fpReasons, textLength = text.length,
            durationSec = audio.size.toDouble() / AudioUtils.SAMPLE_RATE, whisperMs = whisperMs, verifierMs = verifierMs,
            segmentMs = (System.nanoTime() - segmentStartNanos) / 1e6, avgLogProb = result.avgLogProb, avgNoSpeech = result.noSpeechProb
        )

        appendTranscript(TranscriptEntry(decision.speakerId, finalName, text, decision.label, clock(), nowMillis))
    }

    /**
     * Port of the "camera sync" step: a confirmed voice identity names the generic face the ASD pointed at, unless
     * the face tracker has already confirmed a different real name for it; if it has, any other visible generic face
     * is tried when the voice gender is known.
     */
    private fun syncFaceName(decision: Decision, finalName: String?, audioGender: String?) {
        if (finalName == null || IdentityRegistry.isGeneric(finalName)) return
        val faces = registry.activeFacesSnapshot()
        if (decision.speakerId in faces.values) return

        var target: String? = decision.asdPersonId?.takeIf { it in faces.values } ?: return
        var current = registry.nameOf(target!!).orEmpty()

        // Extra guard: do not rename if the face tracker already confirmed another real identity.
        val targetTrack = faces.entries.firstOrNull { it.value == target }?.key
        if (face != null && targetTrack != null) {
            val confirmed = face.trackName(targetTrack)
            if (confirmed != null && !IdentityRegistry.isGeneric(confirmed) && confirmed != finalName) target = null
        }
        // Fallback: ASD pointed at the wrong face, so look for a visible generic face.
        if (target == null && audioGender != null) {
            for ((_, pid) in faces) {
                val name = registry.nameOf(pid).orEmpty()
                if (IdentityRegistry.isGeneric(name) && pid != decision.speakerId) {
                    target = pid
                    current = name
                    break
                }
            }
        }

        val chosen = target ?: return
        if (!IdentityRegistry.isGeneric(current)) return
        registry.setName(chosen, finalName)
        registry.bind(finalName, chosen)
        if (face != null && current.isNotEmpty() && current != "Unknown") {
            val track = faces.entries.firstOrNull { it.value == chosen }?.key
            face.renamePerson(current, finalName, track)
        }
        onFaceRenamed(chosen, current, finalName)
    }

    /** Near-duplicate suppression: padded short segments can yield the same text more than once within seconds. */
    private fun appendTranscript(entry: TranscriptEntry) {
        synchronized(entries) {
            val norm = { t: String -> t.lowercase().replace(NON_WORD, "").trim() }
            for (previous in entries.takeLast(6).asReversed()) {
                if (entry.timeSec - previous.timeSec <= DEDUP_WINDOW_SEC &&
                    TextSimilarity.ratio(norm(entry.text), norm(previous.text)) >= DEDUP_RATIO
                ) return
            }
            entries += entry
        }
        onTranscript(entry)
    }

    companion object {
        private const val TAG = "FusionEngine"
        private const val SILENCE_PEAK = 0.01f
        private const val MIN_TURN_SEC = 0.5
        private const val MIN_SEGMENT_SEC = 2.0
        private const val SAME_VOICE_SIM = 0.40f
        private const val ASD_PAD_SEC = 0.4
        private const val DEDUP_WINDOW_SEC = 5.0
        private const val DEDUP_RATIO = 0.80
        private val NON_WORD = Regex("[^\\p{L}\\p{N}_\\s]")
    }
}
