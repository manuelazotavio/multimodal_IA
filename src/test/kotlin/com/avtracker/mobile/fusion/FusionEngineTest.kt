package com.avtracker.mobile.fusion

import com.avtracker.mobile.asd.ActiveSpeakerDetector
import com.avtracker.mobile.asd.FaceSample
import com.avtracker.mobile.asd.GrayImage
import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.diarization.SpeakerTurn
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.whisper.TranscriptionResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FusionEngineTest {

    private val sr = AudioUtils.SAMPLE_RATE

    /** Voices are identified by the sign of the audio, which survives the engine's loudness normalisation. */
    private class SignVoice : VoiceEncoder {
        override fun embed(audio: FloatArray): FloatArray? {
            if (audio.size < 1_600) return null
            return if (audio.average() >= 0) floatArrayOf(1f, 0f, 0f) else floatArrayOf(0f, 1f, 0f)
        }
        override fun embedNormalized(audio: FloatArray): FloatArray? = if (audio.size < sr2 * 2 / 5) null else embed(audio)
        companion object { const val sr2 = 16_000 }
    }

    private class Script(texts: List<String>, private val logProb: Float = -0.2f) : SpeechRecognizer {
        private val queue = ArrayDeque(texts)
        var calls = 0
        override fun transcribe(audio: FloatArray, language: String): TranscriptionResult? {
            calls++
            val text = queue.removeFirstOrNull() ?: return null
            return TranscriptionResult(text, emptyList(), logProb, 0.01f)
        }
    }

    private fun tone(seconds: Double, value: Float) = FloatArray((seconds * sr).toInt()) { value }

    private var now = 1_000.0
    private val registry = IdentityRegistry()

    private fun engine(
        texts: List<String>,
        turns: List<SpeakerTurn> = emptyList(),
        verifier: SpeakerVerifier = SpeakerVerifier.fromMap(emptyMap()),
        asd: ActiveSpeakerDetector? = null,
        numSpeakers: Int? = null,
        faceBridge: FaceTrackerBridge? = RegistryFaceBridge(registry),
        out: MutableList<TranscriptEntry> = mutableListOf(),
        recognizer: Script = Script(texts)
    ) = FusionEngine(
        timeline = AudioTimeline(),
        turnDetector = TurnDetector { turns },
        voice = SignVoice(),
        verifier = verifier,
        recognizer = recognizer,
        asd = asd,
        registry = registry,
        config = FusionConfig(numSpeakers = numSpeakers),
        faceTracker = faceBridge,
        onTranscript = { out += it },
        clock = { now }
    )

    // ---- processSegment ---------------------------------------------------------------------

    @Test
    fun aKnownVoiceIsNamedByTheVerifier() {
        val engine = engine(listOf("Let us begin the meeting now."), verifier = SpeakerVerifier.fromMap(mapOf("Alice" to floatArrayOf(1f, 0f, 0f))))

        engine.processSegment(tone(3.0, 0.1f), null, 10.0, 13.0)

        val entry = engine.transcript.single()
        assertEquals("(Alice)", entry.label)
        assertTrue(entry.decision.startsWith("VERIFIER_HIGH"))
    }

    @Test
    fun anUnknownVoiceGetsAStableSessionLabel() {
        val engine = engine(listOf("Hello there my friends.", "And then we continued talking."))

        engine.processSegment(tone(3.0, -0.1f), null, 10.0, 13.0)
        now += 10
        engine.processSegment(tone(3.0, -0.1f), null, 20.0, 23.0)

        val labels = engine.transcript.map { it.label }
        assertEquals(2, labels.size)
        assertEquals(labels[0], labels[1])
        assertTrue(labels[0].startsWith("(spk_"))
    }

    @Test
    fun hallucinationsNeverReachTheTranscript() {
        val engine = engine(listOf("Thank you. Thank you. Thank you.", "Subtitles by the Amara.org community"))

        engine.processSegment(tone(3.0, 0.1f), null, 10.0, 13.0)
        engine.processSegment(tone(3.0, 0.1f), null, 20.0, 23.0)

        assertTrue(engine.transcript.isEmpty())
    }

    @Test
    fun aRepeatedTextWithinFiveSecondsIsDroppedButLaterOnesAreKept() {
        val engine = engine(listOf("We should ship this on Friday.", "We should ship this on friday!", "We should ship this on Friday."))

        engine.processSegment(tone(3.0, 0.1f), null, 10.0, 13.0)
        now += 2
        engine.processSegment(tone(3.0, 0.1f), null, 12.0, 15.0)   // near-duplicate, 2 s later
        now += 30
        engine.processSegment(tone(3.0, 0.1f), null, 45.0, 48.0)   // same text, much later

        assertEquals(2, engine.transcript.size)
    }

    // ---- face + voice fusion ----------------------------------------------------------------

    /** A face whose mouth region flickers (speaking) next to one that stays still. */
    private fun speakingScene(asd: ActiveSpeakerDetector, speakingTrack: Int, otherTrack: Int, from: Double, to: Double) {
        fun face(mouth: Int) = GrayImage(40, 40, ByteArray(1600) { i -> (if (i / 40 >= 24) mouth else 100).toByte() })
        var t = from
        var frame = 0
        while (t <= to) {
            asd.update(
                listOf(
                    FaceSample(speakingTrack, null, face(if (frame % 2 == 0) 30 else 220)),
                    FaceSample(otherTrack, null, face(90))
                ),
                t
            )
            t += 0.1
            frame++
        }
    }

    @Test
    fun aFaceThatIsSpeakingLendsItsNameToAnUnknownVoice_andTheVoiceIsRecognisedNextTime() {
        val asd = ActiveSpeakerDetector(lightAsd = null, clock = { now })
        val verifier = SpeakerVerifier.fromMap(emptyMap(), nowSec = { now })
        val engine = engine(listOf("The report is almost ready.", "I will send it tomorrow morning."), verifier = verifier, asd = asd)

        // Two faces on screen: Bob (recognised by the face tracker) is the one whose mouth moves.
        val sync = FaceIdentitySync(registry)
        sync.sync(listOf(FaceResult(5, "Bob", 0.9f), FaceResult(6, "Person_6", 0.8f)))
        speakingScene(asd, speakingTrack = 5, otherTrack = 6, from = 10.0, to = 14.0)

        engine.processSegment(tone(4.0, -0.1f), null, 10.0, 14.0)

        val first = engine.transcript.single()
        assertTrue(first.decision.startsWith("ASD"), first.decision)
        assertEquals("(Bob)", first.label)
        assertTrue("Bob" in verifier, "face-voice binding should have created Bob's voice profile")

        // Bob is off camera now; his voice alone is enough.
        registry.setActiveFaces(emptyMap())
        now += 30
        engine.processSegment(tone(4.0, -0.1f), null, 60.0, 64.0)

        val second = engine.transcript.last()
        assertTrue(second.decision.startsWith("VERIFIER_HIGH"), second.decision)
        assertEquals("(Bob)", second.label)
    }

    @Test
    fun aVoiceProfileCreatedFromAGenericFaceIsNotAllowed() {
        val asd = ActiveSpeakerDetector(lightAsd = null, clock = { now })
        val verifier = SpeakerVerifier.fromMap(emptyMap(), nowSec = { now })
        val engine = engine(listOf("Nobody knows who this person is."), verifier = verifier, asd = asd)

        FaceIdentitySync(registry).sync(listOf(FaceResult(5, "Person_5", 0.9f), FaceResult(6, "Person_6", 0.8f)))
        speakingScene(asd, speakingTrack = 5, otherTrack = 6, from = 10.0, to = 14.0)
        engine.processSegment(tone(4.0, -0.1f), null, 10.0, 14.0)

        assertEquals(0, verifier.size, "an anonymous face must not create a named voice profile")
    }

    // ---- processChunk -----------------------------------------------------------------------

    @Test
    fun silentChunksAreSkippedWithoutRunningAnyModel() {
        val script = Script(listOf("should never be used"))
        val engine = engine(emptyList(), recognizer = script)

        engine.processChunk(FloatArray(8 * sr) { 0.001f }, 100.0)

        assertEquals(0, script.calls)
    }

    @Test
    fun twoSpeakersInOneChunkBecomeTwoAttributedSegments() {
        val turns = listOf(SpeakerTurn(0.0, 3.0, slot = 0), SpeakerTurn(3.0, 6.0, slot = 1))
        val engine = engine(listOf("First speaker talks for a bit.", "Second speaker answers back."), turns)
        val audio = tone(3.0, 0.1f) + tone(3.0, -0.1f)

        engine.processChunk(audio, 100.0)

        val entries = engine.transcript
        assertEquals(2, entries.size)
        assertTrue(entries[0].label != entries[1].label, "different voices must get different session ids: ${entries.map { it.label }}")
    }

    @Test
    fun consecutiveTurnsOfTheSameSlotAndVoiceMergeIntoOneSegment() {
        val turns = listOf(SpeakerTurn(0.0, 2.0, slot = 0), SpeakerTurn(2.5, 5.0, slot = 0))
        val script = Script(listOf("One long thought split by a breath."))
        val engine = engine(emptyList(), turns, recognizer = script)

        engine.processChunk(tone(6.0, 0.1f), 100.0)

        assertEquals(1, script.calls)
    }

    @Test
    fun aSlotThatChangesVoiceIsSplitEvenWithTheSameLabel() {
        val turns = listOf(SpeakerTurn(0.0, 3.0, slot = 0), SpeakerTurn(3.0, 6.0, slot = 0))
        val script = Script(listOf("Voice number one speaking.", "Voice number two speaking."))
        val engine = engine(emptyList(), turns, recognizer = script)

        engine.processChunk(tone(3.0, 0.1f) + tone(3.0, -0.1f), 100.0)

        assertEquals(2, script.calls)
    }

    @Test
    fun onlyTheNewZoneIsTranscribedAndTurnsInTheContextAreIgnored() {
        val turns = listOf(SpeakerTurn(1.0, 4.0, slot = 0), SpeakerTurn(11.0, 14.0, slot = 1))
        val script = Script(listOf("Only the recent speech counts."))
        val engine = engine(emptyList(), turns, recognizer = script)

        // 15 s window whose first 10 s are context (zone starts at 10 s): the turn at 1-4 s was handled by a previous chunk.
        engine.processChunk(tone(15.0, 0.1f), 100.0, newZoneStartSamples = 10 * sr)

        assertEquals(1, script.calls)
    }

    @Test
    fun withoutAnyTurnTheNewZoneIsTranscribedAsOneSegment() {
        val script = Script(listOf("No diarization but plenty of speech."))
        val engine = engine(emptyList(), emptyList(), recognizer = script)

        engine.processChunk(tone(8.0, 0.1f), 100.0)

        assertEquals(1, script.calls)
        assertEquals(1, engine.transcript.size)
    }

    @Test
    fun shortTurnsAreExtendedWithSurroundingAudioForWhisper() {
        val received = mutableListOf<Int>()
        val turns = listOf(SpeakerTurn(5.0, 5.8, slot = 0))
        val recognizer = SpeechRecognizer { audio, _ -> received += audio.size; TranscriptionResult("Short but meaningful.", emptyList(), -0.2f, 0.01f) }
        val engine = FusionEngine(
            timeline = AudioTimeline(), turnDetector = TurnDetector { turns }, voice = SignVoice(),
            verifier = SpeakerVerifier.fromMap(emptyMap()), recognizer = recognizer, asd = null,
            registry = registry, clock = { now }
        )

        engine.processChunk(tone(10.0, 0.1f), 100.0)

        assertEquals(1, received.size)
        assertEquals(2 * sr, received[0], "0.8 s turn padded to the 2 s minimum")
    }
}
