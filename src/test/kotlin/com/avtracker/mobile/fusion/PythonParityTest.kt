package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.diarization.SpeakerTurn
import com.avtracker.mobile.voice.InMemoryVoiceStore
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.voice.StoredVoice
import com.avtracker.mobile.whisper.TranscriptionResult
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.File
import kotlin.test.Test
import kotlin.test.fail

/**
 * Differential test against the ORIGINAL Python code. scripts/parity/engine_parity.py runs av-tracker's own
 * RealtimeTranscriber / MultiSpeakerVerifier (only the neural models are stubbed) over randomised scenarios and records
 * the resulting state after every step; this replays the same scenarios through the Kotlin engine and requires the same
 * decisions, names, identities, saved voices and reports.
 */
class PythonParityTest {

    // ---- the stubs, mirroring engine_parity.py ------------------------------------------------------------------------

    /** Voice identity = most common run length of equal-sign samples (a square wave of period 2*code). */
    private fun voiceCode(audio: FloatArray): Int {
        val runs = ArrayList<Int>()
        var n = 1
        for (i in 1 until audio.size) {
            if ((audio[i] < 0) == (audio[i - 1] < 0)) n++ else { runs += n; n = 1 }
        }
        runs += n
        val interior = if (runs.size > 2) runs.subList(1, runs.size - 1) else runs
        val counts = HashMap<Int, Int>()
        interior.forEach { counts.merge(it, 1, Int::plus) }
        val best = counts.values.max()
        return counts.filter { it.value == best }.keys.min()
    }

    private fun makeAudio(code: Int, seconds: Double): FloatArray {
        val n = (seconds * AudioUtils.SAMPLE_RATE).toInt()
        return FloatArray(n) { if ((it / code) % 2 == 0) 0.2f else -0.2f }
    }

    private class Voices(val vecs: Map<Int, DoubleArray>, val genders: Map<Int, String?>, val code: (FloatArray) -> Int) : VoiceEncoder {
        /** Voice vector plus a deterministic per-segment noise, the same LCG as engine_parity.py's embedding_of. */
        private fun embedding(audio: FloatArray): FloatArray {
            val c = code(audio)
            var state = (audio.size.toLong() * 7919 + c) and 0xFFFFFFFFL
            val base = vecs.getValue(c)
            return FloatArray(base.size) { i ->
                state = (state * 1664525L + 1013904223L) and 0xFFFFFFFFL
                val noise = ((state shr 8).toDouble() / (1 shl 24) - 0.5) * 2.0
                (base[i] + 0.02 * noise).toFloat()
            }
        }

        override fun embed(audio: FloatArray): FloatArray? = if (audio.size < 1_600) null else embedding(audio)
        override fun embedNormalized(audio: FloatArray): FloatArray? =
            if (audio.size < AudioUtils.SAMPLE_RATE * 2 / 5) null else AudioUtils.l2Normalize(embedding(audio))
    }

    private class Asd : SpeakerActivity {
        var active: Int? = null
        var guess: Int? = null
        override fun getActiveSpeaker(timeStart: Double, timeEnd: Double) = active
        override fun getBestGuess(timeStart: Double, timeEnd: Double) = guess
    }

    // ---- scenario runner ------------------------------------------------------------------------------------------------

    private fun load(name: String): JsonArray? =
        File("src/test/resources/parity/$name").takeIf { it.exists() }?.let { Json.parseToJsonElement(it.readText()).jsonArray }

    private fun floats(e: JsonElement) = e.jsonArray.map { it.jsonPrimitive.content.toFloat() }.toFloatArray()

    private fun strList(l: List<String>) = JsonArray(l.map(::JsonPrimitive))

    private fun runScenario(sc: JsonObject) {
        val seed = sc.getValue("seed").jsonPrimitive.content
        val cfg = sc.getValue("config").jsonObject
        val numSpeakers = cfg.getValue("num_speakers").let { if (it is JsonNull) null else it.jsonPrimitive.intOrNull }
        val language = cfg.getValue("language").jsonPrimitive.content

        var epoch = 1_800_000_000.0
        val clock = { epoch }

        val store = InMemoryVoiceStore(sc.getValue("seed_files").jsonArray.map {
            StoredVoice(it.jsonObject.getValue("name").jsonPrimitive.content, floats(it.jsonObject.getValue("vec")).toList())
        })
        val verifier = SpeakerVerifier(store, 0.80f) { epoch }

        val voiceObjs = sc.getValue("voices").jsonObject
        val vecs = voiceObjs.entries.associate { (k, v) -> k.toInt() to v.jsonObject.getValue("vec").jsonArray.map { it.jsonPrimitive.content.toDouble() }.toDoubleArray() }
        val genders = voiceObjs.entries.associate { (k, v) ->
            k.toInt() to v.jsonObject.getValue("gender").let { if (it is JsonNull) null else it.jsonPrimitive.content }
        }
        val voice = Voices(vecs, genders, ::voiceCode)

        val queue = ArrayDeque<String>()
        var noSpeech = 0.01f
        var logProb = -0.2f
        val recognizer = SpeechRecognizer { _, _ ->
            TranscriptionResult(queue.removeFirstOrNull().orEmpty(), emptyList(), logProb, noSpeech)
        }

        var turns: List<SpeakerTurn> = emptyList()
        val asd = Asd()
        val registry = IdentityRegistry()
        var knownFaces = emptySet<String>()
        val renames = ArrayList<List<String?>>()
        val bridge = RegistryFaceBridge(registry, { knownFaces }) { old, new, track -> renames += listOf(old, new, track?.toString()) }

        val engine = FusionEngine(
            timeline = AudioTimeline(), turnDetector = TurnDetector { turns }, voice = voice, verifier = verifier,
            recognizer = recognizer, asd = asd, registry = registry,
            config = FusionConfig(language = language, numSpeakers = numSpeakers),
            entities = entities(sc), faceTracker = bridge, clock = clock, wallClockMillis = { (epoch * 1000).toLong() },
            genderDetector = { audio -> genders[voiceCode(audio)] }
        )

        var seenEntries = 0
        var seenMetrics = 0
        val steps = sc.getValue("steps").jsonArray
        val expected = sc.getValue("expected").jsonArray

        steps.forEachIndexed { index, stepElement ->
            val step = stepElement.jsonObject
            when (step.getValue("type").jsonPrimitive.content) {
                "faces" -> {
                    val active = step.getValue("active_faces").jsonObject.entries.associate { (k, v) -> k.toInt() to v.jsonPrimitive.content }
                    val faceNames = step.getValue("face_names").jsonObject.entries.associate { (k, v) -> k.toInt() to v.jsonPrimitive.content }
                    registry.setActiveFaces(active, faceNames)
                    step.getValue("set_names").jsonObject.forEach { (pid, name) -> registry.setName(pid, name.jsonPrimitive.content) }
                    step.getValue("bind").jsonObject.forEach { (name, pid) -> registry.bind(name, pid.jsonPrimitive.content) }
                    knownFaces = step.getValue("known_faces").jsonArray.map { it.jsonPrimitive.content }.toSet()
                }
                "segment" -> {
                    val t = step.getValue("t").jsonPrimitive.content.toDouble()
                    val seconds = step.getValue("seconds").jsonPrimitive.content.toDouble()
                    epoch = 1_800_000_000.0 + t
                    asd.active = step["asd"]?.let { if (it is JsonNull) null else it.jsonPrimitive.intOrNull }
                    asd.guess = step["guess"]?.let { if (it is JsonNull) null else it.jsonPrimitive.intOrNull }
                    noSpeech = step.getValue("no_speech").jsonPrimitive.content.toFloat()
                    logProb = step.getValue("logprob").jsonPrimitive.content.toFloat()
                    queue += step.getValue("text").jsonPrimitive.content
                    val audio = makeAudio(step.getValue("voice").jsonPrimitive.content.toInt(), seconds)
                    engine.attributeSegment(audio, audio, t, t + seconds)
                    queue.clear()
                }
                "chunk" -> {
                    val t = step.getValue("t").jsonPrimitive.content.toDouble()
                    epoch = 1_800_000_000.0 + t
                    asd.active = step["asd"]?.let { if (it is JsonNull) null else it.jsonPrimitive.intOrNull }
                    asd.guess = step["guess"]?.let { if (it is JsonNull) null else it.jsonPrimitive.intOrNull }
                    noSpeech = 0.01f
                    logProb = -0.2f
                    step.getValue("texts").jsonArray.forEach { queue += it.jsonPrimitive.content }
                    val parts = ArrayList<FloatArray>()
                    val list = ArrayList<SpeakerTurn>()
                    var pos = 0.0
                    for (p in step.getValue("pieces").jsonArray) {
                        val obj = p.jsonObject
                        val seconds = obj.getValue("seconds").jsonPrimitive.content.toDouble()
                        parts += makeAudio(obj.getValue("voice").jsonPrimitive.content.toInt(), seconds)
                        list += SpeakerTurn(pos, pos + seconds, obj.getValue("slot").jsonPrimitive.content.toInt())
                        pos += seconds
                    }
                    turns = list
                    val audio = FloatArray(parts.sumOf { it.size })
                    var offset = 0
                    for (part in parts) { System.arraycopy(part, 0, audio, offset, part.size); offset += part.size }
                    engine.processChunk(audio, t, step.getValue("zone").jsonPrimitive.content.toInt())
                    queue.clear()
                }
            }

            val entries = engine.transcript
            val metrics = engine.log.segmentMetrics
            val actual = buildJsonObject {
                put("appended", JsonArray(entries.drop(seenEntries).map {
                    buildJsonObject {
                        put("speaker", it.speakerId)
                        put("name", it.speakerName?.let(::JsonPrimitive) ?: JsonNull)
                        put("text", it.text)
                    }
                }))
                put("decisions", strList(metrics.drop(seenMetrics).map { it.getValue("decision").jsonPrimitive.content }))
                put("fn", JsonArray(metrics.drop(seenMetrics).map { it.getValue("fn_reasons") }))
                put("fp", JsonArray(metrics.drop(seenMetrics).map { JsonArray(listOf(it.getValue("fp_risk"), it.getValue("fp_reasons"))) }))
                put("names", JsonObject(registry.namesSnapshot().toSortedMap().mapValues { JsonPrimitive(it.value) }))
                put("emb_to_pid", JsonObject(registry.bindingsSnapshot().toSortedMap().mapValues { JsonPrimitive(it.value) }))
                put("identified", strList(engine.book.identifiedSpeakers.sorted()))
                put("session_to_face", JsonObject(engine.book.sessionToFace.toSortedMap().mapValues { JsonPrimitive(it.value) }))
                put("context", strList(engine.book.contextNames.sorted()))
                put("verifier", strList(verifier.names.sorted()))
                put("files", strList(store.baseNames().sorted()))
                put("renames", JsonArray(renames.map { r -> JsonArray(r.map { v -> v?.let(::JsonPrimitive) ?: JsonNull }) }))
                put("session_ids", strList(engine.sessionTracker.ids.sorted()))
            }
            seenEntries = entries.size
            seenMetrics = metrics.size

            val want = expected[index].jsonObject
            for (key in want.keys) {
                if (normalize(want.getValue(key)) != normalize(actual.getValue(key))) {
                    fail(
                        "scenario seed=$seed step=$index (${step.getValue("type").jsonPrimitive.content}) differs in '$key'\n" +
                            "  python: ${want.getValue(key)}\n  kotlin: ${actual.getValue(key)}\n  step: $step"
                    )
                }
            }
        }
    }

    /** Numbers compare as doubles (Python prints 0.5 where Kotlin may print 0.50). */
    private fun normalize(e: JsonElement): JsonElement = when (e) {
        is JsonPrimitive -> e.doubleOrNull?.let { JsonPrimitive(it) } ?: e
        is JsonArray -> JsonArray(e.map(::normalize))
        is JsonObject -> JsonObject(e.mapValues { normalize(it.value) })
        else -> e
    }

    /** Scenarios generated with --ner ran the original with real spaCy; replay them with the Kotlin NER port. */
    private fun entities(sc: JsonObject): EntityExtractor {
        if (sc["ner"]?.jsonPrimitive?.content != "true") return NoEntities
        val lang = sc.getValue("config").jsonObject.getValue("language").jsonPrimitive.content
        return com.avtracker.mobile.ner.SpacyNer.fromFolder(File("src/main/assets/ner/$lang"))
    }

    @Test
    fun kotlinEngineMatchesThePythonOriginalOnEveryScenario() {
        for (sc in load("engine_scenarios.json").orEmpty()) runScenario(sc.jsonObject)
    }

    @Test
    fun kotlinEngineWithTheNerPortMatchesThePythonOriginalWithRealSpacy() {
        org.junit.Assume.assumeTrue("NER scenarios not generated (engine_parity.py --ner)", File("src/test/resources/parity/engine_scenarios_ner.json").exists())
        org.junit.Assume.assumeTrue("NER assets not exported (run scripts/export_ner.py)", File("src/main/assets/ner/en/model.json").exists())
        for (sc in load("engine_scenarios_ner.json").orEmpty()) runScenario(sc.jsonObject)
    }

    @Suppress("unused")
    private fun unusedContentOrNull(e: JsonElement) = e.jsonPrimitive.contentOrNull
}
