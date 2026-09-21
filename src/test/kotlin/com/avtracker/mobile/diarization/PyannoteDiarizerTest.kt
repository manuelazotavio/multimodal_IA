package com.avtracker.mobile.diarization

import com.avtracker.mobile.TestAudio
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The whole diarization pipeline against pyannote/speaker-diarization-3.1 (scripts/parity/diarization_reference.py): the
 * same speaker count on every frame, the same discrete diarization, the same turns and the same speaker labels.
 */
class PyannoteDiarizerTest {
    private val segmentationFile = File("src/main/assets/models/pyannote_segmentation.onnx")
    private val embeddingFile = File("src/main/assets/models/wespeaker_resnet34.onnx")
    private val ready get() = segmentationFile.exists() && embeddingFile.exists()

    private val cases: List<JsonObject> by lazy {
        Json.parseToJsonElement(File("src/test/resources/diarization/expected.json").readText())
            .jsonObject.getValue("pipeline").jsonArray.map { it.jsonObject }
    }

    @Test
    fun turnsCountAndLabelsMatchPyannoteOnEveryClip() {
        assumeTrue("models not exported (run scripts/export_models.py and export_wespeaker.py)", ready)
        val failures = StringBuilder()

        for (case in cases) {
            val name = case.getValue("name").jsonPrimitive.content
            val numSpeakers = case.getValue("num_speakers").let { if (it is JsonNull) null else it.jsonPrimitive.int }
            val audio = TestAudio.readWav("diarization/$name.wav")

            val result = PyannoteDiarizer(Diarizer.fromFile(segmentationFile), WespeakerEmbedder.fromFile(embeddingFile), numSpeakers).use { it.diarize(audio) }

            val expectedTurns = case.getValue("turns").jsonArray.map { t ->
                Triple(t.jsonArray[0].jsonPrimitive.double, t.jsonArray[1].jsonPrimitive.double, t.jsonArray[2].jsonPrimitive.content.removePrefix("SPEAKER_").toInt())
            }
            val problems = ArrayList<String>()

            val expectedCount = case.getValue("count")?.takeIf { it !is JsonNull }?.jsonArray?.map { it.jsonArray[0].jsonPrimitive.int }
            if (expectedCount != null && expectedCount != result.count.toList()) {
                val diff = expectedCount.indices.count { it >= result.count.size || expectedCount[it] != result.count[it] }
                problems += "speaker count differs on $diff of ${expectedCount.size} frames"
            }

            if (expectedTurns.size != result.turns.size) {
                problems += "turn count: pyannote ${expectedTurns.size}, kotlin ${result.turns.size}"
            } else {
                for ((i, e) in expectedTurns.withIndex()) {
                    val a = result.turns[i]
                    if (abs(e.first - a.startSec) > 1e-6 || abs(e.second - a.endSec) > 1e-6 || e.third != a.slot) {
                        problems += "turn $i: pyannote $e, kotlin (${a.startSec}, ${a.endSec}, ${a.slot})"
                    }
                }
            }
            if (problems.isNotEmpty()) failures.append("$name: ${problems.take(4).joinToString("; ")}\n  pyannote: $expectedTurns\n  kotlin:   ${result.turns.map { Triple(it.startSec, it.endSec, it.slot) }}\n")
        }
        assertTrue(failures.isEmpty(), "diarization differs from pyannote:\n$failures")
    }

    @Test
    fun anEmptyOrSilentInputHasNoTurns() {
        assumeTrue("models not exported", ready)
        PyannoteDiarizer(Diarizer.fromFile(segmentationFile), WespeakerEmbedder.fromFile(embeddingFile)).use { diarizer ->
            assertEquals(emptyList(), diarizer.turns(FloatArray(0)))
            assertEquals(emptyList(), diarizer.turns(FloatArray(16_000 * 3)))
        }
    }

    @Test
    fun framesAreTenSecondsOverFiveHundredEightyNine() {
        assertEquals(0, PyannoteDiarizer.closestFrame(0.0))
        assertEquals(588, PyannoteDiarizer.closestFrame(10.0))
        assertEquals(58, PyannoteDiarizer.closestFrame(1.0))
        assertEquals(JsonArray(emptyList()), JsonArray(emptyList()))
    }
}
