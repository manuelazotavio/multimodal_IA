package com.avtracker.mobile.voice

import com.avtracker.mobile.audio.AudioUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.float
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Runs the real exported ecapa_voxceleb.onnx through the Kotlin VoiceEmbedder (desktop ONNX Runtime)
 * and compares with the speechbrain reference. Skipped when the model has not been exported.
 */
class VoiceEmbedderModelTest {

    private val model = File("src/main/assets/models/ecapa_voxceleb.onnx")

    /** Same deterministic signal as scratch make_ecapa_ref.py: three sines plus LCG noise. */
    private fun syntheticSignal(n: Int): FloatArray {
        var state = 12345L
        return FloatArray(n) { i ->
            state = (state * 1664525L + 1013904223L) and 0xFFFFFFFFL
            val noise = ((state shr 8).toDouble() / (1 shl 24) - 0.5) * 0.04
            val t = i.toDouble()
            val x = 0.10 * sin(2 * PI * 220 * t / 16000) +
                0.06 * sin(2 * PI * 1375 * t / 16000 + 0.5) +
                0.03 * sin(2 * PI * 3100 * t / 16000 + 1.1)
            (x + noise).toFloat()
        }
    }

    @Test
    fun embedding_matchesSpeechbrainReference() {
        assumeTrue("ecapa_voxceleb.onnx not exported (run scripts/export_models.py)", model.exists())
        val reference = Json.parseToJsonElement(File("src/test/resources/ecapa_reference.json").readText()).jsonObject
        val expected = reference.getValue("embedding").jsonArray.map { it.jsonPrimitive.float }.toFloatArray()

        VoiceEmbedder.fromFile(model).use { embedder ->
            val actual = assertNotNull(embedder.embed(syntheticSignal(reference.getValue("samples").jsonPrimitive.content.toInt())))

            assertEquals(VoiceEmbedder.DIMENSION, actual.size)
            // Weights are stored as fp16 (halves the APK size); on embeddings of norm ~300 that costs ~0.1 per component.
            assertTrue(AudioUtils.cosine(actual, expected) > 0.99999f, "cosine=${AudioUtils.cosine(actual, expected)}")
            val maxDiff = actual.indices.maxOf { abs(actual[it] - expected[it]) }
            assertTrue(maxDiff < 0.5f, "max abs diff $maxDiff")
        }
    }

    @Test
    fun embedNormalized_isUnitNormAndRejectsShortAudio() {
        assumeTrue("ecapa_voxceleb.onnx not exported (run scripts/export_models.py)", model.exists())

        VoiceEmbedder.fromFile(model).use { embedder ->
            val unit = assertNotNull(embedder.embedNormalized(syntheticSignal(16_000)))
            assertEquals(1f, AudioUtils.dot(unit, unit), 1e-4f)
            assertNull(embedder.embedNormalized(syntheticSignal(4_000))) // < 0.4 s, like _voice_embedding
        }
    }
}
