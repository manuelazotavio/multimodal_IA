package com.avtracker.mobile.asd

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
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

/** Runs the real exported light_asd.onnx through the Kotlin wrapper and compares with the original PyTorch + librosa reference. */
class LightAsdModelTest {

    private val model = File("src/main/assets/models/light_asd.onnx")

    private fun syntheticSignal(n: Int): FloatArray {
        var state = 12345L
        return FloatArray(n) { i ->
            state = (state * 1664525L + 1013904223L) and 0xFFFFFFFFL
            val noise = ((state shr 8).toDouble() / (1 shl 24) - 0.5) * 0.04
            val t = i.toDouble()
            (0.10 * sin(2 * PI * 220 * t / 16000) +
                0.06 * sin(2 * PI * 1375 * t / 16000 + 0.5) +
                0.03 * sin(2 * PI * 3100 * t / 16000 + 1.1) + noise).toFloat()
        }
    }

    private fun syntheticCrops(): List<ByteArray> = List(25) { f ->
        ByteArray(112 * 112) { i -> ((i % 112) * 3 + (i / 112) * 5 + f * 7).toByte() } // (x*3 + y*5 + f*7) % 256
    }

    @Test
    fun scores_matchTheOriginalPipeline() {
        assumeTrue("light_asd.onnx not exported (run scripts/export_models.py)", model.exists())
        val reference = Json.parseToJsonElement(File("src/test/resources/light_asd_reference.json").readText()).jsonObject
        val expected = reference.getValue("scores").jsonArray.map { it.jsonPrimitive.float }

        LightAsdModel.fromFile(model).use { asd ->
            val actual = assertNotNull(asd.score(syntheticCrops(), syntheticSignal(16_000)))

            assertEquals(25, actual.size)
            val maxDiff = actual.indices.maxOf { abs(actual[it] - expected[it]) }
            assertTrue(maxDiff < 1e-4f, "max abs diff $maxDiff")
        }
    }

    @Test
    fun shorterAudioIsZeroPaddedInsideTheGraphAndRejectsBadInput() {
        assumeTrue("light_asd.onnx not exported (run scripts/export_models.py)", model.exists())

        LightAsdModel.fromFile(model).use { asd ->
            assertEquals(25, assertNotNull(asd.score(syntheticCrops(), syntheticSignal(15_200))).size)
            assertNull(asd.score(syntheticCrops().take(24), syntheticSignal(16_000))) // needs exactly 25 crops
            assertNull(asd.score(syntheticCrops(), syntheticSignal(400)))             // < 50 ms of audio
        }
    }
}
