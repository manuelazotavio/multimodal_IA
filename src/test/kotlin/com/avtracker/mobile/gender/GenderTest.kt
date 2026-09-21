package com.avtracker.mobile.gender

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import com.avtracker.mobile.TestAudio.readWav
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class GenderTest {

    // ---- names ------------------------------------------------------------------------------

    @Test
    fun genderOf_usesTheFirstNameIgnoringAccentsAndCase() {
        assertEquals("male", NameGender.genderOf("João Silva"))
        assertEquals("male", NameGender.genderOf("joao"))
        assertEquals("female", NameGender.genderOf("Manuela"))
        assertEquals("female", NameGender.genderOf("ISABEL Souza"))
        assertNull(NameGender.genderOf("Kauã Xavier"))     // unknown first name
        assertNull(NameGender.genderOf(""))
        assertNull(NameGender.genderOf(null))
    }

    @Test
    fun correctForGender_onlyTouchesTheFirstNameOfAFemaleVoice() {
        assertEquals("Manuela", NameGender.correctForGender("Manoel", "female"))
        assertEquals("Paula Otavio", NameGender.correctForGender("Paulo Otavio", "female"), "surname untouched")
        assertEquals("Marca", NameGender.correctForGender("Marco", "female"), "generic -o -> -a rule")
        assertEquals("Isabel", NameGender.correctForGender("Isabel", "female"))
        assertEquals("Manoel", NameGender.correctForGender("Manoel", "male"))
        assertEquals("Manoel", NameGender.correctForGender("Manoel", null))
    }

    // ---- pitch: parity with librosa.yin -----------------------------------------------------

    private fun harmonic(f0: Double, n: Int = 48000): FloatArray {
        var state = 777L
        return FloatArray(n) { i ->
            state = (state * 1664525L + 1013904223L) and 0xFFFFFFFFL
            val noise = ((state shr 8).toDouble() / (1 shl 24) - 0.5) * 0.02
            val t = i / 16000.0
            var x = 0.0
            for ((h, a) in listOf(1 to 0.30, 2 to 0.15, 3 to 0.08, 4 to 0.04)) x += a * sin(2 * PI * f0 * h * t)
            (x + noise).toFloat()
        }
    }

    private val reference = Json.parseToJsonElement(File("src/test/resources/yin_reference.json").readText()).jsonObject

    private fun clip(name: String) = when (name) {
        "tts_en" -> readWav("tts_en.wav")
        "tts_pt" -> readWav("tts_pt.wav")
        "harm_110" -> harmonic(110.0)
        "harm_148" -> harmonic(148.0)
        "harm_220" -> harmonic(220.0)
        else -> error(name)
    }

    @Test
    fun yin_matchesLibrosaFrameByFrame() {
        for (name in reference.keys) {
            val expected = reference.getValue(name).jsonObject.getValue("f0").jsonArray.map { it.jsonPrimitive.content.toDouble() }
            val actual = PitchGender.yin(clip(name), 50.0, 500.0, 16_000)

            assertEquals(expected.size, actual.size, "$name: frame count")
            val worst = expected.indices.maxOf { abs(actual[it] - expected[it]) / expected[it] }
            assertEquals(0.0, worst, 2e-3, "$name: worst relative f0 error")
        }
    }

    @Test
    fun detect_givesTheSameGenderAsThePythonPipeline() {
        for (name in reference.keys) {
            val expected = reference.getValue(name).jsonObject.getValue("gender").let { if (it is JsonNull) null else it.jsonPrimitive.content }
            assertEquals(expected, PitchGender.detect(clip(name)), name)
        }
    }

    @Test
    fun detect_handlesSilenceWithoutCrashing() {
        // Whatever librosa says about a silent frame, it must not throw and must not claim a strong gender by accident.
        PitchGender.detect(FloatArray(16_000))
    }
}
