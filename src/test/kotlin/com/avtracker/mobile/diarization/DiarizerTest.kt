package com.avtracker.mobile.diarization

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
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
import kotlin.test.assertTrue

class DiarizerTest {

    private val step = Diarizer.FRAME_STEP_SEC

    /** Builds log-prob rows where each frame's chosen powerset class gets 0 and the rest -10. */
    private fun rows(vararg runs: Pair<Int, Int>): Array<FloatArray> = runs.flatMap { (cls, count) ->
        List(count) { FloatArray(7) { c -> if (c == cls) 0f else -10f } }
    }.toTypedArray()

    private fun frames(seconds: Double) = (seconds / step).toInt()

    @Test
    fun silenceProducesNoTurns() {
        assertTrue(Diarizer.decodeTurns(rows(0 to 600)).isEmpty())
    }

    @Test
    fun oneSpeakerRunBecomesOneTurn() {
        val turns = Diarizer.decodeTurns(rows(0 to 30, 1 to frames(2.0), 0 to 30))

        assertEquals(1, turns.size)
        assertEquals(0, turns[0].slot)
        assertEquals(30 * step, turns[0].startSec, 1e-9)
        assertEquals((30 + frames(2.0)) * step, turns[0].endSec, 1e-9)
    }

    @Test
    fun shortPausesInsideOneSpeakersTurnAreFilled_longPausesSplitIt() {
        val short = Diarizer.decodeTurns(rows(1 to frames(1.0), 0 to frames(0.3), 1 to frames(1.0)))
        assertEquals(1, short.size, "0.3 s pause < 0.5 s gap fill -> a single turn")

        val long = Diarizer.decodeTurns(rows(1 to frames(1.0), 0 to frames(0.8), 1 to frames(1.0)))
        assertEquals(2, long.size, "0.8 s pause >= 0.5 s -> two turns")
    }

    @Test
    fun turnsShorterThanHalfASecondAreDropped() {
        assertTrue(Diarizer.decodeTurns(rows(1 to frames(0.3), 0 to 100)).isEmpty())
    }

    @Test
    fun twoSpeakersTakeTurnsAndOverlapIsReportedForBoth() {
        // slot 0 talks, then slot 1, with a stretch where both talk (powerset class 4 = {0,1}).
        val alone = frames(2.0)
        val overlap = frames(1.0)
        val turns = Diarizer.decodeTurns(rows(1 to alone, 4 to overlap, 2 to alone))

        val slot0 = turns.single { it.slot == 0 }
        val slot1 = turns.single { it.slot == 1 }
        assertEquals(0.0, slot0.startSec, 1e-9)
        assertEquals((alone + overlap) * step, slot0.endSec, 1e-9)         // 2 s alone + 1 s overlap
        assertEquals(alone * step, slot1.startSec, 1e-9)                   // joins during the overlap
        assertEquals((alone + overlap + alone) * step, slot1.endSec, 1e-9)
        assertEquals(listOf(0, 1), turns.map { it.slot }) // sorted by start
    }

    // ---- real model ---------------------------------------------------------------------------

    private val model = File("src/main/assets/models/pyannote_segmentation.onnx")

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

    @Test
    fun model_matchesPyannoteReferenceAndDetectsNoSpeechInATone() {
        assumeTrue("pyannote_segmentation.onnx not exported (run scripts/export_models.py)", model.exists())
        val reference = Json.parseToJsonElement(File("src/test/resources/segmentation_reference.json").readText()).jsonObject
        val expectedFirst = reference.getValue("first_rows").jsonArray.map { row -> row.jsonArray.map { it.jsonPrimitive.float } }

        Diarizer.fromFile(model).use { diarizer ->
            val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
            val session = env.createSession(model.absolutePath, ai.onnxruntime.OrtSession.SessionOptions())
            val audio = syntheticSignal(reference.getValue("samples").jsonPrimitive.int)

            val tensor = ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(audio), longArrayOf(1, 1, audio.size.toLong()))
            @Suppress("UNCHECKED_CAST")
            val logProbs = tensor.use { session.run(mapOf("waveform" to it)).use { r -> (r[0].value as Array<Array<FloatArray>>)[0] } }
            session.close()

            assertEquals(reference.getValue("frames").jsonPrimitive.int, logProbs.size)
            var maxDiff = 0f
            expectedFirst.forEachIndexed { f, row -> row.forEachIndexed { c, v -> maxDiff = maxOf(maxDiff, abs(v - logProbs[f][c])) } }
            assertTrue(maxDiff < 1e-3f, "max log-prob diff $maxDiff")

            assertTrue(diarizer.turns(audio).isEmpty(), "a synthetic tone is not speech")
            assertTrue(diarizer.turns(FloatArray(800)).isEmpty(), "audio shorter than 0.1 s yields nothing")
        }
    }
}
