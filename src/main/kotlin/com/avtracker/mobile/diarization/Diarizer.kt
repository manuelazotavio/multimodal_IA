package com.avtracker.mobile.diarization

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.fusion.TurnDetector
import com.avtracker.mobile.voice.AssetFiles
import java.io.File
import java.nio.FloatBuffer

/** A stretch of speech by one local speaker slot (0..2), relative to the start of the analysed audio. */
data class SpeakerTurn(val startSec: Double, val endSec: Double, val slot: Int) {
    val durationSec: Double get() = endSec - startSec
}

/**
 * Speech turns from pyannote/segmentation-3.0 (powerset, up to 3 local speakers).
 *
 * The Python pipeline runs the full pyannote/speaker-diarization-3.1 (segmentation + wespeaker
 * embeddings + clustering) on every chunk. Here only the segmentation model runs; who each turn
 * belongs to is decided afterwards from ECAPA voice embeddings (RealtimeTranscriber already merges
 * turns by voice similarity and assigns session speakers that way). Slots are only meaningful
 * inside one call: slot 0 of one chunk is not slot 0 of the next.
 */
class Diarizer(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : TurnDetector, AutoCloseable {
    private val inputName: String = session.inputInfo.keys.first()

    override fun turns(audio: FloatArray): List<SpeakerTurn> = turns(audio, GAP_FILL_SEC, MIN_TURN_SEC)

    fun turns(audio: FloatArray, gapFillSec: Double, minTurnSec: Double): List<SpeakerTurn> {
        if (audio.size < MIN_SAMPLES) return emptyList()
        val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), longArrayOf(1, 1, audio.size.toLong()))
        val logProbs = input.use {
            session.run(mapOf(inputName to it)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<Array<FloatArray>>)[0]
            }
        }
        return decodeTurns(logProbs, gapFillSec, minTurnSec)
    }

    override fun close() = session.close()

    companion object {
        /** Receptive-field hop of the model: 270 samples at 16 kHz (589 frames for 10 s). */
        const val FRAME_STEP_SEC = 270.0 / 16_000
        const val GAP_FILL_SEC = 0.5
        const val MIN_TURN_SEC = 0.5
        private const val MIN_SAMPLES = 1_600

        /** Powerset classes -> active local speakers: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}. */
        private val POWERSET = arrayOf(
            booleanArrayOf(false, false, false),
            booleanArrayOf(true, false, false),
            booleanArrayOf(false, true, false),
            booleanArrayOf(false, false, true),
            booleanArrayOf(true, true, false),
            booleanArrayOf(true, false, true),
            booleanArrayOf(false, true, true)
        )

        /**
         * argmax per frame -> per-slot activity -> runs -> runs closer than [gapFillSec] merged ->
         * turns shorter than [minTurnSec] dropped. Result sorted by start time.
         */
        fun decodeTurns(
            logProbs: Array<FloatArray>,
            gapFillSec: Double = GAP_FILL_SEC,
            minTurnSec: Double = MIN_TURN_SEC
        ): List<SpeakerTurn> {
            val frames = logProbs.size
            val active = Array(frames) { f ->
                var best = 0
                for (c in 1 until POWERSET.size) if (logProbs[f][c] > logProbs[f][best]) best = c
                POWERSET[best]
            }

            val turns = mutableListOf<SpeakerTurn>()
            for (slot in 0 until 3) {
                val merged = mutableListOf<DoubleArray>() // [start, end]
                var f = 0
                while (f < frames) {
                    if (!active[f][slot]) {
                        f++
                        continue
                    }
                    var last = f
                    while (last + 1 < frames && active[last + 1][slot]) last++
                    val start = f * FRAME_STEP_SEC
                    val end = (last + 1) * FRAME_STEP_SEC
                    val previous = merged.lastOrNull()
                    if (previous != null && start - previous[1] < gapFillSec) previous[1] = end else merged += doubleArrayOf(start, end)
                    f = last + 1
                }
                for (m in merged) if (m[1] - m[0] >= minTurnSec) turns += SpeakerTurn(m[0], m[1], slot)
            }
            return turns.sortedWith(compareBy({ it.startSec }, { it.slot }))
        }

        fun fromFile(file: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): Diarizer =
            Diarizer(env, env.createSession(file.absolutePath, OrtSession.SessionOptions()))

        fun fromAssets(context: Context, assetPath: String): Diarizer =
            fromFile(AssetFiles.materialize(context, assetPath))
    }
}
