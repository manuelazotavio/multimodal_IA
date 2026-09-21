package com.avtracker.mobile.whisper

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.voice.AssetFiles
import java.io.File
import java.nio.FloatBuffer

/** faster-whisper's `VadOptions`; the defaults of the ones av-tracker overrides are the library's. */
data class VadOptions(
    val threshold: Double = 0.5,
    val negThreshold: Double? = null,
    val minSpeechDurationMs: Int = 0,
    val maxSpeechDurationS: Double = Double.POSITIVE_INFINITY,
    val minSilenceDurationMs: Int = 2000,
    val speechPadMs: Int = 400
) {
    companion object {
        /** `vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300}` in RealtimeTranscriber._process_segment. */
        val AV_TRACKER = VadOptions(minSilenceDurationMs = 500, speechPadMs = 300)
    }
}

/** A speech chunk as sample offsets into the input audio. */
data class SpeechChunk(val start: Int, val end: Int)

/**
 * Port of faster-whisper's `get_speech_timestamps` / `collect_chunks` over the Silero VAD v6 ONNX model that ships with
 * faster-whisper (`vad_filter=True`). Includes the library's quirks: all frames go through the model as one batch with a
 * 64-sample context taken from the previous frame, and the last frame's tail is zeroed.
 */
class SileroVad(private val env: OrtEnvironment, private val session: OrtSession) : AutoCloseable {

    /** One speech probability per 512-sample frame of [padded] (its length must be a multiple of 512). */
    internal fun speechProbabilities(padded: FloatArray): FloatArray {
        val frames = padded.size / WINDOW
        val width = WINDOW + CONTEXT
        val batch = FloatArray(frames * width)

        // context = batched_audio[..., -64:] is a view: `context[-1] = 0` also zeroes the tail of the last frame.
        for (i in 0 until frames) {
            val base = i * width
            // rolled context: frame i gets the last 64 samples of frame i-1; frame 0 gets zeros
            if (i > 0) System.arraycopy(padded, (i - 1) * WINDOW + (WINDOW - CONTEXT), batch, base, CONTEXT)
            System.arraycopy(padded, i * WINDOW, batch, base + CONTEXT, WINDOW)
        }
        // the last frame's own tail is zeroed in the batch
        val lastBase = (frames - 1) * width + CONTEXT
        java.util.Arrays.fill(batch, lastBase + WINDOW - CONTEXT, lastBase + WINDOW, 0f)

        val probs = FloatArray(frames)
        var h = FloatArray(128)
        var c = FloatArray(128)
        var offset = 0
        while (offset < frames) {
            val n = minOf(ENCODER_BATCH, frames - offset)
            val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(batch, offset * width, n * width), longArrayOf(n.toLong(), width.toLong()))
            val hTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(h), longArrayOf(1, 1, 128))
            val cTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(c), longArrayOf(1, 1, 128))
            try {
                session.run(mapOf("input" to input, "h" to hTensor, "c" to cTensor)).use { result ->
                    val out = (result.get(0) as OnnxTensor).floatBuffer
                    for (k in 0 until n) probs[offset + k] = out.get(k)
                    h = FloatArray(128).also { (result.get(1) as OnnxTensor).floatBuffer.get(it) }
                    c = FloatArray(128).also { (result.get(2) as OnnxTensor).floatBuffer.get(it) }
                }
            } finally {
                input.close(); hTensor.close(); cTensor.close()
            }
            offset += n
        }
        return probs
    }

    /** `get_speech_timestamps(audio, vad_options)` at 16 kHz. */
    fun speechTimestamps(audio: FloatArray, options: VadOptions = VadOptions.AV_TRACKER): List<SpeechChunk> {
        val sr = 16000
        val threshold = options.threshold
        val negThreshold = options.negThreshold ?: maxOf(threshold - 0.15, 0.01)
        val minSpeechSamples = sr * options.minSpeechDurationMs / 1000.0
        val speechPadSamples = sr * options.speechPadMs / 1000.0
        val maxSpeechSamples = sr * options.maxSpeechDurationS - WINDOW - 2 * speechPadSamples
        val minSilenceSamples = sr * options.minSilenceDurationMs / 1000.0
        val minSilenceSamplesAtMaxSpeech = sr * 98 / 1000.0

        val length = audio.size
        val padded = audio.copyOf(length + (WINDOW - length % WINDOW))
        val probs = speechProbabilities(padded)

        var triggered = false
        val speeches = ArrayList<IntArray>() // [start, end]
        var currentStart = -1 // -1: current_speech is empty
        var currentEnd = -1
        var tempEnd = 0
        var prevEnd = 0
        var nextStart = 0

        fun currentIsEmpty() = currentStart < 0

        for ((i, probF) in probs.withIndex()) {
            val prob = probF.toDouble()
            val pos = WINDOW * i
            if (prob >= threshold && tempEnd != 0) {
                tempEnd = 0
                if (nextStart < prevEnd) nextStart = pos
            }

            if (prob >= threshold && !triggered) {
                triggered = true
                currentStart = pos
                continue
            }

            if (triggered && pos - currentStart > maxSpeechSamples) {
                if (prevEnd != 0) {
                    speeches += intArrayOf(currentStart, prevEnd)
                    currentStart = -1
                    if (nextStart < prevEnd) triggered = false else currentStart = nextStart
                    prevEnd = 0; nextStart = 0; tempEnd = 0
                } else {
                    speeches += intArrayOf(currentStart, pos)
                    currentStart = -1
                    prevEnd = 0; nextStart = 0; tempEnd = 0
                    triggered = false
                    continue
                }
            }

            if (prob < negThreshold && triggered) {
                if (tempEnd == 0) tempEnd = pos
                if (pos - tempEnd > minSilenceSamplesAtMaxSpeech) prevEnd = tempEnd
                if (pos - tempEnd < minSilenceSamples) {
                    continue
                } else {
                    currentEnd = tempEnd
                    if (currentEnd - currentStart > minSpeechSamples) speeches += intArrayOf(currentStart, currentEnd)
                    currentStart = -1
                    prevEnd = 0; nextStart = 0; tempEnd = 0
                    triggered = false
                    continue
                }
            }
        }

        if (!currentIsEmpty() && (length - currentStart) > minSpeechSamples) {
            speeches += intArrayOf(currentStart, length)
        }

        for (i in speeches.indices) {
            val speech = speeches[i]
            if (i == 0) speech[0] = maxOf(0.0, speech[0] - speechPadSamples).toInt()
            if (i != speeches.size - 1) {
                val silence = speeches[i + 1][0] - speech[1]
                if (silence < 2 * speechPadSamples) {
                    speech[1] += Math.floorDiv(silence, 2)
                    speeches[i + 1][0] = maxOf(0.0, (speeches[i + 1][0] - Math.floorDiv(silence, 2)).toDouble()).toInt()
                } else {
                    speech[1] = minOf(length.toDouble(), speech[1] + speechPadSamples).toInt()
                    speeches[i + 1][0] = maxOf(0.0, speeches[i + 1][0] - speechPadSamples).toInt()
                }
            } else {
                speech[1] = minOf(length.toDouble(), speech[1] + speechPadSamples).toInt()
            }
        }
        return speeches.map { SpeechChunk(it[0], it[1]) }
    }

    override fun close() = session.close()

    companion object {
        private const val WINDOW = 512
        private const val CONTEXT = 64
        private const val ENCODER_BATCH = 10000

        /** `collect_chunks(audio, chunks)` with no maximum duration: the speech pieces back to back. */
        fun collectChunks(audio: FloatArray, chunks: List<SpeechChunk>): FloatArray {
            val out = FloatArray(chunks.sumOf { it.end - it.start })
            var pos = 0
            for (c in chunks) {
                System.arraycopy(audio, c.start, out, pos, c.end - c.start)
                pos += c.end - c.start
            }
            return out
        }

        fun fromFile(file: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): SileroVad {
            val options = OrtSession.SessionOptions().apply {
                setInterOpNumThreads(1)
                setIntraOpNumThreads(1)
                setMemoryPatternOptimization(false)
            }
            return SileroVad(env, env.createSession(file.absolutePath, options))
        }

        fun fromAssets(context: Context, path: String = "models/silero_vad_v6.onnx"): SileroVad = fromFile(AssetFiles.materialize(context, path))
    }
}
