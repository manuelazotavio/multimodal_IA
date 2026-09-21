package com.avtracker.mobile.whisper

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.fusion.SpeechRecognizer
import com.avtracker.mobile.voice.AssetFiles
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Base64
import kotlin.math.exp
import kotlin.math.ln

/** meta.json written by scripts/export_whisper.py. */
@Serializable
data class WhisperMeta(
    val vocabSize: Int,
    val eot: Int,
    val sot: Int,
    val transcribe: Int,
    val translate: Int,
    val noTimestamps: Int,
    val noSpeech: Int,
    val languages: Map<String, Int>,
    val suppressTokens: List<Int>,
    val beginSuppressTokens: List<Int>,
    val layers: Int,
    val heads: Int,
    val headDim: Int,
    val maxTargetPositions: Int,
    val nSamples: Int
)

data class TranscriptionResult(
    val text: String,
    val tokens: List<Int>,
    /** Sum of chosen-token log-probs (including end-of-text) divided by tokens + 1, like faster-whisper. */
    val avgLogProb: Float,
    /** Probability of the no-speech token at the first decoding step. */
    val noSpeechProb: Float
)

/** Byte-level BPE vocabulary: token id -> raw bytes (empty for special tokens). */
class WhisperTokens(private val bytes: List<ByteArray>) {
    fun decode(ids: List<Int>): String {
        val out = ByteArrayOutputStream()
        for (id in ids) if (id in bytes.indices) out.write(bytes[id])
        return String(out.toByteArray(), Charsets.UTF_8).trim()
    }

    companion object {
        /** tokens.txt: one base64-encoded token per line, line number = token id. */
        fun parse(text: String): WhisperTokens {
            val decoder = Base64.getDecoder()
            return WhisperTokens(text.split('\n').map { if (it.isEmpty()) ByteArray(0) else decoder.decode(it.trim()) })
        }
    }
}

/**
 * Whisper speech-to-text on ONNX Runtime (scripts/export_whisper.py), greedy decoding without timestamps.
 *
 * The encoder graph contains the log-mel front-end, so it takes 30 s of raw 16 kHz audio and returns the
 * cross-attention K/V of every decoder layer. The decoder graph runs one token per call with an explicit
 * self-attention cache. The loop mirrors the Python reference used to validate the export (identical
 * transcripts to HuggingFace `generate` on real meeting speech).
 */
class WhisperTranscriber(
    private val env: OrtEnvironment,
    private val encoder: OrtSession,
    private val decoder: OrtSession,
    private val meta: WhisperMeta,
    private val tokens: WhisperTokens
) : SpeechRecognizer, AutoCloseable {

    private val crossNames = List(meta.layers) { i -> listOf("cross_k_$i", "cross_v_$i") }.flatten()
    private val selfNames = List(meta.layers) { i -> listOf("self_k_$i", "self_v_$i") }.flatten()
    private val suppress = meta.suppressTokens.toIntArray()
    private val beginSuppress = meta.beginSuppressTokens.toIntArray()

    fun supports(language: String) = language in meta.languages

    /**
     * Transcribes up to 30 s of 16 kHz mono audio in [-1, 1]. Returns null if [language] is unknown.
     * Audio longer than 30 s is truncated, shorter audio is zero-padded (as Whisper itself does).
     */
    override fun transcribe(audio: FloatArray, language: String): TranscriptionResult? = transcribe(audio, language, 200)

    fun transcribe(audio: FloatArray, language: String, maxTokens: Int): TranscriptionResult? {
        val languageId = meta.languages[language] ?: return null

        val padded = FloatArray(meta.nSamples)
        System.arraycopy(audio, 0, padded, 0, minOf(audio.size, meta.nSamples))

        val audioTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(padded), longArrayOf(1, meta.nSamples.toLong()))
        return audioTensor.use { input ->
            encoder.run(mapOf("audio" to input)).use { encoderResult ->
                val cross = crossNames.associateWith { name -> encoderResult.get(name).get() as OnnxTensor }
                decode(cross, languageId, maxTokens)
            }
        }
    }

    private fun decode(cross: Map<String, OnnxTensor>, languageId: Int, maxTokens: Int): TranscriptionResult {
        val prefix = intArrayOf(meta.sot, languageId, meta.transcribe, meta.noTimestamps)
        val emptyCache = selfNames.associateWith {
            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, meta.heads.toLong(), 0, meta.headDim.toLong()))
        }

        val generated = mutableListOf<Int>()
        var logProbSum = 0.0
        var noSpeechProb = 0f
        var previous: OrtSession.Result? = null
        var cache: Map<String, OnnxTensor> = emptyCache
        var next = prefix[0]

        try {
            for (step in 0 until prefix.size + maxTokens) {
                if (step >= meta.maxTargetPositions) break

                val token = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(next.toLong())), longArrayOf(1, 1))
                val position = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(step.toLong())), longArrayOf(1))
                val result = try {
                    decoder.run(mapOf("token" to token, "position" to position) + cache + cross)
                } finally {
                    token.close()
                    position.close()
                }

                // The previous step's outputs were this step's cache inputs; they can go now.
                previous?.close()
                previous = result
                // Decoder outputs are [logits, new_self_k_0, new_self_v_0, ...] in the same order as selfNames.
                cache = selfNames.mapIndexed { i, name -> name to (result.get(1 + i) as OnnxTensor) }.toMap()

                @Suppress("UNCHECKED_CAST")
                val logits = (result.get(0).value as Array<FloatArray>)[0]

                if (step == 0) noSpeechProb = softmaxAt(logits, meta.noSpeech)
                if (step < prefix.size - 1) {
                    next = prefix[step + 1]
                    continue
                }

                for (id in suppress) logits[id] = Float.NEGATIVE_INFINITY
                if (generated.isEmpty()) for (id in beginSuppress) logits[id] = Float.NEGATIVE_INFINITY

                val best = argmax(logits)
                logProbSum += logProbability(logits, best)
                if (best == meta.eot) break
                generated += best
                next = best
            }
        } finally {
            previous?.close()
            emptyCache.values.forEach { it.close() }
        }

        return TranscriptionResult(
            text = tokens.decode(generated),
            tokens = generated,
            avgLogProb = (logProbSum / (generated.size + 1)).toFloat(),
            noSpeechProb = noSpeechProb
        )
    }

    private fun argmax(logits: FloatArray): Int {
        var best = 0
        for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
        return best
    }

    private fun logSumExp(logits: FloatArray): Double {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        var sum = 0.0
        for (v in logits) sum += exp((v - max).toDouble())
        return max + ln(sum)
    }

    private fun logProbability(logits: FloatArray, id: Int): Double = logits[id] - logSumExp(logits)

    private fun softmaxAt(logits: FloatArray, id: Int): Float = exp(logProbability(logits, id)).toFloat()

    override fun close() {
        encoder.close()
        decoder.close()
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        fun fromFolder(folder: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): WhisperTranscriber {
            val meta = json.decodeFromString(WhisperMeta.serializer(), File(folder, "meta.json").readText())
            val tokens = WhisperTokens.parse(File(folder, "tokens.txt").readText())
            val options = { OrtSession.SessionOptions() }
            return WhisperTranscriber(
                env,
                env.createSession(File(folder, "encoder.onnx").absolutePath, options()),
                env.createSession(File(folder, "decoder.onnx").absolutePath, options()),
                meta,
                tokens
            )
        }

        /** Assets live under assets/[dir]: encoder.onnx, decoder.onnx, tokens.txt, meta.json. */
        fun fromAssets(context: Context, dir: String = "whisper"): WhisperTranscriber {
            val encoder = AssetFiles.materialize(context, "$dir/encoder.onnx")
            AssetFiles.materialize(context, "$dir/decoder.onnx")
            AssetFiles.materialize(context, "$dir/tokens.txt")
            AssetFiles.materialize(context, "$dir/meta.json")
            return fromFolder(encoder.parentFile!!)
        }
    }
}
