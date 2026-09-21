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
import java.util.PriorityQueue
import java.util.Random
import java.util.zip.Deflater
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
    val nSamples: Int,
    val startOfPrev: Int = 50361,
    val startOfLm: Int = 50360,
    /** The initial prompts ("pt", "en") already tokenized like faster-whisper does (` ` + prompt, no special tokens). */
    val prompts: Map<String, List<Int>> = emptyMap()
)

data class TranscriptionResult(
    val text: String,
    val tokens: List<Int>,
    /** Sum of chosen-token log-probs (including end-of-text) divided by tokens + 1, like faster-whisper. */
    val avgLogProb: Float,
    /** Probability of the no-speech token at the start-of-transcript step. */
    val noSpeechProb: Float
)

/**
 * The decoding settings av-tracker passes to faster-whisper in RealtimeTranscriber._process_segment (beam 5, VAD with a
 * 500 ms silence / 300 ms pad, initial prompt, no conditioning on previous text, the default temperature fallback and the
 * compression ratio / log-prob / no-speech thresholds).
 */
data class WhisperOptions(
    val beamSize: Int = 5,
    val patience: Double = 1.0,
    val lengthPenalty: Double = 1.0,
    val bestOf: Int = 5,
    val temperatures: List<Double> = listOf(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    val compressionRatioThreshold: Double = 2.4,
    val logProbThreshold: Double = -1.0,
    val noSpeechThreshold: Double = 0.6,
    val vad: VadOptions? = VadOptions.AV_TRACKER,
    val usePrompt: Boolean = true,
    /** Upper bound on generated tokens per attempt: a runaway repetition would otherwise cost minutes on a phone. */
    val maxNewTokens: Int = 224,
    val seed: Long = 0
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
 * Whisper speech-to-text on ONNX Runtime (scripts/export_whisper.py) following faster-whisper's transcribe() as
 * av-tracker calls it: Silero VAD keeps the speech, the encoder takes the speech audio (log-mel in the graph, feature
 * padding with zeros like faster-whisper), the decoder starts from the initial prompt and runs a beam search with the
 * temperature fallback. Timestamps are not generated (one segment per 30 s window).
 *
 * The encoder graph returns the cross-attention K/V of every decoder layer. The decoder graph runs one token per call
 * with an explicit self-attention cache, so each beam owns its cache tensors.
 */
class WhisperTranscriber(
    private val env: OrtEnvironment,
    private val encoder: OrtSession,
    private val decoder: OrtSession,
    private val meta: WhisperMeta,
    private val tokens: WhisperTokens,
    private val vad: SileroVad? = null,
    private val options: WhisperOptions = WhisperOptions()
) : SpeechRecognizer, AutoCloseable {

    private val crossNames = List(meta.layers) { i -> listOf("cross_k_$i", "cross_v_$i") }.flatten()
    private val selfNames = List(meta.layers) { i -> listOf("self_k_$i", "self_v_$i") }.flatten()

    /** faster-whisper's suppression list: the model's default set plus the task / start-of-* tokens and every timestamp. */
    private val suppress: IntArray = (
        meta.suppressTokens + listOf(meta.transcribe, meta.translate, meta.sot, meta.startOfPrev, meta.startOfLm) +
            (meta.noTimestamps + 1 until meta.vocabSize)
        ).distinct().sorted().toIntArray()
    private val beginSuppress = meta.beginSuppressTokens.toIntArray()
    private val random = Random(options.seed)

    fun supports(language: String) = language in meta.languages

    // ---- public API ------------------------------------------------------------------------------------------------

    /** The full pipeline. Returns null when there is no speech, the language is unknown, or the window is skipped. */
    override fun transcribe(audio: FloatArray, language: String): TranscriptionResult? {
        val languageId = meta.languages[language] ?: return null

        var speech = audio
        if (options.vad != null && vad != null) {
            val chunks = vad.speechTimestamps(audio, options.vad)
            if (chunks.isEmpty()) return null
            speech = SileroVad.collectChunks(audio, chunks)
        }

        // The initial prompt only applies to the first window: with condition_on_previous_text=False the prompt is reset after it.
        val prompt = if (options.usePrompt) meta.prompts[if (language == "pt") "pt" else "en"] else null

        val segments = ArrayList<TranscriptionResult>()
        var offset = 0
        var window = 0
        while (offset < speech.size) {
            val end = minOf(offset + meta.nSamples, speech.size)
            transcribeWindow(speech.copyOfRange(offset, end), languageId, if (window == 0) prompt else null)?.let { segments += it }
            offset = end
            window++
        }
        if (segments.isEmpty()) return null

        return TranscriptionResult(
            text = segments.joinToString(" ") { it.text }.trim(),
            tokens = segments.flatMap { it.tokens },
            avgLogProb = segments.map { it.avgLogProb }.average().toFloat(),
            noSpeechProb = segments.map { it.noSpeechProb }.average().toFloat()
        )
    }

    /** Plain greedy decoding of up to 30 s, no VAD, prompt or fallback (the reference the export was validated against). */
    fun transcribeGreedy(audio: FloatArray, language: String, maxTokens: Int = 200): TranscriptionResult? {
        val languageId = meta.languages[language] ?: return null
        if (audio.size < HOP) return null
        return withEncoded(audio) { cross ->
            val prefix = intArrayOf(meta.sot, languageId, meta.transcribe, meta.noTimestamps)
            withPrefix(cross, prefix) { root, noSpeech ->
                val generated = ArrayList<Int>()
                var logProbSum = 0.0
                var step = root
                var owned: Step? = null
                try {
                    for (i in 0 until maxTokens) {
                        val logits = step.logits.copyOf()
                        mask(logits, generated.isEmpty())
                        val best = argmax(logits)
                        logProbSum += logProbability(logits, best)
                        if (best == meta.eot) break
                        generated += best
                        if (prefix.size + generated.size >= meta.maxTargetPositions) break
                        val next = runStep(best, prefix.size + generated.size - 1, step.cache, cross)
                        owned?.close()
                        owned = next
                        step = next
                    }
                } finally {
                    owned?.close()
                }
                TranscriptionResult(tokens.decode(generated), generated, (logProbSum / (generated.size + 1)).toFloat(), noSpeech)
            }
        }
    }

    // ---- one 30 s window ---------------------------------------------------------------------------------------------

    private class Decoded(val tokens: List<Int>, val cumLogProb: Double, val noSpeechProb: Float) {
        val avgLogProb: Double get() = cumLogProb / (tokens.size + 1)
    }

    private class Attempt(val decoded: Decoded, val temperature: Double, val compressionRatio: Double)

    private fun transcribeWindow(audio: FloatArray, languageId: Int, prompt: List<Int>?): TranscriptionResult? {
        if (audio.size < HOP) return null
        return withEncoded(audio) { cross ->
            val prefix = ArrayList<Int>()
            if (!prompt.isNullOrEmpty()) {
                prefix += meta.startOfPrev
                // faster-whisper keeps at most max_length / 2 - 1 previous tokens
                prefix += prompt.takeLast(meta.maxTargetPositions / 2 - 1)
            }
            prefix += listOf(meta.sot, languageId, meta.transcribe, meta.noTimestamps)

            withPrefix(cross, prefix.toIntArray()) { root, noSpeech ->
                val attempt = generateWithFallback(root, prefix.size, cross, noSpeech)
                val decoded = attempt.decoded

                // no-speech check: skipped unless the average log-prob is high enough
                var shouldSkip = decoded.noSpeechProb > options.noSpeechThreshold
                if (decoded.avgLogProb > options.logProbThreshold) shouldSkip = false
                if (shouldSkip) return@withPrefix null

                val text = tokens.decode(decoded.tokens)
                if (text.isBlank()) return@withPrefix null
                TranscriptionResult(text, decoded.tokens, decoded.avgLogProb.toFloat(), decoded.noSpeechProb)
            }
        }
    }

    /** faster-whisper `generate_with_fallback`. */
    private fun generateWithFallback(root: Step, prefixLen: Int, cross: Map<String, OnnxTensor>, noSpeech: Float): Attempt {
        val all = ArrayList<Attempt>()
        val belowCompression = ArrayList<Attempt>()

        for (temperature in options.temperatures) {
            val decoded = if (temperature > 0) sample(root, prefixLen, cross, noSpeech, temperature) else beamSearch(root, prefixLen, cross, noSpeech)
            val ratio = compressionRatio(tokens.decode(decoded.tokens))
            val attempt = Attempt(decoded, temperature, ratio)
            all += attempt

            var needsFallback = false
            if (ratio > options.compressionRatioThreshold) needsFallback = true else belowCompression += attempt
            if (decoded.avgLogProb < options.logProbThreshold) needsFallback = true
            if (decoded.noSpeechProb > options.noSpeechThreshold && decoded.avgLogProb < options.logProbThreshold) needsFallback = false // silence
            if (!needsFallback) return attempt
        }
        // every temperature failed: the result with the highest average log-prob (among those under the compression limit if any)
        return (belowCompression.ifEmpty { all }).maxByOrNull { it.decoded.avgLogProb }!!
    }

    // ---- beam search -------------------------------------------------------------------------------------------------

    private class Hyp(val generated: List<Int>, val score: Double, val step: Step, val ownsStep: Boolean)

    private class Finished(val generated: List<Int>, val cumLogProb: Double, val normalized: Double)

    /**
     * Beam search as CTranslate2 runs it for faster-whisper: [WhisperOptions.beamSize] hypotheses, the best `2 * beam`
     * candidates per step of which end-of-text ones count only within the first `beam`, patience-scaled stop once enough
     * hypotheses are finished, and the winner chosen by `cumulative log-prob / length ^ length_penalty`.
     */
    private fun beamSearch(root: Step, prefixLen: Int, cross: Map<String, OnnxTensor>, noSpeech: Float): Decoded {
        val beam = options.beamSize
        val maxCandidates = Math.round(beam * options.patience).toInt().coerceAtLeast(1)
        val maxNew = minOf(options.maxNewTokens, meta.maxTargetPositions - prefixLen)

        var active = listOf(Hyp(emptyList(), 0.0, root, ownsStep = false))
        val finished = ArrayList<Finished>()

        try {
            for (t in 0 until maxNew) {
                class Candidate(val parent: Int, val token: Int, val score: Double)

                val candidates = ArrayList<Candidate>()
                for ((hi, h) in active.withIndex()) {
                    val logits = h.step.logits.copyOf()
                    mask(logits, h.generated.isEmpty())
                    val lse = logSumExp(logits)
                    for ((token, logp) in topK(logits, lse, 2 * beam)) candidates += Candidate(hi, token, h.score + logp)
                }
                candidates.sortByDescending { it.score } // stable: ties keep hypothesis order
                val top = candidates.take(2 * beam)

                val next = ArrayList<Candidate>()
                for ((rank, c) in top.withIndex()) {
                    if (c.token == meta.eot) {
                        // end-of-text candidates only count while inside the first `beam` ranks
                        if (rank < beam) {
                            val h = active[c.parent]
                            finished += Finished(h.generated, c.score, c.score / (h.generated.size.coerceAtLeast(1).toDouble().pow(options.lengthPenalty)))
                        }
                    } else if (next.size < beam) {
                        next += c
                    }
                }

                if (finished.size >= maxCandidates || next.isEmpty()) break
                if (t == maxNew - 1) {
                    // length limit: the survivors end here without an end-of-text
                    for (c in next) {
                        val h = active[c.parent]
                        val generated = h.generated + c.token
                        finished += Finished(generated, c.score, c.score / generated.size.toDouble().pow(options.lengthPenalty))
                    }
                    break
                }

                val position = prefixLen + t
                val children = next.map { c ->
                    val parent = active[c.parent]
                    Hyp(parent.generated + c.token, c.score, runStep(c.token, position, parent.step.cache, cross), ownsStep = true)
                }
                active.forEach { if (it.ownsStep) it.step.close() }
                active = children
            }
        } finally {
            active.forEach { if (it.ownsStep) it.step.close() }
        }

        val best = finished.maxByOrNull { it.normalized } ?: return Decoded(emptyList(), 0.0, noSpeech)
        return Decoded(best.generated, best.cumLogProb, noSpeech)
    }

    /** `best_of` samples at [temperature] over the full distribution (CTranslate2 `sampling_topk=0`); the best by normalized log-prob wins. */
    private fun sample(root: Step, prefixLen: Int, cross: Map<String, OnnxTensor>, noSpeech: Float, temperature: Double): Decoded {
        val maxNew = minOf(options.maxNewTokens, meta.maxTargetPositions - prefixLen)
        var best: Decoded? = null
        var bestScore = Double.NEGATIVE_INFINITY

        repeat(options.bestOf) {
            val generated = ArrayList<Int>()
            var cum = 0.0
            var step = root
            var owned: Step? = null
            try {
                for (t in 0 until maxNew) {
                    val logits = step.logits.copyOf()
                    mask(logits, generated.isEmpty())
                    val lse = logSumExp(logits)
                    val token = sampleToken(logits, temperature)
                    cum += logits[token] - lse
                    if (token == meta.eot) break
                    generated += token
                    if (t == maxNew - 1) break
                    val next = runStep(token, prefixLen + t, step.cache, cross)
                    owned?.close()
                    owned = next
                    step = next
                }
            } finally {
                owned?.close()
            }
            val normalized = cum / generated.size.coerceAtLeast(1).toDouble().pow(options.lengthPenalty)
            if (normalized > bestScore) {
                bestScore = normalized
                best = Decoded(generated, cum, noSpeech)
            }
        }
        return best!!
    }

    private fun sampleToken(logits: FloatArray, temperature: Double): Int {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        val weights = DoubleArray(logits.size)
        var total = 0.0
        for (i in logits.indices) {
            val v = logits[i]
            if (v == Float.NEGATIVE_INFINITY) continue
            weights[i] = exp((v - max) / temperature)
            total += weights[i]
        }
        var r = random.nextDouble() * total
        for (i in weights.indices) {
            r -= weights[i]
            if (r <= 0 && weights[i] > 0) return i
        }
        return argmax(logits)
    }

    // ---- decoder plumbing --------------------------------------------------------------------------------------------

    /** One decoder call: the logits for the next token and the cache including the token just fed. */
    private class Step(val result: OrtSession.Result?, val logits: FloatArray, val cache: Map<String, OnnxTensor>) : AutoCloseable {
        override fun close() { result?.close() }
    }

    private fun runStep(token: Int, position: Int, cache: Map<String, OnnxTensor>, cross: Map<String, OnnxTensor>): Step {
        val tokenTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(token.toLong())), longArrayOf(1, 1))
        val positionTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(position.toLong())), longArrayOf(1))
        val result = try {
            decoder.run(mapOf("token" to tokenTensor, "position" to positionTensor) + cache + cross)
        } finally {
            tokenTensor.close()
            positionTensor.close()
        }
        // Decoder outputs are [logits, new_self_k_0, new_self_v_0, ...] in the same order as selfNames.
        @Suppress("UNCHECKED_CAST")
        val logits = (result.get(0).value as Array<FloatArray>)[0]
        val newCache = selfNames.mapIndexed { i, name -> name to (result.get(1 + i) as OnnxTensor) }.toMap()
        return Step(result, logits, newCache)
    }

    /** Runs the encoder on [audio] (at most 30 s) and hands the cross-attention K/V to [block]. */
    private fun <T> withEncoded(audio: FloatArray, block: (Map<String, OnnxTensor>) -> T): T {
        val samples = audio.copyOf(minOf(audio.size, meta.nSamples))
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(samples), longArrayOf(1, samples.size.toLong()))
        return tensor.use { input ->
            encoder.run(mapOf("audio" to input)).use { encoderResult ->
                block(crossNames.associateWith { name -> encoderResult.get(name).get() as OnnxTensor })
            }
        }
    }

    /**
     * Feeds the decoder prefix and hands the resulting state to [block] together with the no-speech probability, taken at the
     * start-of-transcript position (where faster-whisper reads it, whatever comes before it in the prompt).
     */
    private fun <T> withPrefix(cross: Map<String, OnnxTensor>, prefix: IntArray, block: (Step, Float) -> T): T {
        val emptyCache = selfNames.associateWith {
            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, meta.heads.toLong(), 0, meta.headDim.toLong()))
        }
        var step: Step? = null
        var noSpeech = 0f
        try {
            for ((i, token) in prefix.withIndex()) {
                val next = runStep(token, i, step?.cache ?: emptyCache, cross)
                if (token == meta.sot) noSpeech = softmaxAt(next.logits, meta.noSpeech)
                step?.close()
                step = next
            }
            return block(step!!, noSpeech)
        } finally {
            step?.close()
            emptyCache.values.forEach { it.close() }
        }
    }

    // ---- logits helpers ----------------------------------------------------------------------------------------------

    /** suppress_tokens, and suppress_blank (blank and end-of-text) before the first generated token. */
    private fun mask(logits: FloatArray, first: Boolean) {
        for (id in suppress) logits[id] = Float.NEGATIVE_INFINITY
        if (first) for (id in beginSuppress) logits[id] = Float.NEGATIVE_INFINITY
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

    /** The [k] highest log-probs as (token, log-prob), best first. */
    private fun topK(logits: FloatArray, logSumExp: Double, k: Int): List<Pair<Int, Double>> {
        val heap = PriorityQueue<Int>(k + 1) { a, b -> logits[a].compareTo(logits[b]) } // min-heap on the logit
        for (i in logits.indices) {
            if (logits[i] == Float.NEGATIVE_INFINITY) continue
            if (heap.size < k) heap.add(i) else if (logits[i] > logits[heap.peek()]) { heap.poll(); heap.add(i) }
        }
        return heap.sortedWith(compareByDescending<Int> { logits[it] }.thenBy { it }).map { it to (logits[it] - logSumExp) }
    }

    private fun Double.pow(e: Double) = Math.pow(this, e)

    override fun close() {
        encoder.close()
        decoder.close()
        vad?.close()
    }

    companion object {
        private const val HOP = 160
        private val json = Json { ignoreUnknownKeys = true }

        /** `len(text_bytes) / len(zlib.compress(text_bytes))` (faster-whisper `get_compression_ratio`). */
        fun compressionRatio(text: String): Double {
            val bytes = text.toByteArray(Charsets.UTF_8)
            if (bytes.isEmpty()) return 0.0
            val deflater = Deflater(Deflater.DEFAULT_COMPRESSION)
            deflater.setInput(bytes)
            deflater.finish()
            val buffer = ByteArray(1024)
            var size = 0
            while (!deflater.finished()) size += deflater.deflate(buffer)
            deflater.end()
            return bytes.size.toDouble() / size
        }

        fun fromFolder(
            folder: File,
            env: OrtEnvironment = OrtEnvironment.getEnvironment(),
            vad: SileroVad? = null,
            options: WhisperOptions = WhisperOptions()
        ): WhisperTranscriber {
            val meta = json.decodeFromString(WhisperMeta.serializer(), File(folder, "meta.json").readText())
            val tokens = WhisperTokens.parse(File(folder, "tokens.txt").readText())
            val sessionOptions = { OrtSession.SessionOptions() }
            return WhisperTranscriber(
                env,
                env.createSession(File(folder, "encoder.onnx").absolutePath, sessionOptions()),
                env.createSession(File(folder, "decoder.onnx").absolutePath, sessionOptions()),
                meta,
                tokens,
                vad,
                options
            )
        }

        /** Assets live under assets/[dir]: encoder.onnx, decoder.onnx, tokens.txt, meta.json (and models/silero_vad_v6.onnx for the VAD). */
        fun fromAssets(context: Context, dir: String = "whisper"): WhisperTranscriber {
            val encoder = AssetFiles.materialize(context, "$dir/encoder.onnx")
            AssetFiles.materialize(context, "$dir/decoder.onnx")
            AssetFiles.materialize(context, "$dir/tokens.txt")
            AssetFiles.materialize(context, "$dir/meta.json")
            return fromFolder(encoder.parentFile!!, vad = SileroVad.fromAssets(context))
        }
    }
}
