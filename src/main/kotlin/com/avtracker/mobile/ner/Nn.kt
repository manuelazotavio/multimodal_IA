package com.avtracker.mobile.ner

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/** float32 tensors exported by scripts/export_ner.py / export_parser.py, plus the thinc layers the spaCy models use. */
internal class NnWeights(infos: List<TensorInfo>, weights: ByteArray) {
    private val tensors = HashMap<String, FloatArray>()
    private val shapes = HashMap<String, List<Int>>()

    init {
        val buf = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        for (info in infos) {
            val n = info.shape.fold(1) { a, b -> a * b }
            tensors[info.name] = FloatArray(n).also { buf.get(it) }
            shapes[info.name] = info.shape
        }
    }

    fun tensor(name: String): FloatArray = tensors.getValue(name)

    fun shape(name: String): List<Int> = shapes.getValue(name)

    /** x [n, nI] times w [nO, nP, nI] plus b, max over the nP pieces -> [n, nO]. */
    fun maxout(x: Array<FloatArray>, wName: String, bName: String, nP: Int): Array<FloatArray> {
        val w = tensor(wName)
        val b = tensor(bName)
        val shape = shape(wName) // [nO, nP, nI]
        val nO = shape[0]
        val nI = shape[2]
        return Array(x.size) { r ->
            val row = x[r]
            FloatArray(nO) { o ->
                var best = Float.NEGATIVE_INFINITY
                for (p in 0 until nP) {
                    val base = (o * nP + p) * nI
                    var sum = b[o * nP + p]
                    for (i in 0 until nI) sum += row[i] * w[base + i]
                    if (sum > best) best = sum
                }
                best
            }
        }
    }

    fun layerNorm(x: Array<FloatArray>, gName: String, bName: String): Array<FloatArray> {
        val g = tensor(gName)
        val b = tensor(bName)
        return Array(x.size) { r ->
            val row = x[r]
            var mean = 0.0
            for (v in row) mean += v
            mean /= row.size
            var variance = 0.0
            for (v in row) variance += (v - mean) * (v - mean)
            variance = variance / row.size + 1e-8
            val inv = 1.0 / sqrt(variance)
            FloatArray(row.size) { i -> (((row[i] - mean) * inv).toFloat() * g[i]) + b[i] }
        }
    }

    /** thinc `expand_window(window_size=1)`: each row concatenated with its neighbours, zeros beyond the edges. */
    fun window(x: Array<FloatArray>): Array<FloatArray> {
        val width = x[0].size
        return Array(x.size) { r ->
            val out = FloatArray(width * 3)
            if (r > 0) System.arraycopy(x[r - 1], 0, out, 0, width)
            System.arraycopy(x[r], 0, out, width, width)
            if (r < x.size - 1) System.arraycopy(x[r + 1], 0, out, 2 * width, width)
            out
        }
    }
}

/**
 * spaCy's `HashEmbedCNN` tok2vec followed by the `linear` projection: one hashed embedding per lexical attribute
 * (summed over the four bucket keys of `MurmurHash3_x86_128_uint64`), a maxout+LayerNorm mix, [depth] residual
 * window+maxout+LayerNorm layers over a sequence padded with [pad] zero rows on each side, and the projection.
 */
internal class Tok2Vec(
    private val nn: NnWeights,
    private val width: Int,
    private val seeds: List<Int>,
    private val rows: List<Int>,
    private val depth: Int,
    private val pad: Int,
    private val pieces: Int
) {
    /** [keys] holds, per token, one 64-bit attribute key per embedding table. Returns [n, projection width]. */
    fun encode(keys: List<LongArray>): Array<FloatArray> {
        val n = keys.size
        val columns = seeds.size
        val concat = Array(n) { FloatArray(width * columns) }
        for (col in 0 until columns) {
            val table = nn.tensor("embed$col.E")
            val nRows = rows[col].toLong()
            for (t in 0 until n) {
                for (h in SpacyHash.hashIds(keys[t][col], seeds[col])) {
                    val rowStart = ((h % nRows).toInt()) * width
                    for (i in 0 until width) concat[t][col * width + i] += table[rowStart + i]
                }
            }
        }

        val x = nn.layerNorm(nn.maxout(concat, "mix0.W", "mix0.b", pieces), "mix0.G", "mix0.beta")

        // with_array(pad=4): the sequence is padded with zero rows on each side, and the padding takes part in every layer.
        var flat = Array(n + 2 * pad) { r -> if (r in pad until pad + n) x[r - pad] else FloatArray(width) }
        for (d in 1..depth) {
            val y = nn.layerNorm(nn.maxout(nn.window(flat), "mix$d.W", "mix$d.b", pieces), "mix$d.G", "mix$d.beta")
            flat = Array(flat.size) { r -> FloatArray(width) { i -> flat[r][i] + y[r][i] } }
        }

        val w = nn.tensor("proj.W")
        val b = nn.tensor("proj.b")
        val nO = nn.shape("proj.W")[0]
        return Array(n) { r ->
            val row = flat[pad + r]
            FloatArray(nO) { o -> var s = b[o]; for (i in 0 until width) s += row[i] * w[o * width + i]; s }
        }
    }
}

/**
 * The `PrecomputableAffine` lower layer and `Linear` upper layer of a transition-based parser/NER: every (token, feature
 * slot) contribution is computed once, a state's hidden vector is the maxout of the summed contributions of its feature
 * tokens (slots without a token take the learned padding), and the action scores come from the upper layer.
 */
internal class TransitionScorer(private val nn: NnWeights, private val features: Int, private val hidden: Int, private val pieces: Int, private val actions: Int) {
    private lateinit var cached: Array<Array<FloatArray>>

    /** Precomputes the contribution of every token in every feature slot. */
    fun prepare(vectors: Array<FloatArray>) {
        val nI = vectors[0].size
        val lowerW = nn.tensor("lower.W") // [nF, nH, nP, nI]
        cached = Array(vectors.size) { t ->
            Array(features) { f ->
                FloatArray(hidden * pieces) { j ->
                    val base = (f * hidden * pieces + j) * nI
                    var s = 0f
                    for (i in 0 until nI) s += vectors[t][i] * lowerW[base + i]
                    s
                }
            }
        }
    }

    /** The best-scoring action among those [valid] accepts, or -1 (spaCy `arg_max_if_valid`: the first of equal scores). */
    fun best(ids: IntArray, valid: (Int) -> Boolean): Int {
        val pad = nn.tensor("lower.pad")
        val bias = nn.tensor("lower.b")
        val upperW = nn.tensor("upper.W") // [nActions, nH]
        val upperB = nn.tensor("upper.b")

        val acc = FloatArray(hidden * pieces)
        for (f in 0 until features) {
            val id = ids[f]
            if (id < 0) for (j in acc.indices) acc[j] += pad[f * hidden * pieces + j]
            else for (j in acc.indices) acc[j] += cached[id][f][j]
        }
        val hiddenVec = FloatArray(hidden) { h ->
            var best = Float.NEGATIVE_INFINITY
            for (p in 0 until pieces) best = maxOf(best, acc[h * pieces + p] + bias[h * pieces + p])
            best
        }

        var best = -1
        var bestScore = 0f
        for (a in 0 until actions) {
            if (!valid(a)) continue
            var score = upperB[a]
            for (h in 0 until hidden) score += hiddenVec[h] * upperW[a * hidden + h]
            if (best == -1 || score > bestScore) { best = a; bestScore = score }
        }
        return best
    }
}
