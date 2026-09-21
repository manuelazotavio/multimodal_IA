package com.avtracker.mobile.ner

import android.content.Context
import com.avtracker.mobile.fusion.EntityExtractor
import com.avtracker.mobile.fusion.PyText
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

@Serializable
data class TensorInfo(val name: String, val shape: List<Int>)

@Serializable
data class NerAction(val move: String, val label: String)

/** model.json written by scripts/export_ner.py. */
@Serializable
data class NerMeta(
    val name: String,
    val width: Int,
    val seeds: List<Int>,
    val rows: List<Int>,
    val encoderDepth: Int,
    val encoderPad: Int,
    val maxoutPieces: Int,
    val hiddenWidth: Int,
    val lowerPieces: Int,
    val numFeatures: Int,
    val personLabels: List<String>,
    val actions: List<NerAction>,
    val tensors: List<TensorInfo>,
    /** Strings that spaCy's StringStore maps to fixed symbol ids instead of hashes. */
    val symbols: Map<String, Long> = emptyMap()
)

/** lexeme_norm.json: the id-keyed lexeme_norm table and the language-independent BASE_NORMS. */
@Serializable
data class NormFile(val table: Map<String, String>, val base: Map<String, String>)

class NormTables(file: NormFile) {
    val table: Map<Long, String> = file.table.entries.associate { (k, v) -> java.lang.Long.parseUnsignedLong(k) to v }
    val base: Map<String, String> = file.base
}

/** A recognised entity as token indices [startToken, endToken) plus its label. */
data class NerEntity(val startToken: Int, val endToken: Int, val label: String)

/**
 * spaCy's `ner` component (en_core_web_sm / pt_core_news_sm) as plain Kotlin: hashed lexical attributes -> HashEmbed ->
 * maxout+LayerNorm mixing -> four residual convolution layers -> projection -> a greedy BILUO transition system.
 * Verified against the original on real transcripts (see NerParityTest).
 *
 * Not reproduced: the dependency parser that runs before `ner` in the spaCy pipeline only matters through sentence
 * boundaries, which forbid an entity from spanning two sentences. On av-tracker's own transcripts the entities are
 * identical with and without it.
 */
class SpacyNerModel(private val meta: NerMeta, weights: ByteArray, private val norms: NormTables) {
    private val tensors: Map<String, FloatArray>
    private val shapes: Map<String, List<Int>>

    init {
        val buf = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val t = HashMap<String, FloatArray>()
        val s = HashMap<String, List<Int>>()
        for (info in meta.tensors) {
            val n = info.shape.fold(1) { a, b -> a * b }
            t[info.name] = FloatArray(n).also { buf.get(it) }
            s[info.name] = info.shape
        }
        tensors = t
        shapes = s
    }

    // ---- lexical attributes -------------------------------------------------------------------------------------------

    /**
     * The NORM attribute: a special-case override, else the lexeme_norm table (keyed by string id), else BASE_NORMS,
     * else the lower-cased text.
     */
    fun normOf(token: SpacyToken): String =
        token.norm ?: (norms.table[key(token.text)] ?: norms.base[token.text] ?: token.text.lowercase())

    /** The four attribute hashes the embedding tables take: NORM, PREFIX, SUFFIX, SHAPE. */
    fun attributeKeys(token: SpacyToken): LongArray {
        val text = token.text
        val norm = normOf(token)
        val points = text.codePoints().toArray()
        val prefix = String(points, 0, 1)
        val suffix = String(points, maxOf(0, points.size - 3), minOf(3, points.size))
        return longArrayOf(key(norm), key(prefix), key(suffix), key(wordShape(text, points)))
    }

    /** The id spaCy's StringStore gives an attribute string: a fixed symbol id if it has one, else its hash. */
    private fun key(s: String): Long = meta.symbols[s] ?: SpacyHash.hashString(s)

    /** spaCy `lang.lex_attrs.word_shape`: X/x for letters, d for digits, runs longer than 4 collapsed. */
    private fun wordShape(text: String, points: IntArray): String {
        if (points.size >= 100) return "LONG"
        val shape = StringBuilder()
        var last = -1
        var seq = 0
        for (cp in points) {
            val shapeChar = when {
                Character.isLetter(cp) -> if (Character.isUpperCase(cp)) 'X'.code else 'x'.code
                Character.isDigit(cp) -> 'd'.code
                else -> cp
            }
            if (shapeChar == last) seq++ else { seq = 0; last = shapeChar }
            if (seq < 4) shape.appendCodePoint(shapeChar)
        }
        return shape.toString()
    }

    // ---- network ------------------------------------------------------------------------------------------------------

    private fun tensor(name: String) = tensors.getValue(name)

    /** x [n, nI] times w [nO*nP, nI]^T plus b, max over the nP pieces -> [n, nO]. */
    private fun maxout(x: Array<FloatArray>, wName: String, bName: String, nP: Int): Array<FloatArray> {
        val w = tensor(wName)
        val b = tensor(bName)
        val shape = shapes.getValue(wName) // [nO, nP, nI]
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

    private fun layerNorm(x: Array<FloatArray>, gName: String, bName: String): Array<FloatArray> {
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
    private fun window(x: Array<FloatArray>): Array<FloatArray> {
        val width = x[0].size
        return Array(x.size) { r ->
            val out = FloatArray(width * 3)
            if (r > 0) System.arraycopy(x[r - 1], 0, out, 0, width)
            System.arraycopy(x[r], 0, out, width, width)
            if (r < x.size - 1) System.arraycopy(x[r + 1], 0, out, 2 * width, width)
            out
        }
    }

    /** Token vectors [n, hiddenWidth] from the attribute keys of each token. */
    fun tokenVectors(keys: List<LongArray>): Array<FloatArray> {
        val n = keys.size
        val width = meta.width
        val concat = Array(n) { FloatArray(width * 4) }
        for (col in 0 until 4) {
            val table = tensor("embed$col.E")
            val rows = meta.rows[col].toLong()
            for (t in 0 until n) {
                for (h in SpacyHash.hashIds(keys[t][col], meta.seeds[col])) {
                    val rowStart = ((h % rows).toInt()) * width
                    for (i in 0 until width) concat[t][col * width + i] += table[rowStart + i]
                }
            }
        }

        var x = layerNorm(maxout(concat, "mix0.W", "mix0.b", meta.maxoutPieces), "mix0.G", "mix0.beta")

        // with_array(pad=4): the sequence is padded with 4 zero rows on each side, and the padding takes part in every layer.
        val pad = meta.encoderPad
        var flat = Array(n + 2 * pad) { r -> if (r in pad until pad + n) x[r - pad] else FloatArray(width) }
        for (d in 1..meta.encoderDepth) {
            val y = layerNorm(maxout(window(flat), "mix$d.W", "mix$d.b", meta.maxoutPieces), "mix$d.G", "mix$d.beta")
            flat = Array(flat.size) { r -> FloatArray(width) { i -> flat[r][i] + y[r][i] } }
        }
        x = Array(n) { flat[pad + it] }

        val w = tensor("proj.W")
        val b = tensor("proj.b")
        val nO = shapes.getValue("proj.W")[0]
        return Array(n) { r -> FloatArray(nO) { o -> var s = b[o]; for (i in 0 until width) s += x[r][i] * w[o * width + i]; s } }
    }

    /** Greedy BILUO decoding of token vectors into entities. [isSpace] marks whitespace tokens, which cannot start one. */
    fun decode(vectors: Array<FloatArray>, isSpace: BooleanArray): List<NerEntity> {
        val n = vectors.size
        if (n == 0) return emptyList()

        val nF = meta.numFeatures
        val nH = meta.hiddenWidth
        val nP = meta.lowerPieces
        val nI = vectors[0].size
        val lowerW = tensor("lower.W") // [nF, nH, nP, nI]
        val pad = tensor("lower.pad")  // [nF, nH, nP]
        val bias = tensor("lower.b")   // [nH, nP]
        val upperW = tensor("upper.W") // [nActions, nH]
        val upperB = tensor("upper.b")
        val actions = meta.actions

        // Precompute every (token, feature) contribution once: cached[t][f] is [nH * nP].
        val cached = Array(n) { t ->
            Array(nF) { f ->
                FloatArray(nH * nP) { j ->
                    val base = (f * nH * nP + j) * nI
                    var s = 0f
                    for (i in 0 until nI) s += vectors[t][i] * lowerW[base + i]
                    s
                }
            }
        }

        class Ent(val start: Int, var end: Int, val label: String)
        val ents = ArrayList<Ent>()
        var b = 0
        while (b < n) {
            val open = ents.isNotEmpty() && ents.last().end == -1
            val ids = intArrayOf(b, if (open) ents.last().start else -1, -1)
            if (ids[1] != -1) ids[2] = ids[0] - 1

            val acc = FloatArray(nH * nP)
            for (f in 0 until nF) {
                val id = ids[f]
                if (id < 0) for (j in acc.indices) acc[j] += pad[f * nH * nP + j]
                else for (j in acc.indices) acc[j] += cached[id][f][j]
            }
            val hidden = FloatArray(nH) { h ->
                var best = Float.NEGATIVE_INFINITY
                for (p in 0 until nP) best = maxOf(best, acc[h * nP + p] + bias[h * nP + p])
                best
            }

            val bufferLength = n - b
            var best = -1
            var bestScore = 0f
            for (a in actions.indices) {
                val (move, label) = actions[a]
                val valid = when (move) {
                    "B" -> !open && bufferLength >= 2 && label.isNotEmpty() && !isSpace[b]
                    "I" -> open && bufferLength >= 2 && label.isNotEmpty() && ents.last().label == label
                    "L" -> label.isNotEmpty() && open && ents.last().label == label
                    "U" -> label.isNotEmpty() && !open && !isSpace[b]
                    "O" -> !open
                    else -> false
                }
                if (!valid) continue
                var score = upperB[a]
                for (h in 0 until nH) score += hidden[h] * upperW[a * nH + h]
                if (best == -1 || score > bestScore) { best = a; bestScore = score }
            }
            check(best >= 0) { "no valid NER action" }

            when (actions[best].move) {
                "B" -> ents += Ent(b, -1, actions[best].label)
                "U" -> ents += Ent(b, b + 1, actions[best].label)
                "L" -> ents.last().end = b + 1
            }
            b++
        }
        return ents.filter { it.end != -1 }.map { NerEntity(it.start, it.end, it.label) }
    }

    val personLabels: Set<String> get() = meta.personLabels.toSet()
}

/**
 * [EntityExtractor] backed by the spaCy NER port: `_extract_names_with_ner` of RealtimeTranscriber, i.e. PERSON/PER
 * entities longer than two characters with a leading honorific ("Sr.", "Dra.", "Mr.", ...) removed.
 */
class SpacyNer(private val tokenizer: SpacyTokenizer, private val model: SpacyNerModel) : EntityExtractor {
    override val available: Boolean = true

    fun tokens(text: String): List<SpacyToken> = tokenizer.tokenize(text)

    fun normOf(token: SpacyToken): String = model.normOf(token)

    fun keysOf(token: SpacyToken): LongArray = model.attributeKeys(token)

    fun entities(text: String): List<Pair<NerEntity, String>> {
        val tokens = tokenizer.tokenize(text)
        if (tokens.isEmpty()) return emptyList()
        val vectors = model.tokenVectors(tokens.map(model::attributeKeys))
        val isSpace = BooleanArray(tokens.size) { i -> tokens[i].text.isNotEmpty() && tokens[i].text.all(SpacyTokenizer::isPySpace) }
        return model.decode(vectors, isSpace).map { ent ->
            ent to text.substring(tokens[ent.startToken].idx, tokens[ent.endToken - 1].end)
        }
    }

    override fun personNames(text: String): List<String> =
        entities(text)
            .filter { (ent, _) -> ent.label in model.personLabels }
            .map { (_, entText) -> entText.trim(SpacyTokenizer::isPySpace) }
            .filter { it.length > 2 }
            .map { it.replace(HONORIFIC, "") }

    companion object {
        private val HONORIFIC = PyText.regex("^(Sr\\.|Sra\\.|Dr\\.|Dra\\.|Mr\\.|Mrs\\.|Ms\\.)\\s+", ignoreCase = true)
        private val json = Json { ignoreUnknownKeys = true }

        /** Loads the exported model of one language ("en" or "pt") from a folder produced by scripts/export_ner.py. */
        fun fromFolder(folder: File): SpacyNer {
            val meta = json.decodeFromString(NerMeta.serializer(), File(folder, "model.json").readText())
            val norm = NormTables(json.decodeFromString(NormFile.serializer(), File(folder, "lexeme_norm.json").readText()))
            val tokenizer = SpacyTokenizer(TokenizerData.parse(File(folder, "tokenizer.json").readText()))
            return SpacyNer(tokenizer, SpacyNerModel(meta, File(folder, "weights.bin").readBytes(), norm))
        }

        /** [language] is "en" or "pt", like FusionConfig.language; anything else falls back to English. */
        fun fromAssets(context: Context, language: String): SpacyNer {
            val dir = if (language == "pt") "pt" else "en"
            val folder = File(context.filesDir, "assets/ner/$dir")
            for (name in listOf("model.json", "tokenizer.json", "lexeme_norm.json", "weights.bin")) {
                com.avtracker.mobile.voice.AssetFiles.materialize(context, "ner/$dir/$name")
            }
            return fromFolder(folder)
        }
    }
}
