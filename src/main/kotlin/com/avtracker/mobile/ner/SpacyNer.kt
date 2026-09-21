package com.avtracker.mobile.ner

import android.content.Context
import com.avtracker.mobile.fusion.EntityExtractor
import com.avtracker.mobile.fusion.PyText
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

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
 * The dependency parser that runs before `ner` in the spaCy pipeline matters through sentence boundaries, which forbid an
 * entity from spanning two sentences: [decode] takes them from [SpacyParser] when it is available.
 */
class SpacyNerModel(private val meta: NerMeta, weights: ByteArray, private val norms: NormTables) {
    private val nn = NnWeights(meta.tensors, weights)
    private val tok2vec = Tok2Vec(nn, meta.width, meta.seeds, meta.rows, meta.encoderDepth, meta.encoderPad, meta.maxoutPieces)
    private val scorer = TransitionScorer(nn, meta.numFeatures, meta.hiddenWidth, meta.lowerPieces, meta.actions.size)

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

    /** Token vectors [n, hiddenWidth] from the attribute keys of each token. */
    fun tokenVectors(keys: List<LongArray>): Array<FloatArray> = tok2vec.encode(keys)

    /**
     * Greedy BILUO decoding of token vectors into entities. [isSpace] marks whitespace tokens, which cannot start one;
     * [sentStart] marks the tokens the parser put at the start of a sentence (no entity may run into one).
     */
    fun decode(vectors: Array<FloatArray>, isSpace: BooleanArray, sentStart: BooleanArray? = null): List<NerEntity> {
        val n = vectors.size
        if (n == 0) return emptyList()
        scorer.prepare(vectors)
        val actions = meta.actions

        class Ent(val start: Int, var end: Int, val label: String)
        val ents = ArrayList<Ent>()
        var b = 0
        while (b < n) {
            val open = ents.isNotEmpty() && ents.last().end == -1
            val ids = intArrayOf(b, if (open) ents.last().start else -1, -1)
            if (ids[1] != -1) ids[2] = ids[0] - 1

            val bufferLength = n - b
            val nextStartsSentence = sentStart != null && b + 1 < n && sentStart[b + 1]
            val best = scorer.best(ids) { a ->
                val (move, label) = actions[a]
                when (move) {
                    "B" -> !open && bufferLength >= 2 && label.isNotEmpty() && !nextStartsSentence && !isSpace[b]
                    "I" -> open && bufferLength >= 2 && label.isNotEmpty() && ents.last().label == label && !nextStartsSentence
                    "L" -> label.isNotEmpty() && open && ents.last().label == label
                    "U" -> label.isNotEmpty() && !open && !isSpace[b]
                    "O" -> !open
                    else -> false
                }
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
class SpacyNer(
    private val tokenizer: SpacyTokenizer,
    private val model: SpacyNerModel,
    /** The dependency parser whose sentence boundaries limit the entities; without it the whole text is one sentence. */
    private val parser: SpacyParser? = null
) : EntityExtractor {
    override val available: Boolean = true

    fun tokens(text: String): List<SpacyToken> = tokenizer.tokenize(text)

    fun normOf(token: SpacyToken): String = model.normOf(token)

    fun keysOf(token: SpacyToken): LongArray = model.attributeKeys(token)

    private fun isSpace(tokens: List<SpacyToken>) =
        BooleanArray(tokens.size) { i -> tokens[i].text.isNotEmpty() && tokens[i].text.all(SpacyTokenizer::isPySpace) }

    /** The six attributes of the shared tok2vec: NORM, PREFIX, SUFFIX, SHAPE, SPACY (a space follows) and IS_SPACE. */
    private fun parserKeys(text: String, tokens: List<SpacyToken>, isSpace: BooleanArray): List<LongArray> =
        tokens.mapIndexed { i, t ->
            val four = model.attributeKeys(t)
            val spacy = if (t.end < text.length && text[t.end] == ' ') 1L else 0L
            longArrayOf(four[0], four[1], four[2], four[3], spacy, if (isSpace[i]) 1L else 0L)
        }

    /** The dependency parse of [text], or null when the parser was not loaded. */
    fun parse(text: String): ParseResult? {
        val p = parser ?: return null
        val tokens = tokenizer.tokenize(text)
        val space = isSpace(tokens)
        return p.parse(parserKeys(text, tokens, space), space)
    }

    fun entities(text: String): List<Pair<NerEntity, String>> {
        val tokens = tokenizer.tokenize(text)
        if (tokens.isEmpty()) return emptyList()
        val vectors = model.tokenVectors(tokens.map(model::attributeKeys))
        val isSpace = isSpace(tokens)
        val sentenceStart = parser?.parse(parserKeys(text, tokens, isSpace), isSpace)?.sentenceStart
        return model.decode(vectors, isSpace, sentenceStart).map { ent ->
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
            val parserFiles = File(folder, "parser.json") to File(folder, "parser_weights.bin")
            val parser = if (parserFiles.first.exists() && parserFiles.second.exists()) {
                SpacyParser(json.decodeFromString(ParserMeta.serializer(), parserFiles.first.readText()), parserFiles.second.readBytes())
            } else null
            return SpacyNer(tokenizer, SpacyNerModel(meta, File(folder, "weights.bin").readBytes(), norm), parser)
        }

        /** [language] is "en" or "pt", like FusionConfig.language; anything else falls back to English. */
        fun fromAssets(context: Context, language: String): SpacyNer {
            val dir = if (language == "pt") "pt" else "en"
            val folder = File(context.filesDir, "assets/ner/$dir")
            val available = context.assets.list("ner/$dir").orEmpty().toSet()
            for (name in listOf("model.json", "tokenizer.json", "lexeme_norm.json", "weights.bin", "parser.json", "parser_weights.bin")) {
                if (name in available) com.avtracker.mobile.voice.AssetFiles.materialize(context, "ner/$dir/$name")
            }
            return fromFolder(folder)
        }
    }
}
