package com.avtracker.mobile.ner

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.util.regex.Pattern

@Serializable
data class TokenSpec(val orth: String, val norm: String? = null)

@Serializable
data class TokenizerRule(val string: String, val tokens: List<TokenSpec>)

/** tokenizer.json written by scripts/export_ner.py. */
@Serializable
data class TokenizerData(
    val prefix: String,
    val suffix: String,
    val infix: String,
    val urlMatch: String? = null,
    val tokenMatch: String? = null,
    val fasterHeuristics: Boolean = true,
    val rules: List<TokenizerRule>
) {
    companion object {
        private val json = Json { ignoreUnknownKeys = true }
        fun parse(text: String): TokenizerData = json.decodeFromString(serializer(), text)
    }
}

/** A token: [idx] is its character offset in the text, [norm] an override from a special-case rule. */
class SpacyToken(val idx: Int, val text: String, val norm: String? = null) {
    val end: Int get() = idx + text.length
}

/**
 * Port of spaCy's rule-based Tokenizer (3.8): whitespace-delimited chunks, exceptions, then prefixes, suffixes and
 * infixes peeled off with the language's regexes. Patterns are Python `re` patterns; they are compiled with
 * UNICODE_CHARACTER_CLASS to match Python 3's Unicode-aware `\w`, `\d`, `\s`.
 */
class SpacyTokenizer(data: TokenizerData) {
    private val prefixRegex = compile(data.prefix)
    private val suffixRegex = compile(data.suffix)
    private val infixRegex = compile(data.infix)
    private val urlRegex = data.urlMatch?.let(::compile)
    private val tokenMatchRegex = data.tokenMatch?.let(::compile)
    private val specials: Map<String, List<TokenSpec>> = data.rules.associate { it.string to it.tokens }

    /** Python spells astral code points with an 8-digit U escape, Java as x{...}; all other syntax in the patterns is shared. */
    private fun compile(p: String): Pattern =
        Pattern.compile(p.replace(PY_ASTRAL) { "\\x{${it.groupValues[1]}}" }, Pattern.UNICODE_CHARACTER_CLASS)

    fun tokenize(text: String): List<SpacyToken> {
        val out = ArrayList<SpacyToken>()
        if (text.isEmpty()) return out

        var start = 0
        var inWs = isPySpace(text[0])
        for (i in text.indices) {
            val uc = text[i]
            if (isPySpace(uc) != inWs) {
                if (start < i) addChunk(out, text, start, i)
                // A single ' ' is the trailing whitespace of the previous token; anything else is its own chunk.
                start = if (uc == ' ') i + 1 else i
                inWs = !inWs
            }
        }
        if (start < text.length) addChunk(out, text, start, text.length)
        return out
    }

    private fun addChunk(out: MutableList<SpacyToken>, text: String, from: Int, to: Int) {
        val span = text.substring(from, to)
        val pieces = tokenizeSpan(span)
        var idx = from
        for (p in pieces) {
            out += SpacyToken(idx, p.orth, p.norm)
            idx += p.orth.length
        }
    }

    private fun tokenizeSpan(span: String): List<TokenSpec> {
        specials[span]?.let { return it }

        val prefixes = ArrayList<String>()
        val suffixes = ArrayList<String>()
        val core = splitAffixes(span, prefixes, suffixes)

        val pieces = ArrayList<TokenSpec>()
        prefixes.forEach { pieces += TokenSpec(it) }
        if (core.isNotEmpty()) {
            val special = specials[core]
            if (special != null) {
                pieces += special
            } else if ((tokenMatchRegex?.matcher(core)?.lookingAt() == true) || (urlRegex?.matcher(core)?.lookingAt() == true)) {
                pieces += TokenSpec(core)
            } else {
                val matcher = infixRegex.matcher(core)
                var any = false
                var start = 0
                val startBeforeInfixes = start
                while (matcher.find()) {
                    any = true
                    val infixStart = matcher.start()
                    val infixEnd = matcher.end()
                    if (infixStart == startBeforeInfixes) continue
                    if (infixStart != start) pieces += TokenSpec(core.substring(start, infixStart))
                    if (infixStart != infixEnd) pieces += TokenSpec(core.substring(infixStart, infixEnd))
                    start = infixEnd
                }
                if (!any) {
                    pieces += TokenSpec(core)
                } else {
                    val rest = core.substring(start)
                    if (rest.isNotEmpty()) pieces += TokenSpec(rest)
                }
            }
        }
        for (i in suffixes.indices.reversed()) pieces += TokenSpec(suffixes[i])
        return pieces
    }

    /** spaCy `Tokenizer._split_affixes`: peel prefixes and suffixes until nothing changes or an exception remains. */
    private fun splitAffixes(input: String, prefixes: MutableList<String>, suffixes: MutableList<String>): String {
        var string = input
        var lastSize = 0
        while (string.isNotEmpty() && string.length != lastSize) {
            if (tokenMatchRegex?.matcher(string)?.lookingAt() == true) break
            if (string in specials) break
            lastSize = string.length

            val preLen = findLength(prefixRegex, string)
            var prefix = ""
            var minusPre = ""
            if (preLen != 0) {
                prefix = string.substring(0, preLen)
                minusPre = string.substring(preLen)
                if (minusPre.isNotEmpty() && minusPre in specials) {
                    string = minusPre
                    prefixes += prefix
                    break
                }
            }
            val sufLen = findLength(suffixRegex, string.substring(preLen))
            var suffix = ""
            var minusSuf = ""
            if (sufLen != 0) {
                suffix = string.substring(string.length - sufLen)
                minusSuf = string.substring(0, string.length - sufLen)
                if (minusSuf.isNotEmpty() && minusSuf in specials) {
                    string = minusSuf
                    suffixes += suffix
                    break
                }
            }
            if (preLen != 0 && sufLen != 0 && (preLen + sufLen) <= string.length) {
                string = string.substring(preLen, string.length - sufLen)
                prefixes += prefix
                suffixes += suffix
            } else if (preLen != 0) {
                string = minusPre
                prefixes += prefix
            } else if (sufLen != 0) {
                string = minusSuf
                suffixes += suffix
            }
        }
        return string
    }

    /** `find_prefix` / `find_suffix`: length of the leftmost match, 0 if there is none. */
    private fun findLength(regex: Pattern, string: String): Int {
        val m = regex.matcher(string)
        return if (m.find()) m.end() - m.start() else 0
    }

    companion object {
        private val PY_ASTRAL = Regex("\\\\U([0-9A-Fa-f]{8})")

        /** Python `str.isspace()`. Java's Character.isWhitespace excludes the non-breaking spaces and U+0085. */
        fun isPySpace(c: Char): Boolean = Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'
    }
}
