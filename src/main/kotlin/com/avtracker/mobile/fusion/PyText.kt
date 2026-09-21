package com.avtracker.mobile.fusion

import java.util.regex.Pattern

/** Helpers that reproduce Python string/regex semantics where they differ from Kotlin's defaults. */
object PyText {
    /** A regex with Python 3 `str` semantics: `\s`, `\w`, `\b` are Unicode-aware; optional IGNORECASE. */
    fun regex(pattern: String, ignoreCase: Boolean = false): Regex {
        var flags = Pattern.UNICODE_CHARACTER_CLASS
        if (ignoreCase) flags = flags or Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE
        return Pattern.compile(pattern, flags).toRegex()
    }

    /** Python str.title(): the first letter of each run of letters upper-cased, the rest lower-cased. */
    fun title(s: String): String {
        val out = StringBuilder(s.length)
        var previousIsLetter = false
        for (ch in s) {
            if (ch.isLetter()) {
                out.append(if (previousIsLetter) ch.lowercaseChar() else ch.uppercaseChar())
                previousIsLetter = true
            } else {
                out.append(ch)
                previousIsLetter = false
            }
        }
        return out.toString()
    }

    /** Python re.sub(r'\s+\d+$', '', s): strips a trailing homonym counter such as "Vinícius 2". */
    private val HOMONYM_SUFFIX = regex("\\s+\\d+$")
    fun stripHomonymSuffix(s: String): String = s.replace(HOMONYM_SUFFIX, "")
}
