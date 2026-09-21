package com.avtracker.mobile.db

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** `json.dumps(value, ensure_ascii=False)`: the same separators (", " and ": ") and escapes as Python. */
object PyJson {
    fun dumps(element: JsonElement): String = StringBuilder().also { write(element, it) }.toString()

    private fun write(e: JsonElement, out: StringBuilder) {
        when (e) {
            is JsonNull -> out.append("null")
            is JsonPrimitive -> if (e.isString) string(e.content, out) else out.append(e.content)
            is JsonArray -> {
                out.append('[')
                e.forEachIndexed { i, v -> if (i > 0) out.append(", "); write(v, out) }
                out.append(']')
            }
            is JsonObject -> {
                out.append('{')
                var first = true
                for ((k, v) in e) {
                    if (!first) out.append(", ")
                    first = false
                    string(k, out)
                    out.append(": ")
                    write(v, out)
                }
                out.append('}')
            }
        }
    }

    private fun string(s: String, out: StringBuilder) {
        out.append('"')
        for (c in s) {
            when {
                c == '"' -> out.append("\\\"")
                c == '\\' -> out.append("\\\\")
                c == '\n' -> out.append("\\n")
                c == '\r' -> out.append("\\r")
                c == '\t' -> out.append("\\t")
                c == '\b' -> out.append("\\b")
                c == '\u000c' -> out.append("\\f")
                c < ' ' -> out.append("\\u%04x".format(c.code))
                else -> out.append(c)
            }
        }
        out.append('"')
    }
}
