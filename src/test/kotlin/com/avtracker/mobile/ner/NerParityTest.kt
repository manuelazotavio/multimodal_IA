package com.avtracker.mobile.ner

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The Kotlin NER against spaCy itself: identical hashes, tokens, attributes, entities and person names. */
class NerParityTest {

    private val fixtures: JsonObject by lazy {
        Json.parseToJsonElement(File("src/test/resources/parity/ner_fixtures.json").readText()).jsonObject
    }

    private fun ready(lang: String) = File("src/main/assets/ner/$lang/model.json").exists()

    @Test
    fun hashString_matchesSpacy() {
        for ((s, expected) in fixtures.getValue("hashes").jsonObject) {
            assertEquals(java.lang.Long.parseUnsignedLong(expected.jsonPrimitive.content), SpacyHash.hashString(s), "hash_string('$s')")
        }
    }

    private fun check(lang: String) {
        assumeTrue("NER assets not exported (run scripts/export_ner.py)", ready(lang))
        val ner = SpacyNer.fromFolder(File("src/main/assets/ner/$lang"))
        val cases = fixtures.getValue(lang).jsonArray
        var tokenFailures = 0
        val failures = StringBuilder()

        for (c in cases) {
            val case = c.jsonObject
            val text = case.getValue("text").jsonPrimitive.content

            val expectedTokens = case.getValue("tokens").jsonArray.map {
                val a = it.jsonArray
                Triple(text.offsetByCodePoints(0, a[0].jsonPrimitive.content.toInt()), a[1].jsonPrimitive.content, a[2].jsonPrimitive.content)
            }
            val tokens = ner.tokens(text)
            val actualTokens = tokens.map { Triple(it.idx, it.text, ner.normOf(it)) }
            if (actualTokens != expectedTokens) {
                tokenFailures++
                if (failures.length < 3000) failures.append("TOKENS '$text'\n  spacy: $expectedTokens\n  kotlin: $actualTokens\n")
                continue
            }

            case["keys"]?.let { keys ->
                val expected = keys.jsonArray.map { row -> row.jsonArray.map { java.lang.Long.parseUnsignedLong(it.jsonPrimitive.content) } }
                val actual = tokens.map { ner.keysOf(it).toList() }
                if (expected != actual && failures.length < 3000) failures.append("KEYS '$text'\n  spacy: $expected\n  kotlin: $actual\n")
                if (expected != actual) tokenFailures++
            }

            val expectedEnts = case.getValue("ents").jsonArray.map {
                val a = it.jsonArray
                listOf(a[0].jsonPrimitive.content, a[1].jsonPrimitive.content, a[2].jsonPrimitive.content, a[3].jsonPrimitive.content)
            }
            val actualEnts = ner.entities(text).map { (e, t) -> listOf(e.startToken.toString(), e.endToken.toString(), e.label, t) }
            val expectedNames = case.getValue("names").jsonArray.map { it.jsonPrimitive.content }
            if (actualEnts != expectedEnts || ner.personNames(text) != expectedNames) {
                tokenFailures++
                if (failures.length < 3000) failures.append("ENTS '$text'\n  spacy: $expectedEnts $expectedNames\n  kotlin: $actualEnts ${ner.personNames(text)}\n")
            }
        }
        assertTrue(tokenFailures == 0, "$lang: $tokenFailures of ${cases.size} cases differ from spaCy\n$failures")
    }

    @Test fun english_matchesSpacyOnRealTranscriptsAndTrickyTokenisation() = check("en")

    @Test fun portuguese_matchesSpacyOnRealTranscriptsAndTrickyTokenisation() = check("pt")
}
