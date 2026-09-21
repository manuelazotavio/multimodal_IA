package com.avtracker.mobile.ner

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The Kotlin dependency parser against spaCy's own: identical heads, labels and sentence starts, and, through them, the
 * entities the full pipeline finds (an entity never crosses a sentence boundary).
 */
class ParserParityTest {

    private val fixtures: JsonObject by lazy {
        Json.parseToJsonElement(File("src/test/resources/parity/parser_fixtures.json").readText()).jsonObject
    }

    private fun ready(lang: String) = File("src/main/assets/ner/$lang/parser.json").exists()

    private fun check(lang: String) {
        assumeTrue("parser assets not exported (run scripts/export_parser.py)", ready(lang))
        val ner = SpacyNer.fromFolder(File("src/main/assets/ner/$lang"))
        val cases = fixtures.getValue(lang).jsonArray
        var parseFailures = 0
        var entityFailures = 0
        val report = StringBuilder()

        for (c in cases) {
            val case = c.jsonObject
            val text = case.getValue("text").jsonPrimitive.content
            val expectedHeads = case.getValue("heads").jsonArray.map { it.jsonPrimitive.int }
            val expectedDeps = case.getValue("deps").jsonArray.map { it.jsonPrimitive.content }
            val expectedSents = case.getValue("sents").jsonArray.map { it.jsonPrimitive.boolean }

            val parse = ner.parse(text)!!
            if (parse.heads.toList() != expectedHeads || parse.deps != expectedDeps || parse.sentenceStart.toList() != expectedSents) {
                parseFailures++
                if (report.length < 4000) {
                    report.append("PARSE '${text.take(100)}'\n  spacy:  $expectedHeads $expectedDeps $expectedSents\n  kotlin: ${parse.heads.toList()} ${parse.deps} ${parse.sentenceStart.toList()}\n")
                }
            }

            val expectedEnts = case.getValue("ents").jsonArray.map { e -> e.jsonArray.let { Triple(it[0].jsonPrimitive.int, it[1].jsonPrimitive.int, it[2].jsonPrimitive.content) } }
            val ents = ner.entities(text).map { (e, _) -> Triple(e.startToken, e.endToken, e.label) }
            if (ents != expectedEnts) {
                entityFailures++
                if (report.length < 4000) report.append("ENTS '${text.take(100)}'\n  spacy:  $expectedEnts\n  kotlin: $ents\n")
            }
        }
        assertTrue(parseFailures == 0 && entityFailures == 0, "$lang: $parseFailures parses and $entityFailures entity lists of ${cases.size} differ from spaCy\n$report")
    }

    @Test fun english_parseAndEntitiesMatchSpacy() = check("en")

    @Test fun portuguese_parseAndEntitiesMatchSpacy() = check("pt")

    @Test
    fun aSentenceBoundaryStopsAnEntity() {
        assumeTrue("parser assets not exported", ready("en"))
        val ner = SpacyNer.fromFolder(File("src/main/assets/ner/en"))
        // spaCy: PER "Pedro" only; without the parser the model ran on into "Pedro. Silva said"
        val names = ner.personNames("Thanks, Pedro. Silva said we should wait.")
        assertTrue("Pedro" in names && names.none { it.contains("said") }, "names: $names")
        assertNotNull(ner.parse("Thanks, Pedro. Silva said we should wait."))
    }
}
