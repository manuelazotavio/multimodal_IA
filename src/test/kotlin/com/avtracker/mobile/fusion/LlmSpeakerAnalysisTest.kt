package com.avtracker.mobile.fusion

import com.avtracker.mobile.voice.SpeakerVerifier
import com.sun.net.httpserver.HttpServer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.InetSocketAddress
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The pieces of the LLM speaker analysis that the engine parity scenarios do not cover: repr, the HTTP client, threading. */
class LlmSpeakerAnalysisTest {

    @Test
    fun pyRepr_matchesPythonsRepr() {
        // python: repr(...) of each of these
        assertEquals("'Manuela'", PyText.repr("Manuela"))
        assertEquals("\"O'Neil\"", PyText.repr("O'Neil"))
        assertEquals("'He said \"hi\"'", PyText.repr("He said \"hi\""))
        assertEquals("'both \\' and \"'", PyText.repr("both ' and \""))
        assertEquals("'tab\\there'", PyText.repr("tab\there"))
        assertEquals("'back\\\\slash'", PyText.repr("back\\slash"))
        assertEquals("'João Nascimento'", PyText.repr("João Nascimento"))
        assertEquals("'line\\nbreak'", PyText.repr("line\nbreak"))
        assertEquals("'\\x7f\\x01'", PyText.repr("\u007f\u0001"))
        assertEquals("{'spk_001': 'Manuela', 'spk_002': \"O'Neil\", 'spk_010': 'Say \"x\"'}",
            PyText.dictRepr(linkedMapOf("spk_001" to "Manuela", "spk_002" to "O'Neil", "spk_010" to "Say \"x\"")))
    }

    @Test
    fun hfClient_postsThePromptWithTheTokenAndReadsBothAnswerShapes() {
        var seenAuth: String? = null
        var seenBody: String? = null
        var answer = """[{"generated_text": "{\"spk_001\": \"Ana\"}"}]"""
        var status = 200

        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/generate") { exchange ->
            seenAuth = exchange.requestHeaders.getFirst("Authorization")
            seenBody = exchange.requestBody.readBytes().toString(Charsets.UTF_8)
            val bytes = answer.toByteArray()
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
        server.start()
        try {
            val client = HfInferenceClient("hf_secret", endpoint = "http://127.0.0.1:${server.address.port}/generate", timeoutMs = 5_000)

            assertEquals("""{"spk_001": "Ana"}""", client.generate("Olá, prompt", 150, 0.1))
            assertEquals("Bearer hf_secret", seenAuth)
            val body = Json.parseToJsonElement(assertNotNull(seenBody)).jsonObject
            assertEquals("Olá, prompt", body.getValue("inputs").jsonPrimitive.content)
            val parameters = body.getValue("parameters").jsonObject
            assertEquals("150", parameters.getValue("max_new_tokens").jsonPrimitive.content)
            assertEquals("0.1", parameters.getValue("temperature").jsonPrimitive.content)

            answer = """{"generated_text": "plain"}""" // a text-generation-inference server answers with an object
            assertEquals("plain", client.generate("p", 150, 0.1))

            status = 503
            answer = """{"error": "Model is loading"}"""
            val failure = assertFailsWith<java.io.IOException> { client.generate("p", 150, 0.1) }
            assertTrue("503" in failure.message.orEmpty(), failure.message)
        } finally {
            server.stop(0)
        }
    }

    private fun analysis(client: LlmClient?, registry: IdentityRegistry, book: SpeakerBook, background: Boolean, entries: List<TranscriptEntry>): LlmSpeakerAnalysis {
        val voice = object : VoiceEncoder {
            override fun embed(audio: FloatArray): FloatArray? = null
            override fun embedNormalized(audio: FloatArray): FloatArray? = null
        }
        val naming = SpeakerNaming(registry, SpeakerVerifier.fromMap(emptyMap()), book, voice)
        return LlmSpeakerAnalysis(client, registry, book, naming, numSpeakers = null, transcript = { entries }, segmentCount = { 10 }, background = background)
    }

    private fun entry(id: String, name: String?, text: String) = TranscriptEntry(id, name, text, "ASD", 1.0)

    @Test
    fun withoutAClientNothingHappens() {
        val registry = IdentityRegistry()
        val pid = registry.newPersonId().also { registry.setName(it, it) }
        val a = analysis(null, registry, SpeakerBook(), false, listOf(entry(pid, null, "Bom dia.")))
        a.maybeRun()
        a.identifySpeakers()
        assertEquals(pid, registry.nameOf(pid))
        assertTrue(!a.enabled)
    }

    @Test
    fun aBackgroundAnswerIsAppliedOnTheFusionThreadAtTheNextSegment() {
        val registry = IdentityRegistry()
        val pid = registry.newPersonId().also { registry.setName(it, it) }
        val book = SpeakerBook()
        val asked = CountDownLatch(1)
        val client = LlmClient { _, _, _ -> asked.countDown(); """Aqui: {"$pid": "maria souza"}""" }
        val a = analysis(client, registry, book, true, listOf(entry(pid, null, "Bom dia a todos.")))

        a.maybeRun() // the tenth segment: the request goes out on its own thread
        assertTrue(asked.await(5, TimeUnit.SECONDS))
        Thread.sleep(200)
        assertEquals(pid, registry.nameOf(pid), "nothing is touched until the fusion thread applies the answer")

        a.applyPending()
        assertEquals("Maria Souza", registry.nameOf(pid)) // str.title()
        assertEquals(pid, registry.personFor("Maria Souza"))
        assertTrue(pid in book.identifiedSpeakers)
    }

    @Test
    fun theAnswerNeverOverwritesARealNameOrTakesANamedAlreadyHeld() {
        val registry = IdentityRegistry()
        val real = registry.newPersonId().also { registry.setName(it, "Pedro"); registry.bind("Pedro", it) }
        val generic = registry.newPersonId().also { registry.setName(it, it) }
        val a = analysis(null, registry, SpeakerBook(), false, emptyList())

        a.apply("""{"$real": "Outro", "$generic": "pedro"}""")
        assertEquals("Pedro", registry.nameOf(real), "a confirmed name is kept")
        assertEquals(generic, registry.nameOf(generic), "Pedro belongs to someone else")

        a.apply("no braces at all")
        a.apply("""{"a": {"b": 1}}""") // the regex slice is not valid JSON: ignored, not fatal
        assertNull(registry.personFor("Outro"))
    }
}
