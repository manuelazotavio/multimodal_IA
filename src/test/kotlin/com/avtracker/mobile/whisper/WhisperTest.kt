package com.avtracker.mobile.whisper

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import com.avtracker.mobile.TestAudio.readWav
import java.io.File
import java.util.Base64
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class WhisperTest {

    // ---- tokenizer --------------------------------------------------------------------------

    private fun b64(s: ByteArray) = Base64.getEncoder().encodeToString(s)

    @Test
    fun tokens_decodeConcatenatesBytesAndSkipsSpecials() {
        val vocab = WhisperTokens(listOf("Hello".toByteArray(), " world".toByteArray(), ByteArray(0)))
        assertEquals("Hello world", vocab.decode(listOf(0, 1, 2)))
    }

    @Test
    fun tokens_multiByteCharactersSplitAcrossTokensAreReassembled() {
        val bytes = "ã".toByteArray(Charsets.UTF_8) // 0xC3 0xA3, split over two BPE tokens
        val vocab = WhisperTokens(listOf(byteArrayOf(bytes[0]), byteArrayOf(bytes[1]), " ok".toByteArray()))
        assertEquals("ã ok", vocab.decode(listOf(0, 1, 2)))
    }

    @Test
    fun tokens_parseReadsBase64Lines() {
        val text = listOf(b64("Bom".toByteArray()), b64(" dia".toByteArray()), "").joinToString("\n")
        assertEquals("Bom dia", WhisperTokens.parse(text).decode(listOf(0, 1, 2)))
    }

    // ---- real model -------------------------------------------------------------------------

    private val folder = File("src/main/assets/whisper")
    private val modelReady get() = File(folder, "encoder.onnx").exists() && File(folder, "decoder.onnx").exists()

    private val expected = Json.parseToJsonElement(File("src/test/resources/tts_expected.json").readText()).jsonObject

    @Test
    fun transcribesEnglishSpeechLikeTheHuggingFaceReference() {
        assumeTrue("whisper assets not exported (run scripts/export_whisper.py)", modelReady)

        WhisperTranscriber.fromFolder(folder).use { whisper ->
            val result = assertNotNull(whisper.transcribe(readWav("tts_en.wav"), "en"))
            val ref = expected.getValue("en").jsonObject

            assertEquals(ref.getValue("text").jsonPrimitive.content, result.text)
            assertEquals(ref.getValue("tokens").jsonArray.map { it.jsonPrimitive.int }, result.tokens)
            assertTrue(abs(result.avgLogProb - ref.getValue("avg_logprob").jsonPrimitive.float) < 0.02f, "avgLogProb ${result.avgLogProb}")
            assertTrue(result.noSpeechProb < 0.1f, "clear speech must not look like no-speech: ${result.noSpeechProb}")
        }
    }

    @Test
    fun transcribesPortugueseSpeech() {
        assumeTrue("whisper assets not exported (run scripts/export_whisper.py)", modelReady)

        WhisperTranscriber.fromFolder(folder).use { whisper ->
            val result = assertNotNull(whisper.transcribe(readWav("tts_pt.wav"), "pt"))
            assertEquals(expected.getValue("pt").jsonObject.getValue("text").jsonPrimitive.content, result.text)
        }
    }

    @Test
    fun silenceScoresAsNoSpeechAndUnknownLanguageIsRejected() {
        assumeTrue("whisper assets not exported (run scripts/export_whisper.py)", modelReady)

        WhisperTranscriber.fromFolder(folder).use { whisper ->
            assertNull(whisper.transcribe(readWav("tts_en.wav"), "xx"))
            assertTrue(whisper.supports("pt") && !whisper.supports("xx"))

            val silence = assertNotNull(whisper.transcribe(FloatArray(16_000 * 3), "en"))
            // Whisper hallucinates on pure silence, which is why RealtimeTranscriber filters on no_speech_prob.
            assertTrue(silence.noSpeechProb > 0.3f || silence.avgLogProb < -0.5f, "silence: $silence")
        }
    }
}
