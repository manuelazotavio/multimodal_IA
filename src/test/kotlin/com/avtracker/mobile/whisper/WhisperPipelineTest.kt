package com.avtracker.mobile.whisper

import com.avtracker.mobile.TestAudio
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The transcription pipeline against the ORIGINAL faster-whisper (scripts/parity/whisper_reference.py): the same VAD
 * chunks on every clip, and the same text from the beam search with the initial prompt, whose scores must agree closely.
 */
class WhisperPipelineTest {
    private val folder = File("src/main/assets/whisper")
    private val vadFile = File("src/main/assets/models/silero_vad_v6.onnx")
    private val ready get() = File(folder, "encoder.onnx").exists() && File(folder, "decoder.onnx").exists() && vadFile.exists()

    private val clips: List<JsonObject> by lazy {
        Json.parseToJsonElement(File("src/test/resources/whisper_parity/expected.json").readText()).jsonArray.map { it.jsonObject }
    }

    private fun audio(name: String) = TestAudio.readWav("whisper_parity/$name.wav")

    @Test
    fun vad_findsTheSameSpeechChunksAsFasterWhisper() {
        assumeTrue("VAD model not exported (run scripts/export_models.py)", vadFile.exists())
        SileroVad.fromFile(vadFile).use { vad ->
            for (clip in clips) {
                val name = clip.getValue("name").jsonPrimitive.content
                val expected = clip.getValue("vad").jsonArray.map { it.jsonArray[0].jsonPrimitive.int to it.jsonArray[1].jsonPrimitive.int }
                val actual = vad.speechTimestamps(audio(name)).map { it.start to it.end }
                assertEquals(expected, actual, "VAD chunks of $name")
            }
        }
    }

    @Test
    fun collectChunks_concatenatesTheSpeechOnly() {
        val audio = FloatArray(100) { it.toFloat() }
        val out = SileroVad.collectChunks(audio, listOf(SpeechChunk(10, 13), SpeechChunk(50, 52)))
        assertEquals(listOf(10f, 11f, 12f, 50f, 51f), out.toList())
        assertEquals(0, SileroVad.collectChunks(audio, emptyList()).size)
    }

    @Test
    fun transcription_matchesFasterWhisperOnEveryClip() {
        assumeTrue("whisper/VAD assets not exported", ready)
        WhisperTranscriber.fromFolder(folder, vad = SileroVad.fromFile(vadFile)).use { whisper ->
            for (clip in clips) {
                val name = clip.getValue("name").jsonPrimitive.content
                val language = clip.getValue("language").jsonPrimitive.content
                val segments = clip.getValue("segments").jsonArray.map { it.jsonObject }
                val result = whisper.transcribe(audio(name), language)

                if (segments.isEmpty()) {
                    assertNull(result, "$name: faster-whisper found nothing")
                    continue
                }
                assertNotNull(result, "$name: expected ${segments.first().getValue("text")}")
                val expectedText = segments.joinToString(" ") { it.getValue("text").jsonPrimitive.content }.trim()
                assertEquals(expectedText, result.text, "text of $name")
                val expectedLogProb = segments.map { it.getValue("avg_logprob").jsonPrimitive.double }.average()
                assertTrue(abs(expectedLogProb - result.avgLogProb) < 0.05, "$name avg_logprob: expected $expectedLogProb, got ${result.avgLogProb}")
                val expectedNoSpeech = segments.map { it.getValue("no_speech_prob").jsonPrimitive.double }.average()
                assertTrue(abs(expectedNoSpeech - result.noSpeechProb) < 0.05, "$name no_speech: expected $expectedNoSpeech, got ${result.noSpeechProb}")
            }
        }
    }

    @Test
    fun compressionRatio_matchesPythonsZlib() {
        // python: len(b) / len(zlib.compress(b)) with the sizes below
        assertEquals(160.0 / 27.0, WhisperTranscriber.compressionRatio("Hello everyone. ".repeat(10)), 1e-12)
        assertEquals(15.0 / 23.0, WhisperTranscriber.compressionRatio("a b c d e f g h"), 1e-12)
        assertEquals(40.0 / 45.0, WhisperTranscriber.compressionRatio("Olá, tudo bem? Não sei bem — ação!"), 1e-12)
    }
}
