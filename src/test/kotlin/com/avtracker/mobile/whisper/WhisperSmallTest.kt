package com.avtracker.mobile.whisper

import com.avtracker.mobile.TestAudio
import org.junit.Assume.assumeTrue
import java.io.File
import java.text.Normalizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class WhisperSmallTest {
    @Test fun portugueseSpeechSurvivesVadAndProductionDecoding() {
        val folder = File("src/main/assets/whisper-small")
        val vadFile = File("src/main/assets/models/silero_vad_v6.onnx")
        assumeTrue(File(folder, "meta.json").exists() && vadFile.exists())
        // The transcriber owns and closes the VAD along with its ONNX sessions.
        WhisperTranscriber.fromFolder(folder, vad = SileroVad.fromFile(vadFile), options = WhisperModel.SMALL.decodingOptions()).use { recognizer ->
            val result = assertNotNull(recognizer.transcribe(TestAudio.readWav("tts_pt.wav"), "pt"))
            assertEquals(words("Bom dia a todos hoje vamos testar o reconhecimento de fala no celular"), words(result.text))
            assertTrue(result.noSpeechProb < 0.1f)
        }
    }

    @Test fun smallLoadsAndTranscribesPortugueseReference() {
        val folder = File("src/main/assets/whisper-small")
        assumeTrue(File(folder, "meta.json").exists())
        WhisperTranscriber.fromFolder(folder).use { recognizer ->
            assertEquals("small", recognizer.modelName)
            assertTrue(recognizer.supports("pt"))
            val result = assertNotNull(recognizer.transcribeGreedy(TestAudio.readWav("tts_pt.wav"), "pt"))
            val expected = "Bom dia a todos hoje vamos testar o reconhecimento de fala no celular"
            assertEquals(words(expected), words(result.text), "Portuguese reference: ${result.text}")
        }
    }

    private fun words(text: String): List<String> = Normalizer.normalize(text.lowercase(), Normalizer.Form.NFD)
        .replace(Regex("\\p{M}+"), "").replace(Regex("[^a-z0-9 ]"), " ")
        .trim().split(Regex("\\s+"))
}
