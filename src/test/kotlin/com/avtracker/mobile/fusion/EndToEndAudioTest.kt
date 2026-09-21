package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.diarization.Diarizer
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.voice.VoiceEmbedder
import com.avtracker.mobile.whisper.TranscriptionResult
import com.avtracker.mobile.whisper.WhisperTranscriber
import org.junit.Assume.assumeTrue
import com.avtracker.mobile.TestAudio.readWav
import java.io.File
import kotlin.test.Test
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The whole audio path with the real exported models and the production Kotlin code:
 * segmentation-3.0 turns -> ECAPA voice embeddings -> Whisper -> speaker attribution.
 * Two synthetic voices (Windows TTS: English and Brazilian Portuguese) separated by silence.
 */
class EndToEndAudioTest {

    private val models = File("src/main/assets/models")
    private val whisperDir = File("src/main/assets/whisper")
    private val ready get() = File(models, "pyannote_segmentation.onnx").exists() &&
        File(models, "ecapa_voxceleb.onnx").exists() && File(whisperDir, "decoder.onnx").exists()

    private fun silence(seconds: Double) = FloatArray((seconds * AudioUtils.SAMPLE_RATE).toInt())

    @Test
    fun twoVoicesInTwoLanguagesAreTranscribedAndAttributedToDifferentSpeakers() {
        assumeTrue("models not exported (run scripts/export_models.py and export_whisper.py)", ready)

        Diarizer.fromFile(File(models, "pyannote_segmentation.onnx")).use { diarizer ->
            VoiceEmbedder.fromFile(File(models, "ecapa_voxceleb.onnx")).use { voice ->
                WhisperTranscriber.fromFolder(whisperDir).use { whisper ->
                    // The engine has one language; for this test pick whichever Whisper is more confident in.
                    val recognizer = SpeechRecognizer { audio, _ ->
                        listOf("en", "pt").mapNotNull { whisper.transcribe(audio, it) }.maxByOrNull(TranscriptionResult::avgLogProb)
                    }
                    val registry = IdentityRegistry()
                    val engine = FusionEngine(
                        timeline = AudioTimeline(), turnDetector = diarizer, voice = voice,
                        verifier = SpeakerVerifier.fromMap(emptyMap()), recognizer = recognizer, asd = null, registry = registry
                    )

                    val audio = silence(1.0) + readWav("tts_en.wav") + silence(2.0) + readWav("tts_pt.wav") + silence(1.0)
                    engine.processChunk(audio, chunkStartSec = 0.0)

                    val entries = engine.transcript
                    println("END-TO-END: " + entries.joinToString(" | ") { "${it.label} ${it.decision}: ${it.text}" })

                    assertTrue(entries.size >= 2, "expected one entry per voice, got ${entries.size}")
                    assertTrue(entries.first().text.contains("recognition", ignoreCase = true), entries.first().text)
                    assertTrue(entries.last().text.contains("reconhecimento", ignoreCase = true), entries.last().text)
                    assertNotEquals(entries.first().speakerId, entries.last().speakerId, "two different voices must not share a speaker id")
                }
            }
        }
    }
}
