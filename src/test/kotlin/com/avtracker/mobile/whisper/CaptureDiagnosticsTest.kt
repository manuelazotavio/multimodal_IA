package com.avtracker.mobile.whisper

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*
import org.junit.Assume.assumeTrue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class CaptureDiagnosticsTest {
    @Test fun inspectExistingRecordingWithoutPlayback() {
        assumeTrue(System.getenv("AVTRACKER_CAPTURE_DIAGNOSTICS") == "1")
        val folder = File("build/asr-room-benchmark")
        val bytes = File(folder, "reuniao.wav").readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        // capture_acoustic_benchmark writes a canonical PCM16 WAV via soundfile.
        check(String(bytes, 36, 4, Charsets.US_ASCII) == "data")
        val audio = FloatArray(buffer.getInt(40) / 2) { buffer.getShort(44 + it * 2) / 32768f }
        val vad = SileroVad.fromFile(File("src/main/assets/models/silero_vad_v6.onnx"))
        val chunks = vad.speechTimestamps(audio, VadOptions.AV_TRACKER)
        val outputs = buildJsonObject {
            put("vadChunksSeconds", buildJsonArray {
                chunks.forEach { add(buildJsonObject { put("start", it.start / 16000.0); put("end", it.end / 16000.0) }) }
            })
            WhisperTranscriber.fromFolder(File("src/main/assets/whisper-small"), vad = vad,
                options = WhisperModel.SMALL.decodingOptions().copy(vad = null)).use { recognizer ->
                put("smallWithoutVad", recognizer.transcribe(audio, "pt")?.text.orEmpty())
                // Offline crop only, determined by cross-correlation with the known reference.
                put("smallWithoutVadAlignedCrop", recognizer.transcribe(audio.copyOfRange(45248, 162048), "pt")?.text.orEmpty())
            }
        }
        File(folder, "capture-diagnostics.json").writeText(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), outputs))
        println(outputs)
    }

    @Test fun speechGatePreservesCapturedWordsAndRejectsSilence() {
        val folder = File("build/asr-room-benchmark")
        val audioFile = File(folder, "reuniao.wav")
        assumeTrue(audioFile.exists())
        val bytes = audioFile.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        check(String(bytes, 36, 4, Charsets.US_ASCII) == "data")
        val audio = FloatArray(buffer.getInt(40) / 2) { buffer.getShort(44 + it * 2) / 32768f }
        WhisperTranscriber.fromFolder(File("src/main/assets/whisper-small"),
            vad = SileroVad.fromFile(File("src/main/assets/models/silero_vad_v6.onnx")),
            options = WhisperModel.SMALL.decodingOptions()).use { recognizer ->
            val result = assertNotNull(recognizer.transcribe(audio, "pt"))
            assertEquals("bom dia a todos nesta reunião vamos revisar os resultados do aplicativo e depender dos próximos passos",
                result.text.lowercase().replace(Regex("[^\\p{L} ]"), " ").trim().replace(Regex("\\s+"), " "))
            assertNull(recognizer.transcribe(FloatArray(16000 * 3), "pt"))
        }
    }
}
