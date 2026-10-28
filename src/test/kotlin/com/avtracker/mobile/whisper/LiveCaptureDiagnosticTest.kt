package com.avtracker.mobile.whisper

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*
import org.junit.Assume.assumeTrue
import kotlin.test.Test

class LiveCaptureDiagnosticTest {
    @Test fun inspectSavedCaptureWithoutPlayback() {
        assumeTrue(System.getenv("AVTRACKER_LIVE_DIAGNOSTIC") == "1")
        val directory = File("build/live-asr-diagnostic")
        val bytes = File(directory, "live.wav").readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val audio = FloatArray((bytes.size - 44) / 2) { buffer.getShort(44 + it * 2) / 32768f }
        val rows = mutableListOf<JsonObject>()
        WhisperTranscriber.fromFolder(File("src/main/assets/whisper"),
            options = WhisperModel.BASE.decodingOptions().copy(rejectUnreliable = false), intraOpThreads = 2).use { recognizer ->
            for (seconds in listOf(2, 4, 8)) {
                for (start in 0..(audio.size / 16000 - seconds) step seconds) {
                    val clip = audio.copyOfRange(start * 16000, (start + seconds) * 16000)
                    val result = recognizer.transcribe(clip, "pt")
                    rows += buildJsonObject {
                        put("startSeconds", start); put("durationSeconds", seconds)
                        put("text", result?.text.orEmpty()); put("tokens", result?.tokens?.size ?: 0)
                        put("avgLogProb", result?.avgLogProb?.let { JsonPrimitive(it) } ?: JsonNull)
                        put("noSpeechProb", result?.noSpeechProb?.let { JsonPrimitive(it) } ?: JsonNull)
                        put("compressionRatio", WhisperTranscriber.compressionRatio(result?.text.orEmpty()))
                    }
                    File(directory, "results.json").writeText(Json.encodeToString(JsonArray.serializer(), JsonArray(rows)))
                }
            }
        }
    }
}
