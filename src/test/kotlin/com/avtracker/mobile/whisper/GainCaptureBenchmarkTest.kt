package com.avtracker.mobile.whisper

import com.avtracker.mobile.audio.AudioUtils
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*
import org.junit.Assume.assumeTrue
import kotlin.math.tanh
import kotlin.test.Test

/** Opt-in comparison on the existing microphone recording. Never plays audio. */
class GainCaptureBenchmarkTest {
    @Test fun compareLevelsOnCapturedReference() {
        assumeTrue(System.getenv("AVTRACKER_GAIN_BENCHMARK") == "1")
        val directory = File("build/asr-voice-recognition-retest")
        val bytes = File(directory, "reuniao.wav").readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val raw = FloatArray((bytes.size - 44) / 2) { buffer.getShort(44 + it * 2) / 32768f }
        val variants = listOf(
            "raw" to raw,
            "existing-normalization" to AudioUtils.normalize(raw),
            "gain2" to FloatArray(raw.size) { (raw[it] * 2).coerceIn(-0.98f, 0.98f) },
            "gain4" to FloatArray(raw.size) { (raw[it] * 4).coerceIn(-0.98f, 0.98f) },
            "soft-gain4" to FloatArray(raw.size) { tanh(raw[it] * 4) }
        )
        val rows = mutableListOf<JsonObject>()
        for (model in listOf(WhisperModel.BASE, WhisperModel.SMALL)) {
            WhisperTranscriber.fromFolder(File("src/main/assets/${model.assetDirectory}"),
                vad = SileroVad.fromFile(File("src/main/assets/models/silero_vad_v6.onnx")),
                options = model.decodingOptions().copy(rejectUnreliable = false), intraOpThreads = 2).use { recognizer ->
                for ((variant, audio) in variants) {
                    val start = System.nanoTime()
                    val result = recognizer.transcribe(audio, "pt")
                    rows += buildJsonObject {
                        put("model", model.id); put("variant", variant)
                        put("text", result?.text.orEmpty()); put("seconds", (System.nanoTime() - start) / 1e9)
                        put("avgLogProb", result?.avgLogProb?.let { JsonPrimitive(it) } ?: JsonNull)
                        put("noSpeechProb", result?.noSpeechProb?.let { JsonPrimitive(it) } ?: JsonNull)
                        put("compressionRatio", WhisperTranscriber.compressionRatio(result?.text.orEmpty()))
                    }
                    File(directory, "gain-results.json").writeText(Json.encodeToString(JsonArray.serializer(), JsonArray(rows)))
                }
            }
        }
    }
}
