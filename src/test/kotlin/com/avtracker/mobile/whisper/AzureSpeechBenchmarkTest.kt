package com.avtracker.mobile.whisper

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import kotlinx.serialization.json.*
import org.junit.Assume.assumeTrue
import kotlin.test.Test

/** Opt-in evaluation: scripts/benchmark_azure_speech.py creates the local corpus first. */
class AzureSpeechBenchmarkTest {
    @Test fun comparePackagedModelsOnAzureSpeech() {
        val folder = File(System.getenv("AVTRACKER_BENCHMARK_DIR") ?: "build/asr-benchmark")
        assumeTrue(File(folder, "manifest.json").exists())
        val cases = Json.parseToJsonElement(File(folder, "manifest.json").readText()).jsonArray
        val results = mutableListOf<JsonObject>()
        val sweep = System.getenv("AVTRACKER_DECODER_SWEEP") == "1"
        val variants = if (sweep) listOf(
            Triple(WhisperModel.SMALL, "small-production", WhisperModel.SMALL.decodingOptions().copy(usePrompt = true, beamSize = 3)),
            Triple(WhisperModel.SMALL, "small-no-prompt", WhisperModel.SMALL.decodingOptions().copy(usePrompt = false)),
            Triple(WhisperModel.SMALL, "small-beam1", WhisperModel.SMALL.decodingOptions().copy(beamSize = 1, usePrompt = true)),
            Triple(WhisperModel.SMALL, "small-no-prompt-beam5", WhisperModel.SMALL.decodingOptions().copy(usePrompt = false, beamSize = 5)),
            Triple(WhisperModel.BASE, "base-realtime", WhisperModel.BASE.decodingOptions().copy(
                beamSize = 1, bestOf = 1, temperatures = listOf(0.0), usePrompt = false))
        ) else WhisperModel.entries.map { Triple(it, it.id, it.decodingOptions()) }
        for ((model, variant, options) in variants) {
            val selectedVariant = System.getenv("AVTRACKER_DECODER_VARIANT")
            if (selectedVariant != null && variant != selectedVariant) continue
            val modelFolder = File("src/main/assets/${model.assetDirectory}")
            check(File(modelFolder, "meta.json").exists())
            WhisperTranscriber.fromFolder(modelFolder,
                vad = SileroVad.fromFile(File("src/main/assets/models/silero_vad_v6.onnx")),
                options = options).use { recognizer ->
                for (element in cases) {
                    val case = element.jsonObject
                    val id = case.getValue("id").jsonPrimitive.content
                    val bytes = File(folder, "$id.wav").readBytes()
                    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    var pos = 12
                    while (String(bytes, pos, 4, Charsets.US_ASCII) != "data") {
                        val size = buffer.getInt(pos + 4)
                        pos += 8 + size + (size and 1)
                    }
                    val audio = FloatArray(buffer.getInt(pos + 4) / 2) { buffer.getShort(pos + 8 + it * 2) / 32768f }
                    val started = System.nanoTime()
                    val output = recognizer.transcribe(audio, "pt")
                    val seconds = (System.nanoTime() - started) / 1e9
                    val reference = case.getValue("text").jsonPrimitive.content
                    val expected = words(reference)
                    val actual = words(output?.text.orEmpty())
                    val errors = distance(expected, actual)
                    results += buildJsonObject {
                        put("id", id); put("model", variant); put("reference", reference)
                        put("transcription", output?.text.orEmpty()); put("wordErrors", errors)
                        put("referenceWords", expected.size); put("wer", errors.toDouble() / expected.size)
                        put("audioSeconds", audio.size / 16000.0); put("processingSeconds", seconds)
                        put("noSpeechProbability", output?.noSpeechProb?.toDouble()?.let { JsonPrimitive(it) } ?: JsonNull)
                    }
                    File(folder, "results.json").writeText(Json { prettyPrint = true }.encodeToString(JsonArray.serializer(), JsonArray(results)))
                    println("BENCHMARK $variant/$id: $errors/${expected.size} word errors, ${"%.2f".format(seconds)}s: ${output?.text}")
                }
            }
        }
    }

    private fun words(text: String): List<String> = Normalizer.normalize(text.lowercase(), Normalizer.Form.NFD)
        .replace(Regex("\\p{M}+"), "").replace(Regex("[^a-z0-9 ]"), " ")
        .trim().split(Regex("\\s+")).filter { it.isNotEmpty() }

    private fun distance(a: List<String>, b: List<String>): Int {
        var row = IntArray(b.size + 1) { it }
        for (i in a.indices) {
            val next = IntArray(b.size + 1); next[0] = i + 1
            for (j in b.indices) next[j + 1] = minOf(next[j] + 1, row[j + 1] + 1, row[j] + if (a[i] == b[j]) 0 else 1)
            row = next
        }
        return row[b.size]
    }
}
