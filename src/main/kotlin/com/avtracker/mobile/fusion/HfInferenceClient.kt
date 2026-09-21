package com.avtracker.mobile.fusion

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.net.HttpURLConnection
import java.net.URL

/**
 * [LlmClient] over the Hugging Face text-generation HTTP API, which is what `InferenceClient.text_generation` calls:
 * POST `{"inputs": prompt, "parameters": {...}}` with a bearer token; the answer is `[{"generated_text": ...}]` (serverless
 * API) or `{"generated_text": ...}` (a text-generation-inference server). Point [endpoint] at any server that speaks it.
 *
 * Note that using it sends transcript text over the network: av-tracker itself does so only when its AI analysis is on.
 */
class HfInferenceClient(
    private val token: String?,
    private val model: String = LlmSpeakerAnalysis.DEFAULT_MODEL,
    private val endpoint: String? = null,
    private val timeoutMs: Int = 60_000
) : LlmClient {

    override fun generate(prompt: String, maxNewTokens: Int, temperature: Double): String {
        val url = URL(endpoint ?: "https://api-inference.huggingface.co/models/$model")
        val body = buildJsonObject {
            put("inputs", prompt)
            put("parameters", buildJsonObject {
                put("max_new_tokens", maxNewTokens)
                put("temperature", temperature)
                put("return_full_text", false)
            })
        }.toString()

        val connection = url.openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = timeoutMs
            connection.readTimeout = timeoutMs
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            if (!token.isNullOrEmpty()) connection.setRequestProperty("Authorization", "Bearer $token")
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }

            val code = connection.responseCode
            val text = (if (code in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) throw java.io.IOException("HTTP $code: ${text.take(200)}")
            return parse(text)
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        /** `[{"generated_text": "..."}]` or `{"generated_text": "..."}`. */
        fun parse(text: String): String {
            val root = Json.parseToJsonElement(text)
            val obj: JsonObject = if (root is JsonArray) root.jsonArray.first().jsonObject else root.jsonObject
            return (obj["generated_text"] as? JsonPrimitive)?.content ?: throw java.io.IOException("no generated_text in: ${text.take(200)}")
        }
    }
}
