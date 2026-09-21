package com.avtracker.mobile.fusion

import android.util.Log
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import java.util.concurrent.ConcurrentLinkedQueue

/** A text-generation backend (Python `InferenceClient.text_generation`). Throws on any failure. */
fun interface LlmClient {
    fun generate(prompt: String, maxNewTokens: Int, temperature: Double): String
}

/**
 * Port of RealtimeTranscriber._llm_identify_speakers / _maybe_run_llm_analysis: every [intervalSegments] segments, if some
 * speaker in the last 20 lines still has a generic name, the last 50 transcript lines and what is already known go to an
 * LLM, which answers with a JSON object mapping generic ids (`spk_016`, `Person_1`) to real names; the mapping is applied
 * to speakers that are still unnamed and whose new name nobody else holds, and their voice is saved under it.
 *
 * av-tracker runs with `use_ai_analysis=False`, so this is off unless an [LlmClient] is given. Differences from the
 * Python: names mentioned in the conversation are listed sorted (a Python `set` has no order), and with [background] the
 * request runs on its own thread while the answer is applied on the next segment, on the fusion thread, so the shared
 * speaker tables are never touched from two threads.
 */
class LlmSpeakerAnalysis(
    private val client: LlmClient?,
    private val registry: IdentityRegistry,
    private val book: SpeakerBook,
    private val naming: SpeakerNaming,
    private val numSpeakers: Int?,
    private val transcript: () -> List<TranscriptEntry>,
    private val segmentCount: () -> Int,
    private val background: Boolean = false,
    private val intervalSegments: Int = 10,
    private val model: String = DEFAULT_MODEL
) {
    private var lastAnalysis = 0
    private val answers = ConcurrentLinkedQueue<String>()

    val enabled: Boolean get() = client != null

    /** `_maybe_run_llm_analysis`, called once per processed segment on the fusion thread. */
    fun maybeRun() {
        if (client == null) return
        applyPending()

        val count = segmentCount()
        if (count - lastAnalysis < intervalSegments) return
        lastAnalysis = count

        val last20 = transcript().takeLast(20)
        val hasGeneric = last20.map { it.speakerId }.toSet().any { IdentityRegistry.isGeneric(registry.nameOf(it) ?: "") }
        if (hasGeneric) identifySpeakers()
    }

    /** Applies the answers that arrived while a request was in flight. */
    fun applyPending() {
        while (true) apply(answers.poll() ?: return)
    }

    /** `_llm_identify_speakers`: build the prompt, ask, and (inline, or later when [background]) apply the answer. */
    fun identifySpeakers() {
        val client = client ?: return
        val prompt = buildPrompt() ?: return

        val ask = Runnable {
            try {
                val response = client.generate(prompt, MAX_NEW_TOKENS, TEMPERATURE)
                if (background) answers.add(response) else apply(response)
            } catch (t: Throwable) {
                Log.w(TAG, "LLM speaker analysis failed", t)
            }
        }
        if (background) Thread(ask, "llm-analysis").apply { isDaemon = true }.start() else ask.run()
    }

    /** The prompt the Python builds, or null when there is nothing to identify. */
    fun buildPrompt(): String? {
        val lines = ArrayList<String>()
        val genericSpeakers = HashSet<String>()
        for (entry in transcript().takeLast(50)) {
            val name = entry.speakerName?.takeIf { it.isNotEmpty() } ?: entry.speakerId
            lines += "[$name]: ${entry.text}"
            if (IdentityRegistry.isGeneric(name)) genericSpeakers += name
        }
        if (genericSpeakers.isEmpty() || lines.isEmpty()) return null

        val alreadyIdentified = LinkedHashMap<String, String>()
        for ((pid, name) in registry.namesSnapshot()) if (!IdentityRegistry.isGeneric(name)) alreadyIdentified[pid] = name
        val contextNames = book.contextNames - alreadyIdentified.values.toSet()

        val transcriptText = lines.takeLast(40).joinToString("\n")
        val participants = numSpeakers?.takeIf { it != 0 }?.toString() ?: "desconhecido"

        return "Analise esta transcrição de uma reunião com $participants participantes.\n" +
            "Alguns falantes já foram identificados: ${PyText.dictRepr(alreadyIdentified.entries.take(8).associate { it.key to it.value })}\n" +
            "Nomes mencionados na conversa: ${if (contextNames.isNotEmpty()) contextNames.sorted().joinToString(", ") else "nenhum"}\n" +
            "Falantes genéricos (não identificados): ${genericSpeakers.sorted().joinToString(", ")}\n\n" +
            "Transcrição:\n$transcriptText\n\n" +
            "Baseado no contexto da conversa (quem fala com quem, referências a outros, " +
            "tópicos discutidos, gênero gramatical), identifique o nome real de cada " +
            "falante genérico.\n\n" +
            "Responda APENAS com um JSON mapeando IDs genéricos para nomes reais. " +
            "Exemplo: {\"spk_016\": \"Arthur\", \"Person_1\": \"Kauan\"}\n" +
            "Só inclua mapeamentos que você tem CERTEZA. Se não tem certeza, omita."
    }

    /** The second half of `_llm_identify_speakers`: parse the first `{...}` of [response] and name the speakers. */
    fun apply(response: String) {
        val match = JSON_OBJECT.find(response)
        if (match == null) {
            Log.d(TAG, "LLM response (no JSON found): ${response.take(200)}")
            return
        }
        val mapping: JsonObject = try {
            json.parseToJsonElement(match.value).jsonObject
        } catch (t: Throwable) {
            Log.w(TAG, "LLM speaker analysis failed", t)
            return
        }

        for ((genericId, value) in mapping) {
            val realNameRaw = (value as? JsonPrimitive)?.takeIf { it.isString }?.content ?: continue
            if (realNameRaw.length < 2) continue
            val realName = PyText.title(realNameRaw.trim())

            // The person id whose current name, or whose id, is this generic label.
            val targetPid = registry.namesSnapshot().entries.firstOrNull { (pid, name) -> name == genericId || pid == genericId }?.key ?: continue
            val current = registry.nameOf(targetPid)
            if (!current.isNullOrEmpty() && !IdentityRegistry.isGeneric(current)) continue // never overwrite a real name
            val holder = registry.personFor(realName)
            if (holder != null && holder != targetPid) continue // the name is taken

            val speakersFull = numSpeakers != null && numSpeakers != 0 && book.identifiedSpeakers.size >= numSpeakers
            if (!speakersFull) {
                registry.setName(targetPid, realName)
                registry.bind(realName, targetPid)
                book.identifiedSpeakers += targetPid
                naming.saveLiveEmbedding(targetPid, realName, current)
            }
        }
    }

    companion object {
        private const val TAG = "LlmSpeakerAnalysis"
        const val DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
        const val MAX_NEW_TOKENS = 150
        const val TEMPERATURE = 0.1
        private val json = Json { ignoreUnknownKeys = true }
        private val JSON_OBJECT = Regex("\\{[^}]+\\}")
    }
}
