package com.avtracker.mobile.fusion

/**
 * Whisper hallucination filters from RealtimeTranscriber._process_segment: subtitle/YouTube artefacts,
 * repeated fragments and low-confidence one-liners on silence are dropped before speaker attribution.
 */
object TranscriptFilter {

    // Portuguese phrases stay as-is: they match what Whisper actually emits on PT-BR audio.
    private val BAD_PHRASES = listOf(
        "Amara.org", "Legendas", "Obrigado", "tchau gente", "tchau tchau",
        "transmissão", "inscreva-se", "obrigada por assistir",
        "continue assistindo", "não se esqueça",
        "estou ouvindo", "i'm listening", "subtitles by",
        "se inscreva", "inscreva no canal", "sininho", "ative o",
        "ative as notificações", "deixe seu like", "deixa o like",
        "curta o vídeo", "curtam o vídeo", "compartilhe", "comentários",
        "próximo vídeo", "valeu galera", "até a próxima",
        "nos vemos", "obrigado por assistir"
    )

    // Caption artefacts that are hallucinations only when they are the ENTIRE transcription.
    private val EXACT_HALLUCINATIONS = setOf(
        "atenção", "música", "legenda", "legendas", "aplausos", "risos", "obrigado", "obrigada", "fim", "the end"
    )

    /** The initial_prompt RealtimeTranscriber gives Whisper ("a conversation" hint), by language. */
    fun promptFor(language: String) =
        if (language == "pt") "Transcrição de conversa em português brasileiro." else "Transcript of a conversation."

    private val FRAGMENT_SPLIT = Regex("[.!?,;]+")
    private const val TRIM_CHARS = " .!?,;:\"'-[]()"

    /** True if [text] looks like genuine speech and should go on to speaker attribution. */
    fun accept(text: String, avgLogProb: Float, noSpeechProb: Float, language: String? = null): Boolean {
        // Whisper is not confident there is real speech in the segment.
        if (noSpeechProb > 0.5f) return false

        val lower = text.lowercase()
        if (text.isEmpty() || text.length < 5 || BAD_PHRASES.any { lower.contains(it.lowercase()) }) return false

        // Prompt echo: Whisper regurgitates its initial_prompt on low-speech audio.
        if (language != null && TextSimilarity.ratio(lower, promptFor(language).lowercase()) > 0.6) return false

        // "X. X. X." or "X X X": every fragment identical -> hallucination by repetition.
        val parts = FRAGMENT_SPLIT.split(text).map { it.trim() }.filter { it.isNotEmpty() }
        if (parts.size >= 3 && parts.map { it.lowercase() }.toSet().size == 1) return false

        val normalized = lower.trim { it in TRIM_CHARS }
        if (normalized in EXACT_HALLUCINATIONS) return false

        // Short phrases with low log-prob or moderate no-speech probability are likely hallucinations.
        if (text.length < 20 && (avgLogProb < -0.8f || noSpeechProb > 0.3f)) return false

        // A single short word with any elevated no-speech probability: common on silence/noise.
        val words = normalized.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.size <= 1 && normalized.length < 14 && noSpeechProb > 0.15f) return false

        return true
    }
}
