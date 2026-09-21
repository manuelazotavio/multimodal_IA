package com.avtracker.mobile.voice

import com.avtracker.mobile.audio.AudioUtils

/**
 * Port of RealtimeTranscriber._get_or_create_session_speaker_id: gives every not-yet-identified voice a stable
 * session id by matching its ECAPA embedding against the voices already heard in this session, never merging two
 * voices whose detected gender differs.
 */
class SessionSpeakerTracker(
    private val newId: () -> String,
    private val numSpeakers: Int? = null,
    /** Ids already known elsewhere (the Python `speaker_names` keys), counted against the [numSpeakers] limit. */
    private val otherTrackedIds: () -> Set<String> = { emptySet() }
) {
    private val embeddings = LinkedHashMap<String, FloatArray>()   // _session_unknown_embs
    private val genders = HashMap<String, String>()                // _session_gender

    val ids: Set<String> get() = embeddings.keys

    fun embeddingOf(id: String): FloatArray? = embeddings[id]

    fun genderOf(id: String): String? = genders[id]

    /** Lets a face person id inherit a session voice embedding and gender (`_session_unknown_embs[speaker_id] = ...`). */
    @Synchronized
    fun mirror(fromId: String, toId: String) {
        val emb = embeddings[fromId] ?: return
        if (toId in embeddings) return
        embeddings[toId] = emb.copyOf()
        genders[fromId]?.let { genders[toId] = it }
    }

    @Synchronized
    fun getOrCreate(embedding: FloatArray, excludeIds: Set<String> = emptySet(), audioGender: String? = null): String {
        var bestId: String? = null
        var bestSim = -1f
        for ((id, known) in embeddings) {
            if (id in excludeIds) continue
            // Skip if the gender is known for both and they differ.
            val knownGender = genders[id]
            if (audioGender != null && knownGender != null && audioGender != knownGender) continue
            val sim = AudioUtils.dot(embedding, known)
            if (sim > bestSim) {
                bestSim = sim
                bestId = id
            }
        }

        // Cross-gender merges are blocked above, so within a gender a low threshold is safe (same-person embeddings
        // land at 0.40-0.60 across chunks); with an unknown gender on either side use a moderate one.
        val sameGender = audioGender != null && bestId != null && genders[bestId] == audioGender
        val threshold = if (sameGender) SAME_GENDER_THRESHOLD else UNKNOWN_GENDER_THRESHOLD
        if (bestId != null && bestSim >= threshold) {
            val known = embeddings.getValue(bestId)
            for (i in known.indices) known[i] = (1 - EMA_ALPHA) * known[i] + EMA_ALPHA * embedding[i]
            if (audioGender != null && bestId !in genders) genders[bestId] = audioGender
            return bestId
        }

        // Respect the speaker limit: force a merge only if minimally plausible (>= 0.20).
        if (numSpeakers != null && numSpeakers > 0 && bestId != null && bestSim >= FORCED_MERGE_THRESHOLD) {
            val tracked = otherTrackedIds() + embeddings.keys
            if (tracked.size >= numSpeakers) return bestId
        }

        val id = newId()
        embeddings[id] = embedding.copyOf()
        if (audioGender != null) genders[id] = audioGender
        return id
    }

    companion object {
        const val SAME_GENDER_THRESHOLD = 0.35f
        const val UNKNOWN_GENDER_THRESHOLD = 0.45f
        const val FORCED_MERGE_THRESHOLD = 0.20f
        const val EMA_ALPHA = 0.1f
    }
}
