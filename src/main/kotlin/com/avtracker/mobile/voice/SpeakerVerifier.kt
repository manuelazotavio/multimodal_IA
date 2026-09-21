package com.avtracker.mobile.voice

import com.avtracker.mobile.audio.AudioUtils
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class VerifierResult(
    val bestName: String,
    val bestScore: Float,
    /** All speakers scoring >= threshold, best first. */
    val candidates: List<Pair<String, Float>>
)

/**
 * Port of MultiSpeakerVerifier: cosine similarity of a voice embedding against known speaker centroids (default
 * threshold 0.80, as in run_multimodal_tracker.py), backed by a [VoiceStore] the way the Python one is backed by
 * data/embeddings.
 */
class SpeakerVerifier(
    private val store: VoiceStore,
    val threshold: Float = 0.80f,
    /**
     * Called after a voice file is saved for a face-voice binding or a live embedding, with the `.npy` file name the
     * Python would use (its `db.add_speaker` + `db.save_voice_embedding`). Auto-enrolment is not mirrored, as in Python.
     */
    private val onVoiceSaved: ((name: String, embedding: FloatArray, sourceFile: String) -> Unit)? = null,
    private val nowSec: () -> Double = { System.currentTimeMillis() / 1000.0 }
) {
    /** name -> centroid used for scoring (Python `embeddings`). */
    private val embeddings = LinkedHashMap<String, FloatArray>()

    /** name -> the individual embeddings behind the centroid, for incremental updates (Python `_raw_embeddings`). */
    private val rawEmbeddings = HashMap<String, MutableList<FloatArray>>()
    private val lastAutoEnrollSec = HashMap<String, Double>()

    init {
        loadEmbeddings()
    }

    val names: Set<String> @Synchronized get() = embeddings.keys.toSet()

    operator fun contains(name: String): Boolean = synchronized(this) { name in embeddings }

    val size: Int @Synchronized get() = embeddings.size

    /**
     * Port of MultiSpeakerVerifier.load_embeddings: files with the same clean name are averaged when they sound
     * like the same person, and split into homonyms ("Name", "Name 2") when they do not. Generic names (spk_N,
     * Person_N, Unknown) exist on disk only for the rename cycle and never compete with real voices. Like the
     * Python version this does not clear entries that are already loaded.
     */
    @Synchronized
    fun loadEmbeddings() {
        val accumulated = LinkedHashMap<String, MutableList<FloatArray>>()
        for (voice in store.loadAll()) {
            val name = cleanName(voice.name)
            if (GENERIC.matches(name)) continue
            accumulated.getOrPut(name) { mutableListOf() } += voice.embedding.toFloatArray()
        }

        for ((name, tensors) in accumulated) {
            if (tensors.size == 1) {
                embeddings[name] = tensors[0]
                rawEmbeddings[name] = mutableListOf(tensors[0])
                continue
            }
            // Greedy clustering against group centroids, so a short noisy clip does not fragment a person.
            val groups = ArrayList<MutableList<FloatArray>>()
            val centroids = ArrayList<FloatArray>()
            for (t in tensors) {
                var bestIdx = -1
                var bestSim = -1f
                centroids.forEachIndexed { gi, centroid ->
                    val sim = AudioUtils.cosine(t, centroid)
                    if (sim > bestSim) {
                        bestSim = sim
                        bestIdx = gi
                    }
                }
                if (bestSim >= SAME_PERSON_SIM) {
                    groups[bestIdx] += t
                    centroids[bestIdx] = AudioUtils.l2Normalize(mean(groups[bestIdx]))
                } else {
                    groups += mutableListOf(t)
                    centroids += AudioUtils.l2Normalize(t)
                }
            }
            groups.forEachIndexed { idx, group ->
                val key = if (idx == 0) name else "$name ${idx + 1}"
                embeddings[key] = AudioUtils.l2Normalize(mean(group))
                rawEmbeddings[key] = group.toMutableList()
            }
        }
    }

    /** Registers a brand-new voice under [name] in memory only (the Python `verifier.embeddings[name] = emb`). */
    @Synchronized
    fun enroll(name: String, embedding: FloatArray) {
        val emb = AudioUtils.l2Normalize(embedding)
        embeddings[name] = emb
        rawEmbeddings.getOrPut(name) { mutableListOf() } += emb
    }

    /** Face-voice binding: persists a `_fvbind` file and registers the voice at once. */
    @Synchronized
    fun enrollFromFace(name: String, embedding: FloatArray) {
        val emb = AudioUtils.l2Normalize(embedding)
        val base = "${name}_${timestamp()}_fvbind"
        store.save(base, emb)
        enroll(name, emb)
        onVoiceSaved?.invoke(name, emb, "$base.npy")
    }

    /** Port of MultiSpeakerVerifier._process_audio_chunk (after the embedding has been computed). */
    @Synchronized
    fun identify(embedding: FloatArray): VerifierResult {
        val scored = embeddings.map { (name, centroid) -> name to AudioUtils.cosine(embedding, centroid) }
            .sortedByDescending { it.second }
        val best = scored.firstOrNull()
        return VerifierResult(
            bestName = best?.first ?: "Unknown",
            bestScore = best?.second ?: -1f,
            candidates = scored.filter { it.second >= threshold }
        )
    }

    /**
     * Port of MultiSpeakerVerifier.auto_enroll for an already-computed [embedding]: saves it (as
     * `<name>_<timestamp>_auto`) and extends the centroid if it is plausibly the same person but not a duplicate of
     * what we have.
     */
    @Synchronized
    fun autoEnroll(
        name: String,
        embedding: FloatArray,
        maxTotal: Int = 8,
        cooldownSec: Double = 300.0,
        minSim: Float = 0.65f,
        maxSim: Float = 0.96f
    ): Boolean {
        val now = nowSec()
        if (now - (lastAutoEnrollSec[name] ?: 0.0) < cooldownSec) return false

        // Total file limit on disk for this person (manual enrolments included).
        if (store.baseNames().count { cleanName(it) == name } >= maxTotal) return false

        val emb = AudioUtils.l2Normalize(embedding)
        embeddings[name]?.let { centroid ->
            val sim = AudioUtils.cosine(emb, centroid)
            if (sim < minSim) return false // not this person
            if (sim > maxSim) return false // identical to what we already have: no diversity
        }

        store.save("${name}_${timestamp()}_auto", emb)
        val raws = rawEmbeddings.getOrPut(name) { mutableListOf() }
        raws += emb
        embeddings[name] = AudioUtils.l2Normalize(mean(raws))
        lastAutoEnrollSec[name] = now
        return true
    }

    /** Port of MultiSpeakerVerifier.rename_embedding: moves a voice to its real name in memory and on disk. */
    @Synchronized
    fun renameEmbedding(oldName: String, newName: String) {
        embeddings.remove(oldName)?.let { embeddings[newName] = it }
        for (file in store.baseNames()) {
            if (file == oldName || file.startsWith(oldName + "_")) {
                store.rename(file, newName + file.substring(oldName.length))
            }
        }
    }

    /** True if a stored file exists for exactly [name] (Python: the emb_dir scan in _save_live_embedding). */
    @Synchronized
    fun hasStoredFileFor(name: String): Boolean {
        val norm = name.lowercase().trim()
        return store.baseNames().any { val f = it.lowercase(); f == norm || f.startsWith(norm + "_") }
    }

    @Synchronized
    fun save(name: String, embedding: FloatArray) {
        val base = "${name}_${timestamp()}"
        store.save(base, embedding)
        onVoiceSaved?.invoke(name, embedding, "$base.npy")
    }

    private fun mean(vectors: List<FloatArray>): FloatArray {
        val out = FloatArray(vectors[0].size)
        for (v in vectors) for (i in out.indices) out[i] += v[i]
        for (i in out.indices) out[i] /= vectors.size
        return out
    }

    private fun timestamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date((nowSec() * 1000).toLong()))

    companion object {
        /** Below this similarity two files with the same name are treated as different people (homonyms). */
        const val SAME_PERSON_SIM = 0.20f
        private val GENERIC = Regex("^(spk_\\d+|Person_\\d+|Desconhecido_\\d+|Unknown(_\\d+)?)$")

        /** Port of MultiSpeakerVerifier._clean_name: drops a trailing _auto/_fvbind and a _YYYYMMDD_HHMMSS stamp. */
        fun cleanName(base: String): String {
            var name = base
            for (suffix in listOf("_auto", "_fvbind")) {
                if (name.endsWith(suffix)) {
                    name = name.dropLast(suffix.length)
                    break
                }
            }
            val parts = name.split('_')
            if (parts.size >= 3 && parts.last().all(Char::isDigit) && parts[parts.size - 2].all(Char::isDigit) &&
                parts.last().isNotEmpty() && parts[parts.size - 2].isNotEmpty()
            ) {
                return parts.dropLast(2).joinToString("_")
            }
            return name
        }

        fun fromMap(persons: Map<String, FloatArray>, threshold: Float = 0.80f, nowSec: () -> Double = { System.currentTimeMillis() / 1000.0 }) =
            SpeakerVerifier(InMemoryVoiceStore(persons.map { (n, e) -> StoredVoice(n, e.toList()) }), threshold, nowSec = nowSec)
    }
}
