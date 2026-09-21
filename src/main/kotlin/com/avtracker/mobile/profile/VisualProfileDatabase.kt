package com.avtracker.mobile.profile

import android.content.Context
import android.util.Log
import kotlinx.serialization.json.Json
import kotlin.math.sqrt

/**
 * Port of the Python VisualProfileDatabase: loads known-person embeddings and
 * identifies a query embedding by best cosine similarity above a threshold.
 */
class VisualProfileDatabase private constructor(
    private val knownPersons: Map<String, FloatArray>
) {
    val size: Int get() = knownPersons.size

    /** Names with an enrolled face (Python `face_tracker.known_embeddings`). */
    val names: Set<String> get() = knownPersons.keys

    fun identifyPerson(embedding: FloatArray, threshold: Float): Pair<String?, Float> {
        if (knownPersons.isEmpty()) return null to 0f

        var bestMatch: String? = null
        var bestSimilarity = 0f

        for ((name, known) in knownPersons) {
            val similarity = cosineSimilarity(embedding, known)
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity
                bestMatch = name
            }
        }

        return if (bestSimilarity >= threshold) bestMatch to bestSimilarity else null to bestSimilarity
    }

    companion object {
        private const val TAG = "VisualProfileDatabase"
        private val json = Json { ignoreUnknownKeys = true }

        /** For tests and programmatic construction. */
        fun fromMap(persons: Map<String, FloatArray>): VisualProfileDatabase = VisualProfileDatabase(persons)

        /** Loads every *.json profile bundled under assets/[dir]. */
        fun loadFromAssets(context: Context, dir: String = "profiles"): VisualProfileDatabase {
            val known = mutableMapOf<String, FloatArray>()
            val files = context.assets.list(dir).orEmpty()

            for (file in files.filter { it.endsWith(".json") }) {
                runCatching {
                    val text = context.assets.open("$dir/$file").bufferedReader().use { it.readText() }
                    val profile = json.decodeFromString(VisualProfile.serializer(), text)
                    known[profile.name] = profile.averageEmbedding.toFloatArray()
                }.onFailure { Log.e(TAG, "Failed to load profile $file", it) }
            }

            Log.i(TAG, "Loaded ${known.size} visual profile(s) from assets/$dir")
            return VisualProfileDatabase(known)
        }

        private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            if (a.size != b.size) return 0f
            var dot = 0f
            var normA = 0f
            var normB = 0f
            for (i in a.indices) {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            if (normA == 0f || normB == 0f) return 0f
            return dot / (sqrt(normA) * sqrt(normB))
        }
    }
}
