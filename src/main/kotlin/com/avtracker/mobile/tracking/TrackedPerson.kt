package com.avtracker.mobile.tracking

import java.util.Locale

/**
 * Port of the Python TrackedPerson class. Frame-crop buffering (used in the
 * Python version only for potential future replay/export) is intentionally
 * dropped here to avoid holding onto bitmaps on a memory-constrained device.
 */
class TrackedPerson(
    val trackId: Int,
    bbox: Roi,
    embedding: FloatArray? = null,
    private val embeddingHistorySize: Int = 30
) {
    var bbox: Roi = bbox
        private set

    private val embeddingsHistory = ArrayDeque<FloatArray>(embeddingHistorySize)

    var embedding: FloatArray? = embedding
        private set

    var lastSeen: Long = System.currentTimeMillis()
        private set

    var framesAlive: Int = 1
        private set

    var totalFramesTracked: Int = 1
        private set

    var matchedThisFrame: Boolean = false

    var identifiedName: String? = null
        private set

    var identificationConfidence: Float = 0f
        private set

    init {
        embedding?.let { embeddingsHistory.addLast(it) }
    }

    fun update(bbox: Roi, embedding: FloatArray? = null) {
        this.bbox = bbox
        lastSeen = System.currentTimeMillis()
        framesAlive++
        totalFramesTracked++
        matchedThisFrame = true

        if (embedding != null) {
            if (embeddingsHistory.size >= embeddingHistorySize) embeddingsHistory.removeFirst()
            embeddingsHistory.addLast(embedding)
            this.embedding = averageEmbedding()
        }
    }

    fun setIdentity(name: String?, confidence: Float) {
        identifiedName = name
        identificationConfidence = confidence
    }

    fun displayName(): String =
        identifiedName?.let { "$it (%.2f)".format(Locale.ROOT, identificationConfidence) } ?: "Person $trackId"

    fun averageEmbedding(): FloatArray? {
        if (embeddingsHistory.isEmpty()) return null
        val dim = embeddingsHistory.first().size
        val sum = FloatArray(dim)
        for (e in embeddingsHistory) for (i in 0 until dim) sum[i] += e[i]
        for (i in 0 until dim) sum[i] /= embeddingsHistory.size
        return sum
    }
}
