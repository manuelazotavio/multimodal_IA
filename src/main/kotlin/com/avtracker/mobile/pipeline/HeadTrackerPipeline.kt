package com.avtracker.mobile.pipeline

import android.graphics.Bitmap
import android.util.Log
import com.avtracker.mobile.detection.YoloDetector
import com.avtracker.mobile.embedding.EdgeFaceEmbedder
import com.avtracker.mobile.profile.VisualProfileDatabase
import com.avtracker.mobile.tracking.Roi
import com.avtracker.mobile.tracking.SimpleTracker
import com.avtracker.mobile.tracking.TrackedPerson

/**
 * Port of HeadTrackerWithNameDisplay.process_frame() from personid_tracker.py:
 * detect heads -> motion track -> crop faces -> embed -> match against known
 * tracked persons by appearance -> identify against the profile database.
 */
class HeadTrackerPipeline(
    private val detector: YoloDetector,
    private val embedder: EdgeFaceEmbedder,
    private val database: VisualProfileDatabase,
    private val identificationThreshold: Float = 0.6f,
    private val appearanceThreshold: Float = 0.5f,
    private val staleTimeoutMs: Long = 3000L
) {
    private val motionTracker = SimpleTracker()
    private val trackedPersons = mutableMapOf<Int, TrackedPerson>()
    private val availableIds = sortedSetOf<Int>()
    private var nextPersonIdCounter = 1

    fun processFrame(frame: Bitmap): FrameResult {
        val headRois = detector.detectHeads(frame).map {
            Roi.clamped(it.x1, it.y1, it.x2, it.y2, it.confidence)
        }

        val motionTracks = motionTracker.update(headRois)
        val (faceCrops, validTracks) = extractFaceCrops(frame, motionTracks)

        val embeddings: List<FloatArray?> = if (faceCrops.isEmpty()) {
            emptyList()
        } else {
            runCatching { embedder.extractEmbeddings(faceCrops) }
                .onFailure { Log.e(TAG, "Embedding extraction failed", it) }
                .getOrElse { List(faceCrops.size) { null } }
        }

        val persons = matchWithAppearance(validTracks, embeddings)

        return FrameResult(
            persons = persons.map {
                TrackedPersonSnapshot(
                    trackId = it.trackId,
                    bbox = it.bbox,
                    displayName = it.displayName(),
                    identified = it.identifiedName != null,
                    name = it.identifiedName ?: "Person_${it.trackId}",
                    confidence = it.identificationConfidence
                )
            },
            knownDatabaseSize = database.size
        )
    }

    private fun extractFaceCrops(frame: Bitmap, rois: List<Roi>): Pair<List<Bitmap>, List<Roi>> {
        val crops = mutableListOf<Bitmap>()
        val valid = mutableListOf<Roi>()

        for (roi in rois) {
            val x1 = roi.x1.coerceIn(0, frame.width)
            val y1 = roi.y1.coerceIn(0, frame.height)
            val x2 = roi.x2.coerceIn(0, frame.width)
            val y2 = roi.y2.coerceIn(0, frame.height)

            if (x2 > x1 && y2 > y1 && (x2 - x1) > 10 && (y2 - y1) > 10) {
                crops += Bitmap.createBitmap(frame, x1, y1, x2 - x1, y2 - y1)
                valid += roi
            }
        }
        return crops to valid
    }

    private fun matchWithAppearance(tracks: List<Roi>, embeddings: List<FloatArray?>): List<TrackedPerson> {
        for (person in trackedPersons.values) person.matchedThisFrame = false

        val matched = mutableListOf<TrackedPerson>()

        for ((track, embedding) in tracks.zip(embeddings)) {
            var bestId: Int? = null
            var bestSimilarity = appearanceThreshold

            if (embedding != null) {
                for ((id, person) in trackedPersons) {
                    if (person.matchedThisFrame) continue
                    val personEmbedding = person.averageEmbedding() ?: continue
                    val similarity = dot(embedding, personEmbedding)
                    if (similarity > bestSimilarity) {
                        bestSimilarity = similarity
                        bestId = id
                    }
                }
            }

            val person = if (bestId != null) {
                trackedPersons.getValue(bestId).also { it.update(track, embedding) }
            } else {
                val id = allocatePersonId()
                TrackedPerson(id, track, embedding).also { trackedPersons[id] = it }
            }
            matched += person
        }

        for (person in matched) {
            val embedding = person.embedding ?: continue
            val (name, confidence) = database.identifyPerson(embedding, identificationThreshold)
            person.setIdentity(name, confidence)
        }

        val now = System.currentTimeMillis()
        val staleIds = trackedPersons.filterValues { now - it.lastSeen > staleTimeoutMs }.keys.toList()
        for (id in staleIds) {
            trackedPersons.remove(id)
            availableIds += id
        }

        return matched
    }

    private fun allocatePersonId(): Int {
        val reused = availableIds.firstOrNull()
        return if (reused != null) {
            availableIds.remove(reused)
            reused
        } else {
            nextPersonIdCounter++
        }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    companion object {
        private const val TAG = "HeadTrackerPipeline"
    }
}
