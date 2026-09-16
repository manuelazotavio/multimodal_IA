package com.avtracker.mobile.tracking

/**
 * Simplified stand-in for boxmot's ByteTrack (Kalman filter + Hungarian
 * assignment), which has no ready Kotlin/Android equivalent.
 *
 * In the Python pipeline, ByteTrack's per-frame track id is not actually
 * consumed by the downstream appearance-matching logic (only the box
 * coordinates are) - identity continuity across frames comes entirely from
 * the embedding-similarity matching in HeadTrackerPipeline / TrackedPerson,
 * gated by a multi-second staleness timeout. So a greedy IoU tracker here
 * reproduces the same functional behavior as the Python pipeline; a real
 * Kalman-filter tracker (to predict boxes through brief occlusions) can be
 * dropped in later without touching anything downstream.
 */
class SimpleTracker(private val iouThreshold: Float = 0.3f) {

    private val activeTracks = mutableMapOf<Int, Roi>()
    private var nextId = 1

    /** Associates [detections] with active tracks by IoU and returns them tagged with a track id. */
    fun update(detections: List<Roi>): List<Roi> {
        val unmatched = activeTracks.keys.toMutableSet()
        val result = mutableListOf<Roi>()

        for (det in detections) {
            var bestId: Int? = null
            var bestIou = iouThreshold

            for (id in unmatched) {
                val iou = iou(det, activeTracks.getValue(id))
                if (iou > bestIou) {
                    bestIou = iou
                    bestId = id
                }
            }

            val trackId = if (bestId != null) {
                unmatched.remove(bestId)
                bestId
            } else {
                nextId++
            }

            activeTracks[trackId] = det
            result += det.copy(trackId = trackId)
        }

        // Tracks with no detection this frame are dropped, matching the Python
        // pipeline's behavior of only reporting tracks tied to a current detection.
        for (id in unmatched) activeTracks.remove(id)

        return result
    }

    private fun iou(a: Roi, b: Roi): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)

        val interW = (x2 - x1).coerceAtLeast(0)
        val interH = (y2 - y1).coerceAtLeast(0)
        val inter = (interW * interH).toFloat()

        val areaA = (a.width * a.height).toFloat()
        val areaB = (b.width * b.height).toFloat()
        val union = areaA + areaB - inter

        return if (union <= 0f) 0f else inter / union
    }
}
