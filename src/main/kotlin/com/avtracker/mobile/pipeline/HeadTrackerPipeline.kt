package com.avtracker.mobile.pipeline

import android.graphics.Bitmap
import com.avtracker.mobile.detection.YoloDetector
import com.avtracker.mobile.embedding.EdgeFaceEmbedder
import com.avtracker.mobile.tracking.PersonIdTracker
import com.avtracker.mobile.tracking.Roi
import com.avtracker.mobile.tracking.bytetrack.Detection

/**
 * The face side of run_multimodal_tracker.video_loop: YOLO face detection, the cap of [numSpeakers] largest faces,
 * then [PersonIdTracker] (ByteTrack + EdgeFace identity matching, auto-enrolment, rename cycle).
 */
class HeadTrackerPipeline(
    private val detector: YoloDetector,
    private val embedder: EdgeFaceEmbedder,
    val tracker: PersonIdTracker,
    private val numSpeakers: Int? = null
) {
    fun processFrame(frame: Bitmap): FrameResult {
        var heads = detector.detectHeads(frame)
        if (numSpeakers != null && numSpeakers > 0 && heads.size > numSpeakers) {
            heads = heads.sortedByDescending { (it.x2 - it.x1) * (it.y2 - it.y1) }.take(numSpeakers)
        }
        val detections = heads.map {
            Detection(it.x1.toDouble(), it.y1.toDouble(), it.x2.toDouble(), it.y2.toDouble(), it.confidence.toDouble())
        }

        val faces = tracker.update(frame.width, frame.height, detections) { left, top, right, bottom ->
            val crop = Bitmap.createBitmap(frame, left, top, right - left, bottom - top)
            try {
                embedder.extractEmbedding(crop)
            } finally {
                if (crop !== frame) crop.recycle()
            }
        }

        return FrameResult(
            persons = faces.map {
                TrackedPersonSnapshot(
                    trackId = it.trackId,
                    bbox = Roi(it.x1.toInt(), it.y1.toInt(), it.x2.toInt(), it.y2.toInt(), it.confidence.toFloat(), trackId = it.trackId),
                    displayName = "${it.name} (%.2f)".format(java.util.Locale.ROOT, it.confidence),
                    identified = it.name != PersonIdTracker.UNKNOWN,
                    name = it.name,
                    confidence = it.confidence.toFloat()
                )
            },
            knownDatabaseSize = tracker.knownNames.size
        )
    }
}
