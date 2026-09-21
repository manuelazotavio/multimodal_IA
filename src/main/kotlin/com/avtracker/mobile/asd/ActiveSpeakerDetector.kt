package com.avtracker.mobile.asd

import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.fusion.SpeakerActivity
import kotlin.math.abs

/**
 * Port of ActiveSpeakerDetector (src/active_speaker_detector.py).
 *
 * Primary backend: Light-ASD ([LightAsdDetector]). Fallback: mouth-region pixel-difference,
 * which keeps running every frame so there is always an answer when Light-ASD has no scores
 * for the requested window.
 */
class ActiveSpeakerDetector(
    private val lightAsd: LightAsdDetector?,
    private val bufferSeconds: Double = 15.0,
    private val minFrames: Int = 2,
    private val dominantRatio: Float = 1.4f,
    private val clock: () -> Double = AudioUtils::nowSec
) : SpeakerActivity {
    private val lock = Any()
    private val prevMouth = HashMap<Int, GrayImage>()
    private val prevUpper = HashMap<Int, GrayImage>()
    // Insertion-ordered like the Python dict, so score ties are broken the same way.
    private val activity = LinkedHashMap<Int, ArrayDeque<Pair<Double, Float>>>()
    private val speakState = HashMap<Int, Boolean>()

    fun update(samples: List<FaceSample>, timestamp: Double) {
        synchronized(lock) {
            val currentIds = HashSet<Int>()
            for (s in samples) {
                val face = s.faceGray ?: continue
                val h = face.height
                val w = face.width
                if (h <= 0 || w <= 0) continue

                val mouth = region(face, MOUTH_SIDE_MARGIN, MOUTH_TOP_RATIO, 1f - MOUTH_SIDE_MARGIN, 1f)
                val upper = region(face, MOUTH_SIDE_MARGIN, 0f, 1f - MOUTH_SIDE_MARGIN, UPPER_BOTTOM_RATIO)
                if (mouth.width * mouth.height == 0) continue
                currentIds += s.trackId

                val mouthDiff = prevMouth[s.trackId]?.let { meanAbsDiff(mouth, it) } ?: 0f
                val upperDiff = prevUpper[s.trackId]?.let { meanAbsDiff(upper, it) } ?: 0f

                // Reject occlusion: a hand over the mouth moves the whole face region similarly.
                val diff = if (mouthDiff > 0f && mouthDiff < OCCLUSION_RATIO * (upperDiff + 0.5f)) 0f else mouthDiff

                prevMouth[s.trackId] = mouth
                prevUpper[s.trackId] = upper

                val buf = activity.getOrPut(s.trackId) { ArrayDeque() }
                buf.addLast(timestamp to diff)
                val cutoff = timestamp - bufferSeconds
                while (buf.isNotEmpty() && buf.first().first < cutoff) buf.removeFirst()
            }
            for (tid in prevMouth.keys.toList()) {
                if (tid !in currentIds) {
                    prevMouth.remove(tid)
                    prevUpper.remove(tid)
                }
            }
        }
        lightAsd?.update(samples, timestamp)
    }

    override fun getActiveSpeaker(timeStart: Double, timeEnd: Double): Int? {
        if (lightAsd != null && lightAsd.hasScores(timeStart, timeEnd)) {
            return lightAsd.getActiveSpeaker(timeStart, timeEnd)
        }

        val sorted = pixelScores(timeStart, timeEnd).entries.sortedByDescending { it.value }
        if (sorted.isEmpty()) return null
        val (bestId, bestScore) = sorted.first().toPair()

        if (bestScore < NOISE_FLOOR) return null
        if (sorted.size == 1) return bestId

        val second = sorted[1].value
        if (second < NOISE_FLOOR) return bestId
        return if (bestScore / second < dominantRatio) null else bestId
    }

    /** Weaker fallback used when the verifier has no match: the face with clearly the most mouth movement. */
    override fun getBestGuess(timeStart: Double, timeEnd: Double): Int? {
        if (lightAsd != null && lightAsd.hasScores(timeStart, timeEnd)) {
            return lightAsd.getActiveSpeaker(
                timeStart, timeEnd,
                minScore = LightAsdDetector.GUESS_MIN_SCORE,
                margin = LightAsdDetector.GUESS_MARGIN
            )
        }

        val sorted = pixelScores(timeStart, timeEnd).entries.sortedByDescending { it.value }
        if (sorted.isEmpty()) return null
        val (bestId, bestScore) = sorted.first().toPair()

        if (bestScore <= 0f) return null
        if (sorted.size == 1) return bestId

        val second = sorted[1].value
        if (second > 0f && bestScore / second < 1.5f) return null
        return bestId
    }

    /** Real-time "is this face speaking now" for the overlay: Light-ASD hysteresis, else pixel-diff hysteresis. */
    fun isSpeakingNow(trackId: Int): Boolean {
        if (lightAsd != null) return lightAsd.isSpeakingNow(trackId)

        val now = clock()
        synchronized(lock) {
            val buf = activity[trackId]
            val speaking = speakState[trackId] ?: false
            if (buf.isNullOrEmpty()) {
                if (speaking) speakState[trackId] = false
                return false
            }
            if (speaking) {
                val window = buf.filter { it.first >= now - SPEAK_OFF_SEC }.map { it.second }
                val belowFloor = window.size >= minFrames && window.average() < NOISE_FLOOR
                if (belowFloor) speakState[trackId] = false
                return !belowFloor
            }
            val window = buf.filter { it.first >= now - SPEAK_ON_SEC }.map { it.second }
            val aboveFloor = window.size >= minFrames && window.average() >= SPEAK_ON_FLOOR
            if (aboveFloor) speakState[trackId] = true
            return aboveFloor
        }
    }

    private fun pixelScores(timeStart: Double, timeEnd: Double): Map<Int, Float> = synchronized(lock) {
        val scores = LinkedHashMap<Int, Float>()
        for ((tid, buf) in activity) {
            val window = buf.filter { it.first in timeStart..timeEnd }.map { it.second }
            if (window.size >= minFrames) scores[tid] = window.average().toFloat()
        }
        scores
    }

    private fun region(img: GrayImage, x1: Float, y1: Float, x2: Float, y2: Float): GrayImage {
        val left = (img.width * x1).toInt()
        val right = (img.width * x2).toInt().coerceAtMost(img.width)
        val top = (img.height * y1).toInt()
        val bottom = (img.height * y2).toInt().coerceAtMost(img.height)
        val w = (right - left).coerceAtLeast(0)
        val h = (bottom - top).coerceAtLeast(0)
        val out = ByteArray(w * h)
        for (y in 0 until h) System.arraycopy(img.data, (top + y) * img.width + left, out, y * w, w)
        return GrayImage(w, h, out)
    }

    /** Mean absolute pixel difference; 0 when the two regions differ in shape (a box that changed size). */
    private fun meanAbsDiff(a: GrayImage, b: GrayImage): Float {
        if (a.width != b.width || a.height != b.height || a.data.isEmpty()) return 0f
        var sum = 0L
        for (i in a.data.indices) sum += abs((a.data[i].toInt() and 0xFF) - (b.data[i].toInt() and 0xFF))
        return sum.toFloat() / a.data.size
    }

    companion object {
        // Mouth region: 60-100% vertical, 15-85% horizontal of the face box.
        const val MOUTH_TOP_RATIO = 0.60f
        const val MOUTH_SIDE_MARGIN = 0.15f
        const val UPPER_BOTTOM_RATIO = 0.50f
        const val OCCLUSION_RATIO = 1.8f
        const val NOISE_FLOOR = 1.5f
        const val SPEAK_ON_FLOOR = 3.0f
        const val SPEAK_ON_SEC = 0.35
        const val SPEAK_OFF_SEC = 0.6
    }
}
