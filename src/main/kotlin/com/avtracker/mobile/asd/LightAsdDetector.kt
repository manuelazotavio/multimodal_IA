package com.avtracker.mobile.asd

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/** One face seen in a video frame, with the pixel data the ASD backends need. */
class FaceSample(
    val trackId: Int,
    val asdCrop: ByteArray?,      // 112x112 grayscale crop for Light-ASD, null if the box was degenerate
    val faceGray: GrayImage?      // grayscale face box for the pixel-diff backend
)

class GrayImage(val width: Int, val height: Int, val data: ByteArray) {
    operator fun get(x: Int, y: Int): Int = data[y * width + x].toInt() and 0xFF
}

/**
 * Port of LightASDDetector (src/light_asd_detector.py): per-track audio-visual speaking scores.
 *
 * Every [INFER_INTERVAL] seconds each visible track gets its last second of crops (resampled to
 * 25 frames) and audio scored by the network; per-frame scores are kept for [SCORE_RETENTION_SEC]
 * so the audio side can query them retrospectively. Inference runs on [executor] (one at a time
 * per track) so it never blocks the camera analysis thread.
 */
class LightAsdDetector(
    private val scorer: LightAsdScorer,
    private val audio: AudioTimeline,
    private val executor: Executor = Executors.newSingleThreadExecutor { r -> Thread(r, "light-asd").apply { isDaemon = true } },
    private val clock: () -> Double = AudioUtils::nowSec
) {
    private class Crop(val t: Double, val pixels: ByteArray)

    private val lock = Any()
    private val cropBuf = HashMap<Int, ArrayDeque<Crop>>()
    private val scoreBuf = HashMap<Int, ArrayDeque<Pair<Double, Float>>>()
    private val lastInfer = HashMap<Int, Double>()
    private val inFlight = HashSet<Int>()
    private val speakState = HashMap<Int, Boolean>()
    private val speakSince = HashMap<Int, Double>()

    fun update(samples: List<FaceSample>, timestamp: Double) {
        val due = mutableListOf<Int>()
        synchronized(lock) {
            val currentIds = HashSet<Int>()
            for (s in samples) {
                currentIds += s.trackId
                val crop = s.asdCrop ?: continue
                val buf = cropBuf.getOrPut(s.trackId) { ArrayDeque() }
                buf.addLast(Crop(timestamp, crop))
                val cutoff = timestamp - WINDOW_SEC - 0.5
                while (buf.isNotEmpty() && buf.first().t < cutoff) buf.removeFirst()
            }

            for (tid in currentIds) {
                if (timestamp - (lastInfer[tid] ?: 0.0) >= INFER_INTERVAL && tid !in inFlight) due += tid
            }

            // Working state of tracks that left the frame goes away; scoreBuf is kept for retrospective queries.
            for (tid in cropBuf.keys.toList()) {
                if (tid !in currentIds) {
                    cropBuf.remove(tid)
                    lastInfer.remove(tid)
                    speakState.remove(tid)
                    speakSince.remove(tid)
                }
            }

            val cutoff = timestamp - SCORE_RETENTION_SEC
            for (tid in scoreBuf.keys.toList()) {
                val buf = scoreBuf.getValue(tid)
                while (buf.isNotEmpty() && buf.first().first < cutoff) buf.removeFirst()
                if (buf.isEmpty()) scoreBuf.remove(tid)
            }
        }

        for (tid in due) scheduleInference(tid, timestamp)
    }

    private fun scheduleInference(trackId: Int, now: Double) {
        val tStart = now - WINDOW_SEC
        val crops: List<ByteArray>
        synchronized(lock) {
            lastInfer[trackId] = now
            val window = cropBuf[trackId]?.filter { it.t >= tStart } ?: return
            if (window.size < 3) return
            crops = resampleTo25(window)
            inFlight += trackId
        }

        val audioSamples = audio.slice(tStart, now)
        if (audioSamples == null || audioSamples.size < MIN_AUDIO_SAMPLES) {
            synchronized(lock) { inFlight -= trackId }
            return
        }

        executor.execute {
            try {
                val scores = scorer.score(crops, audioSamples)
                if (scores != null) storeScores(trackId, tStart, scores)
            } finally {
                synchronized(lock) { inFlight -= trackId }
            }
        }
    }

    /** Nearest-previous-crop resampling to 25 frames over the window, like the np.linspace / searchsorted code. */
    private fun resampleTo25(window: List<Crop>): List<ByteArray> {
        val ts = window.map { it.t }
        val first = ts.first()
        val last = ts.last()
        return List(VIDEO_FPS) { i ->
            val target = if (VIDEO_FPS == 1) first else first + (last - first) * i / (VIDEO_FPS - 1)
            val idx = (searchSortedLeft(ts, target) - 1).coerceIn(0, window.size - 1)
            window[idx].pixels
        }
    }

    private fun searchSortedLeft(sorted: List<Double>, value: Double): Int {
        var lo = 0
        var hi = sorted.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (sorted[mid] < value) lo = mid + 1 else hi = mid
        }
        return lo
    }

    private fun storeScores(trackId: Int, tStart: Double, scores: FloatArray) {
        synchronized(lock) {
            val buf = scoreBuf.getOrPut(trackId) { ArrayDeque() }
            val lastT = buf.lastOrNull()?.first ?: -1.0
            // Consecutive windows overlap: only append frames newer than what is already stored.
            scores.forEachIndexed { i, s ->
                val tFrame = tStart + (i.toDouble() / VIDEO_FPS) * WINDOW_SEC
                if (tFrame > lastT) buf.addLast(tFrame to s)
            }
        }
    }

    /** Hysteresis on the live score: on above [SPEAK_ON_THRESH] for [SPEAK_ON_SEC], off below [SPEAK_OFF_THRESH] for [SPEAK_OFF_SEC]. */
    fun isSpeakingNow(trackId: Int): Boolean {
        val now = clock()
        val score = getScore(trackId)
        synchronized(lock) {
            val speaking = speakState[trackId] ?: false
            if (speaking) {
                if (score < SPEAK_OFF_THRESH) {
                    val since = speakSince[trackId] ?: now
                    if (now - since >= SPEAK_OFF_SEC) {
                        speakState[trackId] = false
                        speakSince[trackId] = now
                        return false
                    }
                } else {
                    speakSince[trackId] = now
                }
                return true
            }
            if (score >= SPEAK_ON_THRESH) {
                val since = speakSince[trackId] ?: now
                if (now - since >= SPEAK_ON_SEC) {
                    speakState[trackId] = true
                    speakSince[trackId] = now
                    return true
                }
            } else {
                speakSince[trackId] = now
            }
            return false
        }
    }

    /** Mean speaking score over the last two inference intervals. */
    fun getScore(trackId: Int): Float {
        val now = clock()
        synchronized(lock) {
            val recent = scoreBuf[trackId]?.filter { it.first >= now - INFER_INTERVAL * 2 }.orEmpty()
            return if (recent.isEmpty()) 0f else recent.map { it.second }.average().toFloat()
        }
    }

    fun hasScores(timeStart: Double, timeEnd: Double): Boolean = synchronized(lock) {
        scoreBuf.values.any { buf -> buf.any { it.first in timeStart..timeEnd } }
    }

    /** Dominant speaker in [timeStart, timeEnd], or null when nobody clearly speaks or two faces are too close to call. */
    fun getActiveSpeaker(
        timeStart: Double,
        timeEnd: Double,
        minScore: Float = QUERY_MIN_SCORE,
        margin: Float = QUERY_MARGIN
    ): Int? {
        val ranked = synchronized(lock) {
            scoreBuf.mapNotNull { (tid, buf) ->
                val window = buf.filter { it.first in timeStart..timeEnd }.map { it.second }
                if (window.size >= 2) tid to window.average().toFloat() else null
            }
        }.sortedByDescending { it.second }

        if (ranked.isEmpty()) return null
        val (bestId, best) = ranked.first()
        if (best < minScore) return null
        if (ranked.size == 1) return bestId
        if (best - ranked[1].second < margin) return null
        return bestId
    }

    companion object {
        const val WINDOW_SEC = 1.0
        const val VIDEO_FPS = 25
        const val INFER_INTERVAL = 0.60
        const val SCORE_RETENTION_SEC = 20.0
        const val QUERY_MIN_SCORE = 0.50f
        const val QUERY_MARGIN = 0.15f
        const val GUESS_MIN_SCORE = 0.30f
        const val GUESS_MARGIN = 0.10f
        const val SPEAK_ON_THRESH = 0.55f
        const val SPEAK_OFF_THRESH = 0.35f
        const val SPEAK_ON_SEC = 0.25
        const val SPEAK_OFF_SEC = 0.50
        private const val MIN_AUDIO_SAMPLES = 800
    }
}
