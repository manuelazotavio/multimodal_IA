package com.avtracker.mobile.audio

/**
 * Rolling window of the continuous microphone stream, addressable by time.
 *
 * Replaces the Python `asd_audio_buf` deque of (t_start, chunk) pairs: Light-ASD needs the last
 * second of audio aligned to video timestamps, and the segment processor needs arbitrary
 * [tStart, tEnd] slices. Timestamps use [AudioUtils.nowSec].
 */
class AudioTimeline(private val capacitySamples: Int = AudioUtils.SAMPLE_RATE * 40) {
    private val ring = FloatArray(capacitySamples)
    private var written = 0L
    private var originSec = 0.0

    /** Time of the first sample ever appended. */
    val startTimeSec: Double
        @Synchronized get() = originSec

    val endTimeSec: Double
        @Synchronized get() = originSec + written.toDouble() / AudioUtils.SAMPLE_RATE

    val hasAudio: Boolean
        @Synchronized get() = written > 0

    @Synchronized
    fun append(chunk: FloatArray, chunkStartSec: Double) {
        if (written == 0L) originSec = chunkStartSec
        for (v in chunk) {
            ring[(written % capacitySamples).toInt()] = v
            written++
        }
    }

    /** Samples for [tStartSec, tEndSec), clamped to what is still buffered; null if nothing overlaps. */
    @Synchronized
    fun slice(tStartSec: Double, tEndSec: Double): FloatArray? {
        if (written == 0L) return null
        val oldest = maxOf(0L, written - capacitySamples)
        val from = maxOf(oldest, ((tStartSec - originSec) * AudioUtils.SAMPLE_RATE).toLong())
        val to = minOf(written, ((tEndSec - originSec) * AudioUtils.SAMPLE_RATE).toLong())
        if (to <= from) return null
        return FloatArray((to - from).toInt()) { ring[((from + it) % capacitySamples).toInt()] }
    }
}
