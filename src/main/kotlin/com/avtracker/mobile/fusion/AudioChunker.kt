package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils

/**
 * Port of RealtimeTranscriber._process_audio_chunks: sliding windows so diarization labels stay consistent
 * between consecutive chunks.
 *
 *  - chunk 0: the first 8 s, consumed entirely (faster first identification)
 *  - chunk 1: the next 15 s, all new (no overlap with chunk 0)
 *  - chunk k >= 2: 15 s windows advancing 5 s; the first 10 s are context, only the last 5 s are new
 */
class AudioChunker(private val lowLatency: Boolean = false) {
    class Window(val startSec: Double, val endSec: Double, val newZoneStartSamples: Int)

    private var index = 0
    private var cursorSec: Double? = null

    /** The next window if the timeline already holds all of its audio, else null. */
    fun next(timeline: AudioTimeline): Window? {
        if (!timeline.hasAudio) return null
        val start = cursorSec ?: timeline.startTimeSec.also { cursorSec = it }

        if (lowLatency) {
            // Four seconds of new audio each time, with up to four seconds of preceding context.
            // No 15-second second-window warm-up and no overlapping new zones.
            val end = start + 4.0
            if (timeline.endTimeSec < end) return null
            val contextStart = maxOf(timeline.startTimeSec, end - 8.0)
            cursorSec = end
            return Window(contextStart, end, ((start - contextStart) * AudioUtils.SAMPLE_RATE).toInt())
        }

        val window = when (index) {
            0 -> Window(start, start + FIRST_WINDOW_SEC, 0)
            1 -> Window(start, start + WINDOW_SEC, 0)
            else -> Window(start, start + WINDOW_SEC, ((WINDOW_SEC - STEP_SEC) * AudioUtils.SAMPLE_RATE).toInt())
        }
        if (timeline.endTimeSec < window.endSec) return null

        cursorSec = start + if (index == 0) FIRST_WINDOW_SEC else STEP_SEC
        index++
        return window
    }

    companion object {
        const val FIRST_WINDOW_SEC = 8.0
        const val WINDOW_SEC = 15.0
        const val STEP_SEC = 5.0
    }
}
