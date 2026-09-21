package com.avtracker.mobile.audio

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

object AudioUtils {
    const val SAMPLE_RATE = 16_000

    /** Monotonic clock in seconds, shared by the audio and video sides so their timestamps are comparable. */
    fun nowSec(): Double = System.nanoTime() / 1e9

    /** Port of RealtimeTranscriber._normalize_audio: scale to [targetDb] dBFS RMS, then peak-limit to 1.0. */
    fun normalize(audio: FloatArray, targetDb: Double = -20.0): FloatArray {
        if (audio.isEmpty()) return audio
        var sumSq = 0.0
        for (v in audio) sumSq += v.toDouble() * v
        val rms = sqrt(sumSq / audio.size)
        if (rms < 1e-6) return audio

        val scale = (10.0.pow(targetDb / 20.0) / rms).toFloat()
        val out = FloatArray(audio.size) { audio[it] * scale }
        var peak = 0f
        for (v in out) peak = maxOf(peak, abs(v))
        if (peak > 1f) for (i in out.indices) out[i] /= peak
        return out
    }

    fun peak(audio: FloatArray): Float {
        var p = 0f
        for (v in audio) p = maxOf(p, abs(v))
        return p
    }

    fun l2Normalize(v: FloatArray): FloatArray {
        var sumSq = 0.0
        for (x in v) sumSq += x.toDouble() * x
        val norm = sqrt(sumSq).toFloat()
        if (norm == 0f) return v
        return FloatArray(v.size) { v[it] / norm }
    }

    fun dot(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var d = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            d += a[i].toDouble() * b[i]
            na += a[i].toDouble() * a[i]
            nb += b[i].toDouble() * b[i]
        }
        if (na == 0.0 || nb == 0.0) return 0f
        return (d / (sqrt(na) * sqrt(nb))).toFloat()
    }
}
