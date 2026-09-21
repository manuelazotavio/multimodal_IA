package com.avtracker.mobile.gender

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min

/**
 * Gender from the fundamental frequency of the voice: port of RealtimeTranscriber._detect_gender_from_audio,
 * which runs `librosa.yin(audio, fmin=50, fmax=500, sr=16000)` and thresholds the median f0
 * (above 155 Hz female, below 140 Hz male, in between undecided).
 */
object PitchGender {
    private const val SAMPLE_RATE = 16_000
    private const val FRAME_LENGTH = 2048
    private const val HOP_LENGTH = FRAME_LENGTH / 4
    private const val TROUGH_THRESHOLD = 0.1
    private const val FLOAT32_TINY = 1.17549435e-38

    fun detect(audio: FloatArray): String? {
        val f0 = yin(audio, fmin = 50.0, fmax = 500.0, sampleRate = SAMPLE_RATE)
        val valid = f0.filter { it > 0 }.sorted()
        if (valid.size < 10) return null
        val median = if (valid.size % 2 == 1) valid[valid.size / 2] else (valid[valid.size / 2 - 1] + valid[valid.size / 2]) / 2
        return when {
            median > 155 -> NameGender.FEMALE
            median < 140 -> NameGender.MALE
            else -> null
        }
    }

    /**
     * librosa.yin (0.11): cumulative mean normalised difference on centred 2048-sample frames (zero-padded, hop 512),
     * absolute threshold 0.1 with global-minimum fallback, then parabolic interpolation.
     */
    fun yin(y: FloatArray, fmin: Double, fmax: Double, sampleRate: Int): DoubleArray {
        val padded = DoubleArray(y.size + FRAME_LENGTH) { i ->
            val src = i - FRAME_LENGTH / 2
            if (src in y.indices) y[src].toDouble() else 0.0
        }
        if (padded.size < FRAME_LENGTH) return DoubleArray(0)
        val frames = 1 + (padded.size - FRAME_LENGTH) / HOP_LENGTH

        val minPeriod = floor(sampleRate / fmax).toInt()
        val maxPeriod = min(ceil(sampleRate / fmin).toInt(), FRAME_LENGTH - 1)

        return DoubleArray(frames) { f ->
            val yinFrame = cumulativeMeanNormalizedDifference(padded, f * HOP_LENGTH, minPeriod, maxPeriod)
            periodToF0(yinFrame, minPeriod, sampleRate)
        }
    }

    /** Frame `start until start + FRAME_LENGTH` -> d'(tau) for tau = minPeriod..maxPeriod. */
    private fun cumulativeMeanNormalizedDifference(x: DoubleArray, start: Int, minPeriod: Int, maxPeriod: Int): DoubleArray {
        // Linear autocorrelation up to maxPeriod (librosa does the same through a zero-padded FFT).
        val acf = DoubleArray(maxPeriod + 1)
        for (lag in 0..maxPeriod) {
            var sum = 0.0
            for (m in 0 until FRAME_LENGTH - lag) sum += x[start + m] * x[start + m + lag]
            acf[lag] = sum
        }

        // d(k) = 2 * (acf(0) - acf(k)) - sum_{m < k} x(m)^2, d(0) = 0
        val difference = DoubleArray(maxPeriod + 1)
        var energy = 0.0
        for (k in 1..maxPeriod) {
            energy += x[start + k - 1] * x[start + k - 1]
            difference[k] = 2 * (acf[0] - acf[k]) - energy
        }

        val out = DoubleArray(maxPeriod - minPeriod + 1)
        var running = 0.0
        for (k in 1..maxPeriod) {
            running += difference[k]
            if (k >= minPeriod) out[k - minPeriod] = difference[k] / (running / k + FLOAT32_TINY)
        }
        return out
    }

    private fun periodToF0(yinFrame: DoubleArray, minPeriod: Int, sampleRate: Int): Double {
        val n = yinFrame.size

        // Local minima: x[i] < x[i-1] and x[i] <= x[i+1], with edge-padding; the first bin is handled separately.
        fun isTrough(i: Int): Boolean {
            if (i == 0) return n > 1 && yinFrame[0] < yinFrame[1]
            val next = if (i == n - 1) yinFrame[i] else yinFrame[i + 1]
            return yinFrame[i] < yinFrame[i - 1] && yinFrame[i] <= next
        }

        var target = -1
        for (i in 0 until n) if (isTrough(i) && yinFrame[i] < TROUGH_THRESHOLD) { target = i; break }
        if (target < 0) {
            target = 0
            for (i in 1 until n) if (yinFrame[i] < yinFrame[target]) target = i
        }

        val shift = if (target == 0 || target == n - 1) 0.0 else parabolicShift(yinFrame, target)
        return sampleRate / (minPeriod + target + shift)
    }

    /** librosa's _pi_stencil: offset of the parabola's optimum, 0 if it would move by a full bin or more. */
    private fun parabolicShift(x: DoubleArray, i: Int): Double {
        val a = x[i + 1] + x[i - 1] - 2 * x[i]
        val b = (x[i + 1] - x[i - 1]) / 2
        return if (abs(b) >= abs(a)) 0.0 else -b / a
    }
}
