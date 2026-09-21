package com.avtracker.mobile.tracking.bytetrack

import kotlin.math.sqrt

/**
 * Port of boxmot's KalmanFilterXYAH (state x, y, a, h, vx, vy, va, vh; constant-velocity model) in double precision,
 * with the same noise weights and the same operations, so the predicted boxes match boxmot's.
 */
class KalmanFilterXyah {
    private val stdWeightPosition = 1.0 / 20
    private val stdWeightVelocity = 1.0 / 160

    fun initiate(measurement: DoubleArray): Pair<DoubleArray, Array<DoubleArray>> {
        val mean = DoubleArray(8)
        for (i in 0 until 4) mean[i] = measurement[i]

        val h = measurement[3]
        val std = doubleArrayOf(
            2 * stdWeightPosition * h,
            2 * stdWeightPosition * h,
            1e-2,
            2 * stdWeightPosition * h,
            10 * stdWeightVelocity * h,
            10 * stdWeightVelocity * h,
            1e-5,
            10 * stdWeightVelocity * h
        )
        val covariance = Array(8) { DoubleArray(8) }
        for (i in 0 until 8) covariance[i][i] = std[i] * std[i]
        return mean to covariance
    }

    /** F x with F = [[I, I], [0, I]] (dt = 1), and F P F^T + Q. */
    fun predict(mean: DoubleArray, covariance: Array<DoubleArray>): Pair<DoubleArray, Array<DoubleArray>> {
        val h = mean[3]
        val std = doubleArrayOf(
            stdWeightPosition * h, stdWeightPosition * h, 1e-2, stdWeightPosition * h,
            stdWeightVelocity * h, stdWeightVelocity * h, 1e-5, stdWeightVelocity * h
        )

        val newMean = DoubleArray(8) { i -> if (i < 4) mean[i] + mean[i + 4] else mean[i] }

        // rows of F P: row i (< 4) is P[i] + P[i + 4]
        val fp = Array(8) { i -> DoubleArray(8) { j -> if (i < 4) covariance[i][j] + covariance[i + 4][j] else covariance[i][j] } }
        // (F P) F^T: column j (< 4) is (F P)[.][j] + (F P)[.][j + 4]
        val newCov = Array(8) { i -> DoubleArray(8) { j -> if (j < 4) fp[i][j] + fp[i][j + 4] else fp[i][j] } }
        for (i in 0 until 8) newCov[i][i] += std[i] * std[i]
        return newMean to newCov
    }

    /** Correction with a measurement (x, y, a, h); the confidence is always 0 in boxmot's ByteTrack. */
    fun update(mean: DoubleArray, covariance: Array<DoubleArray>, measurement: DoubleArray): Pair<DoubleArray, Array<DoubleArray>> {
        val h = mean[3]
        val noise = doubleArrayOf(stdWeightPosition * h, stdWeightPosition * h, 1e-1, stdWeightPosition * h)

        // S = H P H^T + R
        val s = Array(4) { i -> DoubleArray(4) { j -> covariance[i][j] } }
        for (i in 0 until 4) s[i][i] += noise[i] * noise[i]

        // K = P H^T S^-1, via the Cholesky factor of S
        val l = cholesky(s)
        val pht = Array(8) { i -> DoubleArray(4) { j -> covariance[i][j] } }
        val k = Array(8) { i -> solveCholeskyRow(l, pht[i]) }

        val newMean = DoubleArray(8) { i ->
            var acc = 0.0
            for (j in 0 until 4) acc += (measurement[j] - mean[j]) * k[i][j]
            mean[i] + acc
        }

        // P - K S K^T
        val ks = Array(8) { i -> DoubleArray(4) { j -> var a = 0.0; for (m in 0 until 4) a += k[i][m] * s[m][j]; a } }
        val newCov = Array(8) { i ->
            DoubleArray(8) { j ->
                var a = 0.0
                for (m in 0 until 4) a += ks[i][m] * k[j][m]
                covariance[i][j] - a
            }
        }
        return newMean to newCov
    }

    private fun cholesky(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val l = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = a[i][j]
                for (m in 0 until j) sum -= l[i][m] * l[j][m]
                l[i][j] = if (i == j) sqrt(sum) else sum / l[j][j]
            }
        }
        return l
    }

    /** Solves x S = b (S symmetric, S = L L^T) for the row vector x. */
    private fun solveCholeskyRow(l: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = l.size
        val y = DoubleArray(n)
        for (i in 0 until n) {
            var sum = b[i]
            for (m in 0 until i) sum -= l[i][m] * y[m]
            y[i] = sum / l[i][i]
        }
        val x = DoubleArray(n)
        for (i in n - 1 downTo 0) {
            var sum = y[i]
            for (m in i + 1 until n) sum -= l[m][i] * x[m]
            x[i] = sum / l[i][i]
        }
        return x
    }
}
