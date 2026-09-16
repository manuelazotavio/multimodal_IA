package com.avtracker.mobile.tracking

/** Port of the Python ROI class: a detection/track box with helper accessors. */
data class Roi(
    val x1: Int,
    val y1: Int,
    val x2: Int,
    val y2: Int,
    val confidence: Float = 0f,
    val classId: Int = 0,
    val trackId: Int? = null
) {
    val width: Int get() = x2 - x1
    val height: Int get() = y2 - y1
    val centerX: Int get() = x1 + width / 2
    val centerY: Int get() = y1 + height / 2

    companion object {
        fun clamped(
            x1: Float, y1: Float, x2: Float, y2: Float,
            confidence: Float = 0f, classId: Int = 0, trackId: Int? = null
        ): Roi = Roi(
            x1.coerceAtLeast(0f).toInt(),
            y1.coerceAtLeast(0f).toInt(),
            x2.coerceAtLeast(0f).toInt(),
            y2.coerceAtLeast(0f).toInt(),
            confidence, classId, trackId
        )
    }
}
