package com.avtracker.mobile.detection

/** A detected head bounding box in the coordinate space of the original frame. */
data class HeadBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float
)
