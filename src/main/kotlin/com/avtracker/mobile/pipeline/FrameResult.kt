package com.avtracker.mobile.pipeline

import com.avtracker.mobile.tracking.Roi

data class TrackedPersonSnapshot(
    val trackId: Int,
    val bbox: Roi,
    val displayName: String,
    val identified: Boolean,
    /** Identity for the fusion layer: the recognised name, or the generic "Person_N" placeholder. */
    val name: String = displayName,
    val confidence: Float = 0f
)

data class FrameResult(
    val persons: List<TrackedPersonSnapshot>,
    val knownDatabaseSize: Int
)
