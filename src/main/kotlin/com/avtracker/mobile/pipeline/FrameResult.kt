package com.avtracker.mobile.pipeline

import com.avtracker.mobile.tracking.Roi

data class TrackedPersonSnapshot(
    val trackId: Int,
    val bbox: Roi,
    val displayName: String,
    val identified: Boolean
)

data class FrameResult(
    val persons: List<TrackedPersonSnapshot>,
    val knownDatabaseSize: Int
)
