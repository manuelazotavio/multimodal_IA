package com.avtracker.mobile.profile

import kotlinx.serialization.Serializable

/**
 * On-disk visual profile format, replacing the Python .pkl format with plain
 * JSON so profiles can be bundled as Android assets or written to app-private
 * storage without pickle/numpy dependencies.
 */
@Serializable
data class VisualProfileMetadata(
    val personName: String,
    val createdAt: String,
    val numImages: Int,
    val modelName: String,
    val embeddingDimension: Int,
    val imageQualityScore: Float,
    val captureEnvironment: String
)

@Serializable
data class VisualProfile(
    val name: String,
    val averageEmbedding: List<Float>,
    val metadata: VisualProfileMetadata
)
