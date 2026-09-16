package com.avtracker.mobile.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import kotlin.math.sqrt

/**
 * EdgeFace embedding extractor.
 *
 * Port of the EdgeFaceEmbedder class duplicated in personid_tracker.py /
 * person_verification.py / video_profile_creator.py. Unlike the YOLO
 * detector, the Python code explicitly converts BGR -> RGB before feeding
 * EdgeFace, so this preprocessing uses natural RGB channel order.
 */
class EdgeFaceEmbedder(
    private val env: OrtEnvironment,
    modelBytes: ByteArray
) {
    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())
    private val inputName: String = session.inputInfo.keys.first()

    fun close() = session.close()

    fun extractEmbedding(faceCrop: Bitmap): FloatArray = extractEmbeddings(listOf(faceCrop)).first()

    /** Batched embedding extraction, mirroring the Python batch preprocessing path. */
    fun extractEmbeddings(faceCrops: List<Bitmap>): List<FloatArray> {
        if (faceCrops.isEmpty()) return emptyList()

        val batch = faceCrops.size
        val data = FloatArray(batch * 3 * INPUT_SIZE * INPUT_SIZE)
        val channelSize = INPUT_SIZE * INPUT_SIZE

        faceCrops.forEachIndexed { index, crop ->
            val resized = Bitmap.createScaledBitmap(crop, INPUT_SIZE, INPUT_SIZE, true)
            val pixels = IntArray(channelSize)
            resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            if (resized !== crop) resized.recycle()

            val base = index * 3 * channelSize
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
                val g = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
                val b = ((p and 0xFF) - 127.5f) / 127.5f
                data[base + i] = r
                data[base + channelSize + i] = g
                data[base + 2 * channelSize + i] = b
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(data),
            longArrayOf(batch.toLong(), 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        @Suppress("UNCHECKED_CAST")
        val embeddings = inputTensor.use {
            session.run(mapOf(inputName to it)).use { result ->
                (result[0].value as Array<FloatArray>)
            }
        }

        return embeddings.map { l2Normalize(it) }
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var normSq = 0f
        for (v in vector) normSq += v * v
        val norm = sqrt(normSq)
        if (norm == 0f) return vector
        return FloatArray(vector.size) { vector[it] / norm }
    }

    companion object {
        private const val INPUT_SIZE = 112

        fun fromAssets(context: Context, assetPath: String): EdgeFaceEmbedder {
            val bytes = context.assets.open(assetPath).use { it.readBytes() }
            return EdgeFaceEmbedder(OrtEnvironment.getEnvironment(), bytes)
        }
    }
}
