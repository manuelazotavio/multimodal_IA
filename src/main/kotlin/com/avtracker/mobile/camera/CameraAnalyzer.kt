package com.avtracker.mobile.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.toBitmap
import com.avtracker.mobile.pipeline.FrameResult
import com.avtracker.mobile.pipeline.HeadTrackerPipeline

/**
 * CameraX analyzer that feeds frames through the tracking pipeline on the
 * analysis executor. Requires ImageAnalysis to be configured with
 * OUTPUT_IMAGE_FORMAT_RGBA_8888 so ImageProxy.toBitmap() is available.
 *
 * Mirrors the Python capture thread + bounded queue: CameraX's
 * STRATEGY_KEEP_ONLY_LATEST backpressure setting (applied by the caller)
 * plays the same "drop old frames under load" role as the Python
 * queue.Queue(maxsize=5).
 */
class CameraAnalyzer(
    private val pipeline: HeadTrackerPipeline,
    private val onResult: (FrameResult, width: Int, height: Int, processingMs: Long) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(imageProxy: ImageProxy) {
        val start = System.currentTimeMillis()
        try {
            val bitmap = imageProxy.toBitmap().rotated(imageProxy.imageInfo.rotationDegrees)
            val result = pipeline.processFrame(bitmap)
            onResult(result, bitmap.width, bitmap.height, System.currentTimeMillis() - start)
        } catch (t: Throwable) {
            Log.e("CameraAnalyzer", "Frame processing failed", t)
        } finally {
            imageProxy.close()
        }
    }

    private fun Bitmap.rotated(degrees: Int): Bitmap {
        if (degrees == 0) return this
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }
}
