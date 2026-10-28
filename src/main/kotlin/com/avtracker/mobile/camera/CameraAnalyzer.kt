package com.avtracker.mobile.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import com.avtracker.mobile.audio.AudioUtils
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
 *
 * [onFrame] runs on the analysis thread while the bitmap is still alive, so the multimodal layer can crop faces
 * from it; its return value is handed to [onResult] together with the result.
 */
class CameraAnalyzer<T>(
    private val pipeline: HeadTrackerPipeline,
    private val onFrame: (frame: Bitmap, result: FrameResult, timestampSec: Double) -> T,
    private val onResult: (result: FrameResult, extra: T, width: Int, height: Int, processingMs: Long, transform: OutputTransform) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(imageProxy: ImageProxy) {
        val start = System.currentTimeMillis()
        val frameTimeSec = AudioUtils.nowSec()
        try {
            // Detection coordinates belong to the full, rotated buffer, not to a stretched preview.
            val transform = ImageProxyTransformFactory().apply {
                isUsingRotationDegrees = true
                isUsingCropRect = false
            }.getOutputTransform(imageProxy)
            val original = imageProxy.toBitmap()
            val bitmap = original.rotated(imageProxy.imageInfo.rotationDegrees)
            try {
                val result = pipeline.processFrame(bitmap)
                val extra = onFrame(bitmap, result, frameTimeSec)
                onResult(result, extra, bitmap.width, bitmap.height, System.currentTimeMillis() - start, transform)
            } finally {
                if (bitmap !== original) bitmap.recycle()
                original.recycle()
            }
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
