package com.avtracker.mobile.detection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.FloatBuffer

/**
 * Unified YOLOv5 / YOLOv8 / YOLOv11 ONNX detector (heads or faces, class 0).
 *
 * The model format is auto-detected from the output tensors:
 *  - YOLOv8 (Ultralytics export, e.g. yolov8n-face used by run_multimodal_tracker.py):
 *    single channels-first output [1, 4 + nc, N] = rows xc, yc, w, h, class scores (no objectness).
 *    Input is RGB in [0, 1] and the letterbox padding is gray (114), as Ultralytics does.
 *  - YOLOv5: single output [1, N, 7] = [xc, yc, w, h, obj, class0, class1]
 *  - YOLOv11: three outputs boxes[1,N,4] (y1,x1,y2,x2), scores[1,N], classes[1,N]
 *
 * Note: the legacy YOLOv5/YOLOv11 Python pipeline fed raw BGR frames (straight
 * from cv2, no color conversion) into the model, so those two keep B,G,R
 * channel order and black padding to stay numerically identical.
 */
class YoloDetector(
    env: OrtEnvironment,
    modelBytes: ByteArray,
    private val confidenceThreshold: Float = 0.5f,
    private val nmsThreshold: Float = 0.45f
) {
    private val env = env
    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())
    private val inputName: String = session.inputInfo.keys.first()
    private val modelWidth: Int
    private val modelHeight: Int
    private val modelType: ModelType

    private enum class ModelType { YOLOV5, YOLOV8, YOLOV11 }

    init {
        val inputShape = (session.inputInfo.getValue(inputName).info as ai.onnxruntime.TensorInfo).shape
        modelHeight = inputShape.getOrElse(2) { 640L }.let { if (it <= 0) 640 else it.toInt() }
        modelWidth = inputShape.getOrElse(3) { 640L }.let { if (it <= 0) 640 else it.toInt() }

        modelType = when (session.outputInfo.size) {
            1 -> if (isChannelsFirst(session.outputInfo.values.first().info as ai.onnxruntime.TensorInfo)) {
                ModelType.YOLOV8
            } else {
                ModelType.YOLOV5
            }
            3 -> ModelType.YOLOV11
            else -> throw IllegalStateException("Unsupported model with ${session.outputInfo.size} outputs")
        }
    }

    // [1, 4 + nc, N] (YOLOv8) has far fewer attribute rows than detections; [1, N, 5 + nc] (YOLOv5) is the opposite.
    private fun isChannelsFirst(output: ai.onnxruntime.TensorInfo): Boolean {
        val attrs = output.shape.getOrElse(1) { -1L }
        val detections = output.shape.getOrElse(2) { -1L }
        return attrs > 0 && detections > 0 && attrs < detections
    }

    fun close() = session.close()

    /** Detect heads (class 0) in [frame] and return boxes in the frame's own coordinate space. */
    fun detectHeads(frame: Bitmap): List<HeadBox> = detect(frame, targetClass = 0)

    private fun detect(frame: Bitmap, targetClass: Int): List<HeadBox> {
        val isV8 = modelType == ModelType.YOLOV8
        val letterboxed = letterbox(frame, modelWidth, modelHeight, if (isV8) YOLOV8_PAD_COLOR else Color.BLACK)
        val inputTensor = toChwTensor(letterboxed.bitmap, bgr = !isV8)

        val outputs = inputTensor.use {
            session.run(mapOf(inputName to it))
        }

        return outputs.use {
            when (modelType) {
                ModelType.YOLOV5 -> processYolov5(it, targetClass, letterboxed)
                ModelType.YOLOV8 -> processYolov8(it, targetClass, letterboxed)
                ModelType.YOLOV11 -> processYolov11(it, targetClass, letterboxed)
            }
        }
    }

    // ------------------------------------------------------------------
    // Preprocessing
    // ------------------------------------------------------------------

    private class Letterbox(val bitmap: Bitmap, val padTop: Int, val padLeft: Int, val scaleRatio: Float)

    private class Candidate(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

    private fun letterbox(frame: Bitmap, targetWidth: Int, targetHeight: Int, padColor: Int): Letterbox {
        val ratio = minOf(targetWidth.toFloat() / frame.width, targetHeight.toFloat() / frame.height)
        val newWidth = (frame.width * ratio).toInt()
        val newHeight = (frame.height * ratio).toInt()

        val resized = Bitmap.createScaledBitmap(frame, newWidth, newHeight, true)

        val padLeft = (targetWidth - newWidth) / 2
        val padTop = (targetHeight - newHeight) / 2

        val padded = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawColor(padColor)
        canvas.drawBitmap(resized, padLeft.toFloat(), padTop.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
        if (resized !== frame) resized.recycle()

        return Letterbox(padded, padTop, padLeft, ratio)
    }

    private fun toChwTensor(bitmap: Bitmap, bgr: Boolean): OnnxTensor {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val channelSize = w * h
        val data = FloatArray(3 * channelSize)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            data[i] = if (bgr) b else r
            data[channelSize + i] = g
            data[2 * channelSize + i] = if (bgr) r else b
        }
        bitmap.recycle()

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    // ------------------------------------------------------------------
    // YOLOv5 postprocessing
    // ------------------------------------------------------------------

    @Suppress("UNCHECKED_CAST")
    private fun processYolov5(result: OrtSession.Result, targetClass: Int, box: Letterbox): List<HeadBox> {
        val raw = (result[0].value as Array<Array<FloatArray>>)[0] // [N, 7]

        val candidates = mutableListOf<Candidate>()
        for (row in raw) {
            val objectness = row[4]
            if (objectness <= confidenceThreshold) continue

            val classScore = row[5 + targetClass] * objectness
            if (classScore <= confidenceThreshold) continue

            val xc = row[0]; val yc = row[1]; val w = row[2]; val h = row[3]
            val x1 = (xc - w / 2f).coerceAtLeast(0f)
            val y1 = (yc - h / 2f).coerceAtLeast(0f)
            val x2 = (xc + w / 2f).coerceAtLeast(0f)
            val y2 = (yc + h / 2f).coerceAtLeast(0f)
            candidates += Candidate(x1, y1, x2, y2, classScore)
        }

        return nms(candidates.map { floatArrayOf(it.x1, it.y1, it.x2, it.y2) }, candidates.map { it.score })
            .map { idx ->
                val c = candidates[idx]
                unproject(c.x1, c.y1, c.x2, c.y2, c.score, box)
            }
    }

    // ------------------------------------------------------------------
    // YOLOv8 postprocessing (channels-first, no objectness)
    // ------------------------------------------------------------------

    @Suppress("UNCHECKED_CAST")
    private fun processYolov8(result: OrtSession.Result, targetClass: Int, box: Letterbox): List<HeadBox> {
        val raw = (result[0].value as Array<Array<FloatArray>>)[0] // [4 + nc, N]
        val classScores = raw[4 + targetClass]

        val candidates = mutableListOf<Candidate>()
        for (i in classScores.indices) {
            val score = classScores[i]
            if (score <= confidenceThreshold) continue

            val xc = raw[0][i]; val yc = raw[1][i]; val w = raw[2][i]; val h = raw[3][i]
            candidates += Candidate(xc - w / 2f, yc - h / 2f, xc + w / 2f, yc + h / 2f, score)
        }

        return nms(candidates.map { floatArrayOf(it.x1, it.y1, it.x2, it.y2) }, candidates.map { it.score })
            .map { idx ->
                val c = candidates[idx]
                unproject(c.x1, c.y1, c.x2, c.y2, c.score, box)
            }
    }

    // ------------------------------------------------------------------
    // YOLOv11 postprocessing
    // ------------------------------------------------------------------

    @Suppress("UNCHECKED_CAST")
    private fun processYolov11(result: OrtSession.Result, targetClass: Int, box: Letterbox): List<HeadBox> {
        val boxes = (result[0].value as Array<Array<FloatArray>>)[0] // [N, 4] as (y1, x1, y2, x2)
        val scores = (result[1].value as Array<FloatArray>)[0]      // [N]
        val classes = (result[2].value as Array<FloatArray>)[0]     // [N]

        val candidates = mutableListOf<Candidate>()
        for (i in boxes.indices) {
            if (classes[i].toInt() != targetClass) continue
            if (scores[i] <= confidenceThreshold) continue

            val y1 = boxes[i][0]; val x1 = boxes[i][1]
            val y2 = boxes[i][2]; val x2 = boxes[i][3]
            candidates += Candidate(
                x1.coerceAtLeast(0f), y1.coerceAtLeast(0f),
                x2.coerceAtLeast(0f), y2.coerceAtLeast(0f),
                scores[i]
            )
        }

        return nms(candidates.map { floatArrayOf(it.x1, it.y1, it.x2, it.y2) }, candidates.map { it.score })
            .map { idx ->
                val c = candidates[idx]
                unproject(c.x1, c.y1, c.x2, c.y2, c.score, box)
            }
    }

    private fun unproject(x1: Float, y1: Float, x2: Float, y2: Float, score: Float, box: Letterbox): HeadBox {
        return HeadBox(
            x1 = (x1 - box.padLeft) / box.scaleRatio,
            y1 = (y1 - box.padTop) / box.scaleRatio,
            x2 = (x2 - box.padLeft) / box.scaleRatio,
            y2 = (y2 - box.padTop) / box.scaleRatio,
            confidence = score
        )
    }

    // ------------------------------------------------------------------
    // Class-agnostic greedy NMS (replaces cv2.dnn.NMSBoxes). Boxes passed in
    // are already filtered to a single target class, matching the Python usage.
    // ------------------------------------------------------------------

    private fun nms(boxes: List<FloatArray>, scores: List<Float>): List<Int> {
        val order = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        val kept = mutableListOf<Int>()

        while (order.isNotEmpty()) {
            val current = order.removeAt(0)
            kept += current
            order.removeAll { iou(boxes[current], boxes[it]) > nmsThreshold }
        }
        return kept
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = maxOf(a[0], b[0])
        val y1 = maxOf(a[1], b[1])
        val x2 = minOf(a[2], b[2])
        val y2 = minOf(a[3], b[3])

        val interW = (x2 - x1).coerceAtLeast(0f)
        val interH = (y2 - y1).coerceAtLeast(0f)
        val inter = interW * interH

        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        val union = areaA + areaB - inter

        return if (union <= 0f) 0f else inter / union
    }

    companion object {
        private val YOLOV8_PAD_COLOR = Color.rgb(114, 114, 114)

        fun fromAssets(context: Context, assetPath: String, confidenceThreshold: Float = 0.5f, nmsThreshold: Float = 0.45f): YoloDetector {
            val bytes = context.assets.open(assetPath).use { it.readBytes() }
            return YoloDetector(OrtEnvironment.getEnvironment(), bytes, confidenceThreshold, nmsThreshold)
        }
    }
}
