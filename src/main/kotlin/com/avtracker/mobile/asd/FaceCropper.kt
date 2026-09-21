package com.avtracker.mobile.asd

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.avtracker.mobile.tracking.Roi

/** Builds the per-face pixel data the ASD backends consume from a camera frame. */
object FaceCropper {

    /**
     * Port of LightASDDetector._extract_crop: a square around the face centre with side max(w, h),
     * grayscale, resized to 224 and centre-cropped to 112. Resizing 224 and keeping the middle 112 is
     * the same as resizing the central half of the square straight to 112, which is what this does.
     *
     * Pixels outside the frame are filled like cv2.copyMakeBorder(value=110) does in Python: an int
     * value only sets the blue channel, so the padding is dark blue (gray ~13), not gray 110.
     */
    fun asdCrop(frame: Bitmap, box: Roi): ByteArray? {
        val h = box.y2 - box.y1
        val w = box.x2 - box.x1
        if (h <= 0 || w <= 0) return null

        val cs = maxOf(h, w) / 2
        if (cs < 2) return null
        val side = cs
        val left = (box.x1 + box.x2) / 2 - side / 2
        val top = (box.y1 + box.y2) / 2 - side / 2

        val srcLeft = left.coerceIn(0, frame.width)
        val srcTop = top.coerceIn(0, frame.height)
        val srcRight = (left + side).coerceIn(0, frame.width)
        val srcBottom = (top + side).coerceIn(0, frame.height)
        if (srcRight <= srcLeft || srcBottom <= srcTop) return null

        val scale = CROP.toFloat() / side
        val dst = Rect(
            ((srcLeft - left) * scale).toInt(), ((srcTop - top) * scale).toInt(),
            ((srcRight - left) * scale).toInt().coerceAtLeast(1), ((srcBottom - top) * scale).toInt().coerceAtLeast(1)
        )

        val out = Bitmap.createBitmap(CROP, CROP, Bitmap.Config.ARGB_8888)
        try {
            val canvas = Canvas(out)
            canvas.drawColor(Color.rgb(0, 0, 110))
            canvas.drawBitmap(frame, Rect(srcLeft, srcTop, srcRight, srcBottom), dst, Paint(Paint.FILTER_BITMAP_FLAG))
            return toGray(out).data
        } finally {
            out.recycle()
        }
    }

    /** Grayscale pixels of the face box (clamped to the frame), for the pixel-difference backend. */
    fun grayFace(frame: Bitmap, box: Roi): GrayImage? {
        val x1 = box.x1.coerceIn(0, frame.width)
        val y1 = box.y1.coerceIn(0, frame.height)
        val x2 = box.x2.coerceIn(0, frame.width)
        val y2 = box.y2.coerceIn(0, frame.height)
        if (x2 - x1 < 2 || y2 - y1 < 2) return null

        val w = x2 - x1
        val h = y2 - y1
        val pixels = IntArray(w * h)
        frame.getPixels(pixels, 0, w, x1, y1, w, h)
        return GrayImage(w, h, ByteArray(w * h) { luma(pixels[it]) })
    }

    private fun toGray(bitmap: Bitmap): GrayImage {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        return GrayImage(w, h, ByteArray(w * h) { luma(pixels[it]) })
    }

    /** OpenCV's BGR2GRAY weights. */
    private fun luma(argb: Int): Byte {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        return ((0.299f * r + 0.587f * g + 0.114f * b) + 0.5f).toInt().coerceIn(0, 255).toByte()
    }

    const val CROP = LightAsdModel.CROP
}
