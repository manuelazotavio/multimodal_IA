package com.avtracker.mobile.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.avtracker.mobile.app.FrameOverlay
import com.avtracker.mobile.pipeline.FrameResult
import java.util.Locale

/**
 * Draws tracking boxes, names and an FPS/status overlay on top of the camera
 * preview, replacing the Python cv2.rectangle / putText / imshow calls. Faces
 * whose mouth the active-speaker detector says is moving get a thick cyan box.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var result: FrameResult? = null
    private var overlay: FrameOverlay? = null
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var processingMs = 0L

    private var fps = 0f
    private var frameCount = 0
    private var lastFpsTimestamp = System.currentTimeMillis()

    private val boxPaintKnown = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val boxPaintUnknown = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val boxPaintSpeaking = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }
    private val labelBackgroundPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
    }

    fun update(result: FrameResult, overlay: FrameOverlay?, sourceWidth: Int, sourceHeight: Int, processingMs: Long) {
        this.result = result
        this.overlay = overlay
        this.sourceWidth = sourceWidth
        this.sourceHeight = sourceHeight
        this.processingMs = processingMs

        frameCount++
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTimestamp
        if (elapsed >= 1000) {
            fps = frameCount * 1000f / elapsed
            frameCount = 0
            lastFpsTimestamp = now
        }

        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val frameResult = result ?: return
        if (sourceWidth <= 0 || sourceHeight <= 0) return

        val scaleX = width.toFloat() / sourceWidth
        val scaleY = height.toFloat() / sourceHeight

        for (person in frameResult.persons) {
            val fused = overlay?.labels?.get(person.trackId)
            val unknown = fused?.unknown ?: !person.identified
            val speaking = overlay?.speaking?.contains(person.trackId) == true
            val paint = when {
                speaking -> boxPaintSpeaking
                unknown -> boxPaintUnknown
                else -> boxPaintKnown
            }

            val left = person.bbox.x1 * scaleX
            val top = person.bbox.y1 * scaleY
            val right = person.bbox.x2 * scaleX
            val bottom = person.bbox.y2 * scaleY
            canvas.drawRect(left, top, right, bottom, paint)

            val label = fused?.text ?: person.displayName
            val textWidth = textPaint.measureText(label)
            val labelTop = (top - 44f).coerceAtLeast(0f)
            canvas.drawRect(left, labelTop, left + textWidth + 16f, labelTop + 44f, labelBackgroundPaint)
            canvas.drawText(label, left + 8f, labelTop + 32f, textPaint)
        }

        val info = "FPS: %.1f | %dms | Known: %d | Tracked: %d".format(
            Locale.ROOT, fps, processingMs, frameResult.knownDatabaseSize, frameResult.persons.size
        )
        canvas.drawRect(0f, 0f, textPaint.measureText(info) + 24f, 56f, labelBackgroundPaint)
        canvas.drawText(info, 12f, 40f, textPaint)
    }
}
