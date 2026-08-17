package com.example.signlanguagedetector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import java.util.Locale

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val clsName: String
)

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var mirrorHorizontally = false
    private var sourceWidth = 1
    private var sourceHeight = 1
    private var fillCenter = true

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 8f
        style = Paint.Style.STROKE
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        style = Paint.Style.FILL
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val srcW = sourceWidth.toFloat().coerceAtLeast(1f)
        val srcH = sourceHeight.toFloat().coerceAtLeast(1f)
        val viewW = width.toFloat().coerceAtLeast(1f)
        val viewH = height.toFloat().coerceAtLeast(1f)

        val scale = if (fillCenter) {
            maxOf(viewW / srcW, viewH / srcH)
        } else {
            minOf(viewW / srcW, viewH / srcH)
        }
        val contentW = srcW * scale
        val contentH = srcH * scale
        val offsetX = (viewW - contentW) / 2f
        val offsetY = (viewH - contentH) / 2f

        for (box in results) {
            val normalizedLeft = if (mirrorHorizontally) 1f - box.x2 else box.x1
            val normalizedRight = if (mirrorHorizontally) 1f - box.x1 else box.x2

            val left = offsetX + (normalizedLeft * contentW)
            val top = offsetY + (box.y1 * contentH)
            val right = offsetX + (normalizedRight * contentW)
            val bottom = offsetY + (box.y2 * contentH)

            canvas.drawRect(left, top, right, bottom, boxPaint)
            val label = "${box.clsName.uppercase(Locale.ROOT)} ${(box.cnf * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)

            canvas.drawRect(left, top - 50f, left + textWidth + 20f, top, textBackgroundPaint)
            canvas.drawText(label, left + 10f, top - 10f, textPaint)
        }
    }

    fun setResults(
        newResults: List<BoundingBox>,
        mirrored: Boolean = false,
        frameWidth: Int = sourceWidth,
        frameHeight: Int = sourceHeight,
        previewFillCenter: Boolean = true
    ) {
        results = newResults
        mirrorHorizontally = mirrored
        sourceWidth = frameWidth.coerceAtLeast(1)
        sourceHeight = frameHeight.coerceAtLeast(1)
        fillCenter = previewFillCenter
        postInvalidate()
    }
}