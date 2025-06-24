package com.example.facerecognizer

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.graphics.Color

class OvalOverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private val paint = Paint().apply {
        color = Color.BLACK
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val ovalRect = RectF()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)

        val overlayWidth = w * 0.8f
        val overlayHeight = h * 0.5f

        val left = (w - overlayWidth) / 2f
        val top = (h - overlayHeight) / 2f
        val right = left + overlayWidth
        val bottom = top + overlayHeight
        ovalRect.set(left, top, right, bottom)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawOval(ovalRect, paint)
    }
}
