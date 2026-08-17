package com.example.signlanguagedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.roundToInt

class YoloDetector(
    private val context: Context,
    private val modelPath: String = "best_float32.tflite",
    private val labelPath: String = "lables.txt",
    private val detectorListener: DetectorListener
) {
    private companion object {
        private const val CONFIDENCE_THRESHOLD = 0.12f
        private const val CLASS_ONLY_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.45f
        private val DEFAULT_CLASS_NAMES = listOf(
            "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "k",
            "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y"
        )
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    private var tensorWidth = 640
    private var tensorHeight = 640

    init {
        setup()
    }

    private fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)

        val inputTensor = interpreter?.getInputTensor(0)
        val inputShape = inputTensor?.shape()
        if (inputShape != null) {
            tensorWidth = inputShape[2]
            tensorHeight = inputShape[1]
        }

        loadLabels()
        if (labels.isEmpty()) {
            labels.addAll(DEFAULT_CLASS_NAMES)
        }
    }

    private fun loadLabels() {
        val inputStream = context.assets.open(labelPath)
        val reader = BufferedReader(InputStreamReader(inputStream))
        reader.forEachLine { line ->
            if (line.isNotBlank()) {
                val raw = line.trim()
                // Keep numeric labels like "1", "2" intact. Only strip numeric prefixes like "0 person".
                val cleaned = Regex("^\\d+[:\\.\\-\\s]+(.+)$").find(raw)?.groupValues?.get(1)?.trim() ?: raw
                labels.add(cleaned)
            }
        }
        reader.close()
    }

    fun detect(bitmap: Bitmap) {
        if (interpreter == null) return

        val startTime = System.currentTimeMillis()

        val (letterboxedBitmap, letterboxMeta) = letterboxToTensor(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0f, 255f))
            .build()

        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(letterboxedBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val outputTensor = interpreter?.getOutputTensor(0)
        val outputShape = outputTensor?.shape() ?: return

        val output = Array(1) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
        interpreter?.run(tensorImage.buffer, output)

        val bestBoxes = extractBestBoxes(output[0], outputShape, letterboxMeta)
        val inferenceTime = System.currentTimeMillis() - startTime

        if (bestBoxes.isEmpty()) {
            detectorListener.onEmptyDetect()
        } else {
            detectorListener.onDetect(bestBoxes, inferenceTime)
        }
    }

    private fun extractBestBoxes(
        array: Array<FloatArray>,
        outputShape: IntArray,
        letterboxMeta: LetterboxMeta
    ): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        if (outputShape.size < 3) return emptyList()

        val dim1 = outputShape[1]
        val dim2 = outputShape[2]
        val featuresFirst = dim1 < dim2
        val featureCount = if (featuresFirst) dim1 else dim2
        val predictionCount = if (featuresFirst) dim2 else dim1

        if (featureCount < 5) return emptyList()

        val hasObjectness = when {
            featureCount - 4 == labels.size -> false
            featureCount - 5 == labels.size -> true
            else -> false // Default for YOLOv8/YOLO11 TFLite exports
        }
        val classStart = if (hasObjectness) 5 else 4
        val modelClassCount = featureCount - classStart
        val activeLabels = resolveLabels(modelClassCount)
        val classCount = activeLabels.size

        if (classCount <= 0) return emptyList()

        for (c in 0 until predictionCount) {
            var maxConf = -1.0f
            var maxClass = -1

            for (p in 0 until classCount) {
                val clsIndex = classStart + p
                val clsRaw = if (featuresFirst) array[clsIndex][c] else array[c][clsIndex]
                val clsConf = toProbability(clsRaw)
                if (clsConf > maxConf) {
                    maxConf = clsConf
                    maxClass = p
                }
            }

            if (maxClass == -1) continue

            val objectness = if (hasObjectness) {
                val objectnessRaw = if (featuresFirst) array[4][c] else array[c][4]
                toProbability(objectnessRaw)
            } else {
                1.0f
            }

            val confidence = maxConf * objectness
            if (confidence <= CONFIDENCE_THRESHOLD && maxConf <= CLASS_ONLY_THRESHOLD) continue

            val a = if (featuresFirst) array[0][c] else array[c][0]
            val b = if (featuresFirst) array[1][c] else array[c][1]
            val c3 = if (featuresFirst) array[2][c] else array[c][2]
            val d = if (featuresFirst) array[3][c] else array[c][3]

            // Try both xywh and xyxy and convert to original-image normalized coords
            val maybeXywh = toBoxFromXywh(a, b, c3, d)
            val maybeXyxy = toBoxFromXyxy(a, b, c3, d)

            val selectedRaw = when {
                maybeXywh != null && maybeXyxy != null -> {
                    val areaXywh = (maybeXywh.third - maybeXywh.first) * (maybeXywh.fourth - maybeXywh.second)
                    val areaXyxy = (maybeXyxy.third - maybeXyxy.first) * (maybeXyxy.fourth - maybeXyxy.second)
                    if (areaXywh >= areaXyxy) maybeXywh else maybeXyxy
                }
                maybeXywh != null -> maybeXywh
                else -> maybeXyxy
            } ?: continue

            val (x1n, y1n) = mapTensorNormToSourceNorm(selectedRaw.first, selectedRaw.second, letterboxMeta)
            val (x2n, y2n) = mapTensorNormToSourceNorm(selectedRaw.third, selectedRaw.fourth, letterboxMeta)

            if (x2n <= x1n || y2n <= y1n) continue

            val cx = (x1n + x2n) / 2f
            val cy = (y1n + y2n) / 2f
            val w = x2n - x1n
            val h = y2n - y1n

            boundingBoxes.add(
                BoundingBox(
                    x1 = x1n,
                    y1 = y1n,
                    x2 = x2n,
                    y2 = y2n,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = confidence.coerceAtLeast(maxConf),
                    cls = maxClass,
                    clsName = activeLabels[maxClass]
                )
            )
        }

        return applyNMS(boundingBoxes)
    }

    private fun letterboxToTensor(source: Bitmap): Pair<Bitmap, LetterboxMeta> {
        val srcW = source.width
        val srcH = source.height
        val dstW = tensorWidth
        val dstH = tensorHeight

        val scale = minOf(dstW.toFloat() / srcW.toFloat(), dstH.toFloat() / srcH.toFloat())
        val resizedW = (srcW * scale).roundToInt().coerceAtLeast(1)
        val resizedH = (srcH * scale).roundToInt().coerceAtLeast(1)
        val padX = (dstW - resizedW) / 2f
        val padY = (dstH - resizedH) / 2f

        val letterboxed = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.BLACK)

        val dstRect = RectF(padX, padY, padX + resizedW, padY + resizedH)
        canvas.drawBitmap(source, null, dstRect, null)

        return Pair(
            letterboxed,
            LetterboxMeta(
                scale = scale,
                padX = padX,
                padY = padY,
                srcW = srcW,
                srcH = srcH,
                dstW = dstW,
                dstH = dstH
            )
        )
    }

    private fun mapTensorNormToSourceNorm(
        xNorm: Float,
        yNorm: Float,
        letterboxMeta: LetterboxMeta
    ): Pair<Float, Float> {
        val xTensor = xNorm.coerceIn(0f, 1f) * letterboxMeta.dstW
        val yTensor = yNorm.coerceIn(0f, 1f) * letterboxMeta.dstH

        val xSrc = (xTensor - letterboxMeta.padX) / letterboxMeta.scale
        val ySrc = (yTensor - letterboxMeta.padY) / letterboxMeta.scale

        val xSrcNorm = (xSrc / letterboxMeta.srcW.toFloat()).coerceIn(0f, 1f)
        val ySrcNorm = (ySrc / letterboxMeta.srcH.toFloat()).coerceIn(0f, 1f)
        return Pair(xSrcNorm, ySrcNorm)
    }

    private fun toProbability(value: Float): Float {
        return if (value < 0f || value > 1f) {
            (1f / (1f + kotlin.math.exp(-value)))
        } else {
            value
        }
    }

    private fun resolveLabels(modelClassCount: Int): List<String> {
        if (modelClassCount <= 0) return emptyList()
        return when {
            labels.size == modelClassCount -> labels
            DEFAULT_CLASS_NAMES.size == modelClassCount -> DEFAULT_CLASS_NAMES
            labels.size > modelClassCount -> labels.take(modelClassCount)
            labels.isNotEmpty() -> {
                val generated = (labels.size until modelClassCount).map { idx -> "class_$idx" }
                labels + generated
            }
            DEFAULT_CLASS_NAMES.size > modelClassCount -> DEFAULT_CLASS_NAMES.take(modelClassCount)
            else -> (0 until modelClassCount).map { idx -> "class_$idx" }
        }
    }

    private fun toBoxFromXywh(cxRaw: Float, cyRaw: Float, wRaw: Float, hRaw: Float): Box? {
        var cx = cxRaw
        var cy = cyRaw
        var w = wRaw
        var h = hRaw

        if (cx > 1f || cy > 1f || w > 1f || h > 1f) {
            cx /= tensorWidth.toFloat()
            w /= tensorWidth.toFloat()
            cy /= tensorHeight.toFloat()
            h /= tensorHeight.toFloat()
        }

        val x1 = cx - (w / 2f)
        val y1 = cy - (h / 2f)
        val x2 = cx + (w / 2f)
        val y2 = cy + (h / 2f)
        return sanitizeBox(x1, y1, x2, y2)
    }

    private fun toBoxFromXyxy(x1Raw: Float, y1Raw: Float, x2Raw: Float, y2Raw: Float): Box? {
        var x1 = x1Raw
        var y1 = y1Raw
        var x2 = x2Raw
        var y2 = y2Raw

        if (x1 > 1f || y1 > 1f || x2 > 1f || y2 > 1f) {
            x1 /= tensorWidth.toFloat()
            x2 /= tensorWidth.toFloat()
            y1 /= tensorHeight.toFloat()
            y2 /= tensorHeight.toFloat()
        }
        return sanitizeBox(x1, y1, x2, y2)
    }

    private fun sanitizeBox(x1Raw: Float, y1Raw: Float, x2Raw: Float, y2Raw: Float): Box? {
        val x1 = x1Raw.coerceIn(0f, 1f)
        val y1 = y1Raw.coerceIn(0f, 1f)
        val x2 = x2Raw.coerceIn(0f, 1f)
        val y2 = y2Raw.coerceIn(0f, 1f)

        if (x2 <= x1 || y2 <= y1) return null
        if (x1 >= 1f || y1 >= 1f || x2 <= 0f || y2 <= 0f) return null
        return Box(x1, y1, x2, y2)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.removeAt(0)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou > IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)

        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    private data class Box(
        val first: Float,
        val second: Float,
        val third: Float,
        val fourth: Float
    )

    private data class LetterboxMeta(
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val srcW: Int,
        val srcH: Int,
        val dstW: Int,
        val dstH: Int
    )
}