package com.example.signlanguagedetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.signlanguagedetector.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.Locale

class MainActivity : AppCompatActivity(), YoloDetector.DetectorListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var yoloDetector: YoloDetector
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    @Volatile
    private var lastFrameWidth = 1
    @Volatile
    private var lastFrameHeight = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        yoloDetector = YoloDetector(this, detectorListener = this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.viewFinder.scaleType = PreviewView.ScaleType.FILL_CENTER
        binding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()

        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        } catch (exc: Exception) {
            Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun switchCamera() {
        val provider = cameraProvider ?: return
        val newLensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }

        val newCameraSelector = CameraSelector.Builder()
            .requireLensFacing(newLensFacing)
            .build()

        try {
            if (provider.hasCamera(newCameraSelector)) {
                lensFacing = newLensFacing
                bindCameraUseCases()
            } else {
                Toast.makeText(this, "Selected camera is not available", Toast.LENGTH_SHORT).show()
            }
        } catch (exc: CameraInfoUnavailableException) {
            Toast.makeText(this, "Unable to access camera information", Toast.LENGTH_SHORT).show()
        }
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        lastFrameWidth = rotatedBitmap.width
        lastFrameHeight = rotatedBitmap.height

        yoloDetector.detect(rotatedBitmap)
        imageProxy.close()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.overlay.setResults(
                boundingBoxes,
                lensFacing == CameraSelector.LENS_FACING_FRONT,
                frameWidth = lastFrameWidth,
                frameHeight = lastFrameHeight,
                previewFillCenter = true
            )
            // Show the first detected class name in the big display at the bottom
            val topResult = boundingBoxes.firstOrNull()
            binding.detectedText.text = topResult?.clsName?.uppercase(Locale.ROOT) ?: ""
        }
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.setResults(
                emptyList(),
                mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT,
                frameWidth = lastFrameWidth,
                frameHeight = lastFrameHeight,
                previewFillCenter = true
            )
            binding.detectedText.text = ""
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
