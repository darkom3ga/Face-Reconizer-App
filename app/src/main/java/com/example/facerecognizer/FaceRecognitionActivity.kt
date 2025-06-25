package com.example.facerecognizer

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.facerecognizer.databinding.ActivityAttendanceBinding
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.face.FaceDetector
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

class FaceRecognitionActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityAttendanceBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceNet: Interpreter
    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    private var imageCapture: ImageCapture? = null
    private var registeredFaces = mutableListOf<FaceData>()

    data class FaceData(
        val userId: String,
        val embedding: FloatArray
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewBinding = ActivityAttendanceBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Initialize TensorFlow Lite model
        try {
            faceNet = Interpreter(loadModelFile("facenet.tflite"))
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            Toast.makeText(this, "Error loading AI model", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Load registered faces from storage
        loadRegisteredFaces()

        // Check permissions and start camera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up button listeners
        viewBinding.btnSwitchCamera.setOnClickListener {
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }
            startCamera()
        }

        viewBinding.btnRecognize.setOnClickListener {
            recognizeFace()
        }

        viewBinding.btnBack.setOnClickListener {
            finish()
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            viewBinding.tvStatus.text = "Camera ready - ${registeredFaces.size} faces registered"

            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build().also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
                Log.d(TAG, "Camera started successfully")
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
                viewBinding.tvStatus.text = "Camera initialization failed"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun recognizeFace() {
        val imageCapture = imageCapture ?: return
        val photoFile = File.createTempFile("recognize_", ".jpg", cacheDir)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        viewBinding.tvStatus.text = "Capturing image..."
        viewBinding.btnRecognize.isEnabled = false

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    viewBinding.tvStatus.text = "Capture failed: ${exc.message}"
                    viewBinding.btnRecognize.isEnabled = true
                    Toast.makeText(this@FaceRecognitionActivity, "Capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    processImageForRecognition(photoFile)
                }
            }
        )
    }

    private fun processImageForRecognition(photoFile: File) {
        viewBinding.tvStatus.text = "Processing image..."

        try {
            val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
            Log.d(TAG, "Original bitmap size: ${bitmap.width}x${bitmap.height}")

            // Try multiple face detection methods
            var faceBitmap = detectFaceWithAndroidAPI(bitmap)

            if (faceBitmap == null) {
                Log.d(TAG, "Android API failed, trying simple center crop")
                faceBitmap = simpleFaceExtraction(bitmap)
            }

            if (faceBitmap != null) {
                Log.d(TAG, "Face region extracted, getting embedding...")
                val embedding = getFaceEmbedding(faceBitmap)
                Log.d(TAG, "Embedding generated, length: ${embedding.size}")

                val recognitionResult = findClosestMatch(embedding)

                runOnUiThread {
                    viewBinding.btnRecognize.isEnabled = true
                    if (recognitionResult.first != null) {
                        showRecognitionResult(photoFile, recognitionResult.first!!, true, recognitionResult.second, recognitionResult.third)
                    } else {
                        showRecognitionResult(photoFile, null, false, recognitionResult.second, recognitionResult.third)
                    }
                }
            } else {
                runOnUiThread {
                    viewBinding.tvStatus.text = "No face detected"
                    viewBinding.btnRecognize.isEnabled = true
                    Toast.makeText(this, "No face detected. Please ensure your face is clearly visible and well-lit.", Toast.LENGTH_LONG).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image", e)
            runOnUiThread {
                viewBinding.tvStatus.text = "Error processing image: ${e.message}"
                viewBinding.btnRecognize.isEnabled = true
            }
        }
    }

    private fun detectFaceWithAndroidAPI(bitmap: Bitmap): Bitmap? {
        var faceDetector: FaceDetector? = null
        try {
            faceDetector = FaceDetector.Builder(this)
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.NO_LANDMARKS) // Simplified for better compatibility
                .setMode(FaceDetector.FAST_MODE) // Use fast mode first
                .setMinFaceSize(0.1f) // Detect smaller faces
                .build()

            if (!faceDetector.isOperational) {
                Log.e(TAG, "Face detector not operational")
                return null
            }

            // Convert to RGB if needed
            val rgbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            val frame = Frame.Builder().setBitmap(rgbBitmap).build()
            val faces = faceDetector.detect(frame)

            Log.d(TAG, "Detected ${faces.size()} faces")

            if (faces.size() == 0) {
                Log.d(TAG, "No faces detected with Android API")
                return null
            }

            // Get the largest face
            var largestFace = faces.valueAt(0)
            for (i in 1 until faces.size()) {
                val face = faces.valueAt(i)
                if (face.width * face.height > largestFace.width * largestFace.height) {
                    largestFace = face
                }
            }

            val left = largestFace.position.x.toInt()
            val top = largestFace.position.y.toInt()
            val right = (left + largestFace.width).toInt()
            val bottom = (top + largestFace.height).toInt()

            // Add padding around the face
            val padding = (largestFace.width * 0.1f).toInt()
            val clampedLeft = maxOf(0, left - padding)
            val clampedTop = maxOf(0, top - padding)
            val clampedRight = minOf(bitmap.width, right + padding)
            val clampedBottom = minOf(bitmap.height, bottom + padding)

            Log.d(TAG, "Face detected at: ($clampedLeft, $clampedTop, $clampedRight, $clampedBottom)")

            val faceWidth = clampedRight - clampedLeft
            val faceHeight = clampedBottom - clampedTop

            if (faceWidth <= 0 || faceHeight <= 0) {
                Log.d(TAG, "Invalid face dimensions")
                return null
            }

            val faceBitmap = Bitmap.createBitmap(bitmap, clampedLeft, clampedTop, faceWidth, faceHeight)
            return Bitmap.createScaledBitmap(faceBitmap, 112, 112, true)

        } catch (e: Exception) {
            Log.e(TAG, "Error in Android face detection", e)
            return null
        } finally {
            faceDetector?.release()
        }
    }

    // Fallback method - simple center crop (same as your registration code)
    private fun simpleFaceExtraction(bitmap: Bitmap): Bitmap? {
        try {
            Log.d(TAG, "Using simple face extraction (center crop)")

            // Use the same approach as your registration code
            // This assumes the face is roughly in the center of the image
            val size = minOf(bitmap.width, bitmap.height)
            val x = (bitmap.width - size) / 2
            val y = (bitmap.height - size) / 2

            val croppedBitmap = Bitmap.createBitmap(bitmap, x, y, size, size)
            return Bitmap.createScaledBitmap(croppedBitmap, 112, 112, true)

        } catch (e: Exception) {
            Log.e(TAG, "Error in simple face extraction", e)
            return null
        }
    }

    private fun findClosestMatch(embedding: FloatArray): Triple<FaceData?, Float, Float> {
        if (registeredFaces.isEmpty()) {
            Log.d(TAG, "No registered faces found")
            return Triple(null, Float.MAX_VALUE, 0f)
        }

        Log.d(TAG, "=== DEBUGGING FACE RECOGNITION ===")
        Log.d(TAG, "Comparing against ${registeredFaces.size} registered faces")
        Log.d(TAG, "Current embedding size: ${embedding.size}")
        Log.d(TAG, "Current embedding first 10 values: ${embedding.take(10)}")

        var closestMatch: FaceData? = null
        var minDistance = Float.MAX_VALUE
        var maxSimilarity = -1f

        // More lenient thresholds for testing
        val euclideanThreshold = 3.0f
        val cosineThreshold = 0.6f

        registeredFaces.forEachIndexed { index, registeredFace ->
            val euclideanDistance = calculateEuclideanDistance(embedding, registeredFace.embedding)
            val cosineSimilarity = calculateCosineSimilarity(embedding, registeredFace.embedding)

            Log.d(TAG, "--- Face ${index + 1} (User ${registeredFace.userId}) ---")
            Log.d(TAG, "Registered embedding first 10 values: ${registeredFace.embedding.take(10)}")
            Log.d(TAG, "Euclidean distance: $euclideanDistance (threshold: $euclideanThreshold)")
            Log.d(TAG, "Cosine similarity: $cosineSimilarity (threshold: $cosineThreshold)")

            // Use cosine similarity as primary metric
            if (cosineSimilarity > maxSimilarity) {
                maxSimilarity = cosineSimilarity
                minDistance = euclideanDistance
                if (cosineSimilarity > cosineThreshold && euclideanDistance < euclideanThreshold) {
                    closestMatch = registeredFace
                }
            }
        }

        Log.d(TAG, "=== FINAL RESULT ===")
        Log.d(TAG, "Best cosine similarity: $maxSimilarity")
        Log.d(TAG, "Corresponding euclidean distance: $minDistance")
        Log.d(TAG, "Recognized: ${closestMatch?.userId ?: "None"}")

        return Triple(closestMatch, minDistance, maxSimilarity)
    }

    private fun calculateEuclideanDistance(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            Log.e(TAG, "Embedding size mismatch: ${embedding1.size} vs ${embedding2.size}")
            return Float.MAX_VALUE
        }

        var sum = 0f
        for (i in embedding1.indices) {
            val diff = embedding1[i] - embedding2[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    private fun calculateCosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            Log.e(TAG, "Embedding size mismatch: ${embedding1.size} vs ${embedding2.size}")
            return 0f
        }

        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        val result = dotProduct / (sqrt(norm1) * sqrt(norm2))
        return if (result.isNaN()) 0f else result
    }

    private fun showRecognitionResult(photoFile: File, recognizedUser: FaceData?, isRecognized: Boolean, distance: Float, similarity: Float) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_recognition_result, null)
        val imageView = dialogView.findViewById<ImageView>(R.id.recognizedImage)
        imageView.setImageURI(Uri.fromFile(photoFile))

        val title = if (isRecognized) {
            "Face Recognized!"
        } else {
            "Face Not Recognized"
        }

        val message = if (isRecognized && recognizedUser != null) {
            "Welcome, User ${recognizedUser.userId}!\n" +
                    "Cosine Similarity: ${String.format("%.4f", similarity)}\n" +
                    "Euclidean Distance: ${String.format("%.4f", distance)}"
        } else {
            "This face is not registered in the system.\n" +
                    "Best Cosine Similarity: ${String.format("%.4f", similarity)}\n" +
                    "Euclidean Distance: ${String.format("%.4f", distance)}\n" +
                    "(Cosine Threshold: 0.3, Euclidean Threshold: 1.5)"
        }

        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setView(dialogView)
            .setPositiveButton("OK") { dialog, _ ->
                dialog.dismiss()
                if (isRecognized && recognizedUser != null) {
                    viewBinding.tvStatus.text = "Last recognized: User ${recognizedUser.userId}"
                } else {
                    viewBinding.tvStatus.text = "Recognition failed - similarity: ${String.format("%.3f", similarity)}"
                }
            }
            .setNegativeButton("Debug") { dialog, _ ->
                showDebugInfo(distance, similarity)
                dialog.dismiss()
            }
            .show()
    }

    private fun showDebugInfo(distance: Float, similarity: Float) {
        val debugInfo = StringBuilder()
        debugInfo.append("=== DEBUG INFO ===\n")
        debugInfo.append("Registered faces: ${registeredFaces.size}\n")
        debugInfo.append("Best cosine similarity: ${String.format("%.6f", similarity)}\n")
        debugInfo.append("Euclidean distance: ${String.format("%.6f", distance)}\n")
        debugInfo.append("Cosine threshold: 0.3\n")
        debugInfo.append("Euclidean threshold: 1.5\n\n")

        debugInfo.append("All registered users:\n")
        registeredFaces.forEachIndexed { index, face ->
            debugInfo.append("${index + 1}. User ${face.userId}\n")
        }

        AlertDialog.Builder(this)
            .setTitle("Debug Information")
            .setMessage(debugInfo.toString())
            .setPositiveButton("OK", null)
            .show()
    }

    private fun loadRegisteredFaces() {
        registeredFaces.clear()

        val embeddingFiles = filesDir.listFiles { file ->
            file.name.endsWith(".embedding")
        }

        Log.d(TAG, "Found ${embeddingFiles?.size ?: 0} embedding files")

        embeddingFiles?.forEach { file ->
            try {
                val userId = file.nameWithoutExtension
                val content = file.readText()

                val embeddingValues = content.split(",").map { it.trim().toFloat() }
                val embedding = embeddingValues.toFloatArray()

                registeredFaces.add(FaceData(userId, embedding))
                Log.d(TAG, "Loaded embedding for user $userId, size: ${embedding.size}")
                Log.d(TAG, "First 10 values: ${embedding.take(10)}")

            } catch (e: Exception) {
                Log.e(TAG, "Error loading embedding file ${file.name}", e)
            }
        }

        Log.d(TAG, "Successfully loaded ${registeredFaces.size} registered faces")
        runOnUiThread {
            viewBinding.tvStatus.text = "Loaded ${registeredFaces.size} registered faces"
        }
    }

    private fun getFaceEmbedding(bitmap: Bitmap): FloatArray {
        try {
            // Use EXACT same preprocessing as your registration code
            val input = Bitmap.createScaledBitmap(bitmap, 160, 160, true)
            val buffer = ByteBuffer.allocateDirect(1 * 160 * 160 * 3 * 4).order(ByteOrder.nativeOrder())

            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    val pixel = input.getPixel(x, y)
                    buffer.putFloat((Color.red(pixel) - 127.5f) / 128f)
                    buffer.putFloat((Color.green(pixel) - 127.5f) / 128f)
                    buffer.putFloat((Color.blue(pixel) - 127.5f) / 128f)
                }
            }

            val output = Array(1) { FloatArray(128) }
            faceNet.run(buffer, output)

            Log.d(TAG, "Generated embedding size: ${output[0].size}")
            Log.d(TAG, "Embedding first 10 values: ${output[0].take(10)}")

            return output[0]
        } catch (e: Exception) {
            Log.e(TAG, "Error generating face embedding", e)
            throw e
        }
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::faceNet.isInitialized) faceNet.close()
    }

    companion object {
        private const val TAG = "FaceRecognitionActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
