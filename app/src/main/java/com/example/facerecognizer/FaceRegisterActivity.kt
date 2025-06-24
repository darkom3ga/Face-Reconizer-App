package com.example.facerecognizer

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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
import com.example.facerecognizer.databinding.ActivityFaceRecognitionBinding
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

var userName = ""
var userId = ""

class FaceRegisterActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityFaceRecognitionBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var blazeFace: Interpreter
    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        userName = intent.getStringExtra("USER_NAME") ?: "Unknown"
        userId = intent.getStringExtra("USER_ID") ?: "0"

        viewBinding = ActivityFaceRecognitionBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

//        blazeFace = Interpreter(loadModelFile("blaze_face_short_range.tflite"))

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera(userName, userId)
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        viewBinding.btnSwitchCamera.setOnClickListener {
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }
            startCamera(userName, userId)
        }

        viewBinding.btnCaptureimg.setOnClickListener {
            captureImage()
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startCamera(userName: String, userID: String) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            viewBinding.userInfo.text = "Name: $userName, ID: $userID"

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder().build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureImage() {
        val imageCapture = imageCapture ?: return

        val photoFile = File.createTempFile("face_", ".jpg", cacheDir)
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(this@FaceRegisterActivity, "Capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    showConfirmationDialog(photoFile)
                }
            }
        )
    }

    private fun showConfirmationDialog(photoFile: File) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_capture_confirmation, null)
        val imageView = dialogView.findViewById<ImageView>(R.id.capturedImage)
        imageView.setImageURI(Uri.fromFile(photoFile))

        AlertDialog.Builder(this)
            .setTitle("Use this photo?")
            .setView(dialogView)
            .setPositiveButton("Yes") { _, _ ->
                val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
//                val faceBitmap = detectFaceWithBlaze(bitmap)
//                if (faceBitmap != null) {

                if (bitmap != null) {
                    // You can pass faceBitmap to anti-spoof or FaceNet here
                    Toast.makeText(this, "Face detected and cropped.", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "No face detected.", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("No") { dialog, _ -> dialog.dismiss() }
            .show()
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera(userName, userId)
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
