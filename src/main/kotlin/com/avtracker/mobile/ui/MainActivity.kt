package com.avtracker.mobile.ui

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.avtracker.mobile.R
import com.avtracker.mobile.app.FrameOverlay
import com.avtracker.mobile.app.MultimodalSystem
import com.avtracker.mobile.camera.CameraAnalyzer
import com.avtracker.mobile.fusion.TranscriptEntry
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusView: TextView
    private lateinit var transcriptView: TextView
    private lateinit var transcriptScroll: ScrollView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var loaderExecutor: ExecutorService

    @Volatile private var system: MultimodalSystem? = null
    private val transcriptLines = ArrayDeque<String>()

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            if (granted[Manifest.permission.CAMERA] == true) startCamera()
            else Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()

            if (granted[Manifest.permission.RECORD_AUDIO] == true) startAudio()
            else onVideoOnly()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusView = findViewById(R.id.statusView)
        transcriptView = findViewById(R.id.transcriptView)
        transcriptScroll = findViewById(R.id.transcriptScroll)
        cameraExecutor = Executors.newSingleThreadExecutor()
        loaderExecutor = Executors.newSingleThreadExecutor()

        // Loading ~200 MB of models takes seconds: keep the UI thread free.
        loaderExecutor.execute {
            try {
                val loaded = MultimodalSystem.load(this, ::onTranscript)
                system = loaded
                runOnUiThread { requestPermissionsLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)) }
            } catch (t: Throwable) {
                Log.e(TAG, "Model loading failed", t)
                runOnUiThread { statusView.text = getString(R.string.status_error, t.message ?: t.javaClass.simpleName) }
            }
        }
    }

    private fun startAudio() {
        val loaded = system ?: return
        runCatching { loaded.startAudio() }
            .onSuccess { statusView.text = getString(R.string.status_running) }
            .onFailure {
                Log.e(TAG, "Audio start failed", it)
                onVideoOnly()
            }
    }

    private fun onVideoOnly() {
        statusView.text = getString(R.string.status_video_only)
        Toast.makeText(this, R.string.microphone_permission_required, Toast.LENGTH_LONG).show()
    }

    private fun onTranscript(entry: TranscriptEntry) {
        runOnUiThread {
            transcriptLines.addLast("${entry.label}: ${entry.text}")
            while (transcriptLines.size > MAX_TRANSCRIPT_LINES) transcriptLines.removeFirst()
            transcriptView.text = transcriptLines.joinToString("\n")
            transcriptScroll.post { transcriptScroll.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun startCamera() {
        val loaded = system ?: return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analyzer = CameraAnalyzer<FrameOverlay>(
                pipeline = loaded.facePipeline,
                onFrame = { frame, result, timestampSec -> loaded.onFrame(frame, result, timestampSec) },
                onResult = { result, overlay, w, h, processingMs ->
                    runOnUiThread { overlayView.update(result, overlay, w, h, processingMs) }
                }
            )

            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, analyzer) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        loaderExecutor.shutdown()
        system?.close()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val MAX_TRANSCRIPT_LINES = 6
    }
}
