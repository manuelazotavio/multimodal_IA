package com.avtracker.mobile.ui

import android.Manifest
import android.app.AlertDialog
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.widget.Spinner
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.doOnLayout
import com.avtracker.mobile.R
import com.avtracker.mobile.app.FrameOverlay
import com.avtracker.mobile.app.MultimodalSystem
import com.avtracker.mobile.app.FaceOnlySession
import com.avtracker.mobile.app.TrackingSession
import com.avtracker.mobile.camera.CameraAnalyzer
import com.avtracker.mobile.fusion.HfInferenceClient
import com.avtracker.mobile.fusion.LlmClient
import com.avtracker.mobile.fusion.SpeakerNaming
import com.avtracker.mobile.fusion.TranscriptEntry
import com.avtracker.mobile.whisper.WhisperModel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusView: TextView
    private lateinit var transcriptView: TextView
    private lateinit var transcriptScroll: ScrollView
    private lateinit var finishButton: Button
    private lateinit var vocalIdButton: Button
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var loaderExecutor: ExecutorService
    private val player = PcmPlayer()

    @Volatile private var system: TrackingSession? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val transcriptLines = ArrayDeque<String>()
    private var finishing = false
    private var activeWhisperModel = WhisperModel.BASE

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            if (granted[Manifest.permission.CAMERA] == true) startCamera()
            else Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()

            if (system?.localAudioEnabled == false) {
                statusView.text = getString(R.string.status_vocalid_faces)
            } else if (granted[Manifest.permission.RECORD_AUDIO] == true) startAudio()
            else onVideoOnly()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        overlayView = findViewById(R.id.overlayView)
        overlayView.previewView = previewView
        statusView = findViewById(R.id.statusView)
        transcriptView = findViewById(R.id.transcriptView)
        transcriptScroll = findViewById(R.id.transcriptScroll)
        finishButton = findViewById(R.id.finishButton)
        vocalIdButton = findViewById(R.id.vocalIdButton)
        vocalIdButton.setOnClickListener { openVocalId() }
        cameraExecutor = Executors.newSingleThreadExecutor()
        loaderExecutor = Executors.newSingleThreadExecutor()

        finishButton.isEnabled = false
        finishButton.setOnClickListener { finishSession() }

        askSessionOptions { numSpeakers, llm, fullDiarization, vocalIdAudio, whisperModel ->
            activeWhisperModel = whisperModel
            vocalIdButton.visibility = if (vocalIdAudio) View.VISIBLE else View.GONE
            if (vocalIdAudio) transcriptView.text = getString(R.string.vocalid_audio_hint)
            // Loading ~200 MB of models takes seconds: keep the UI thread free.
            loaderExecutor.execute {
                try {
                    val loaded: TrackingSession = if (vocalIdAudio) FaceOnlySession.load(this, numSpeakers)
                    else MultimodalSystem.load(this, ::onTranscript, numSpeakers = numSpeakers, llm = llm, fullDiarization = fullDiarization, whisperModel = whisperModel)
                    system = loaded
                    runOnUiThread {
                        finishButton.isEnabled = true
                        val permissions = if (loaded.localAudioEnabled) arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
                        else arrayOf(Manifest.permission.CAMERA)
                        requestPermissionsLauncher.launch(permissions)
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "Model loading failed", t)
                    runOnUiThread { statusView.text = getString(R.string.status_error, t.message ?: t.javaClass.simpleName) }
                }
            }
        }
    }

    /**
     * run_multimodal_tracker.py: "How many people in the meeting? (Enter for no limit)", plus the switch for the LLM speaker
     * analysis (av-tracker's `use_ai_analysis`, off by default). Turning it on sends transcript text to the Hugging Face
     * inference API with the token typed here (kept only in this app's private preferences).
     */
    private fun askSessionOptions(onAnswer: (Int?, LlmClient?, Boolean, Boolean, WhisperModel) -> Unit) {
        val prefs = getSharedPreferences("av_tracker", MODE_PRIVATE)
        val speechModels = listOf(WhisperModel.BASE, WhisperModel.SMALL).filter { model ->
            assets.list(model.assetDirectory).orEmpty().toSet()
                .containsAll(listOf("encoder.onnx", "decoder.onnx", "meta.json", "tokens.txt"))
        }
        val speechModel = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                speechModels.map { getString(when (it) {
                    WhisperModel.BASE -> R.string.whisper_base
                    WhisperModel.SMALL -> R.string.whisper_small
                }) })
            val preferred = prefs.getString("whisper_model", "base")
            setSelection(speechModels.indexOfFirst { it.id == preferred }.coerceAtLeast(0))
        }
        val people = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            hint = getString(R.string.num_speakers_hint)
        }
        val diarizationSwitch = CheckBox(this).apply {
            text = getString(R.string.full_diarization)
            isChecked = prefs.getBoolean("full_diarization", false)
        }
        val aiSwitch = CheckBox(this).apply {
            text = getString(R.string.ai_analysis)
            isChecked = prefs.getBoolean("ai_analysis", false)
        }
        val token = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            hint = getString(R.string.hf_token_hint)
            setText(prefs.getString("hf_token", ""))
            visibility = if (aiSwitch.isChecked) View.VISIBLE else View.GONE
        }
        val vocalIdSwitch = CheckBox(this).apply {
            text = getString(R.string.use_vocalid_audio)
            isChecked = prefs.getBoolean("vocalid_audio", false)
        }
        fun updateAudioOptions() {
            val localAudio = !vocalIdSwitch.isChecked
            diarizationSwitch.isEnabled = localAudio
            aiSwitch.isEnabled = localAudio
            speechModel.isEnabled = localAudio
            token.visibility = if (localAudio && aiSwitch.isChecked) View.VISIBLE else View.GONE
        }
        vocalIdSwitch.setOnCheckedChangeListener { _, _ -> updateAudioOptions() }
        aiSwitch.setOnCheckedChangeListener { _, _ -> updateAudioOptions() }
        updateAudioOptions()
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 16, 48, 0)
            addView(people)
            addView(vocalIdSwitch)
            addView(TextView(this@MainActivity).apply { text = getString(R.string.transcription_model) })
            addView(speechModel)
            addView(diarizationSwitch)
            addView(aiSwitch)
            addView(token)
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.num_speakers_title)
            .setMessage(R.string.num_speakers_message)
            .setView(form)
            .setCancelable(false)
            .setPositiveButton(R.string.ok) { _, _ ->
                val useAi = !vocalIdSwitch.isChecked && aiSwitch.isChecked && token.text.toString().isNotBlank()
                val selectedModel = speechModels[speechModel.selectedItemPosition]
                prefs.edit().putString("whisper_model", selectedModel.id).putBoolean("vocalid_audio", vocalIdSwitch.isChecked).putBoolean("full_diarization", diarizationSwitch.isChecked).putBoolean("ai_analysis", aiSwitch.isChecked).putString("hf_token", token.text.toString().trim()).apply()
                onAnswer(
                    people.text.toString().trim().toIntOrNull()?.takeIf { it > 0 },
                    if (useAi) HfInferenceClient(token.text.toString().trim()) else null,
                    diarizationSwitch.isChecked,
                    vocalIdSwitch.isChecked,
                    selectedModel
                )
            }
            .show()
    }

    private fun openVocalId() {
        val launch = packageManager.getLaunchIntentForPackage("com.vocalid.jabra")
        if (launch == null) {
            Toast.makeText(this, R.string.vocalid_not_installed, Toast.LENGTH_LONG).show()
        } else {
            runCatching { startActivity(launch) }.onFailure {
                Log.e(TAG, "Could not open VocalID", it)
                Toast.makeText(this, R.string.vocalid_not_installed, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun startAudio() {
        val loaded = system ?: return
        runCatching { loaded.startAudio() }
            .onSuccess { statusView.text = getString(R.string.status_running_model, activeWhisperModel.id) }
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
            val text = if (entry.uncertain) getString(R.string.transcript_uncertain, entry.text) else entry.text
            transcriptLines.addLast("${entry.label}: $text")
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
            cameraProvider = provider

            previewView.doOnLayout {
                val viewPort = previewView.viewPort ?: return@doOnLayout
                // The Jabra's 4:3 analysis stream is letterboxed differently from its 16:9 preview.
                // Request the same buffer size and viewport for both before mapping detections.
                val resolution = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                    .build()
                val preview = Preview.Builder().setResolutionSelector(resolution).build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val analyzer = CameraAnalyzer<FrameOverlay>(
                    pipeline = loaded.facePipeline,
                    onFrame = { frame, result, timestampSec -> loaded.onFrame(frame, result, timestampSec) },
                    onResult = { result, overlay, w, h, processingMs, transform ->
                        runOnUiThread { overlayView.update(result, overlay, w, h, processingMs, transform) }
                    }
                )

                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolution)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, analyzer) }

                provider.unbindAll()
                val useCases = UseCaseGroup.Builder().setViewPort(viewPort)
                    .addUseCase(preview).addUseCase(analysis).build()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, useCases)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ---- end of session: stop, name the unidentified voices, save (run_multimodal_tracker.run) ----

    private fun finishSession() {
        val loaded = system ?: return
        if (finishing) return
        finishing = true
        finishButton.isEnabled = false
        statusView.text = getString(R.string.status_finishing)
        cameraProvider?.unbindAll() // the video loop ends first, as in the Python

        loaderExecutor.execute {
            loaded.stopAudio()
            val pending = loaded.pendingSpeakers()
            runOnUiThread { nameNext(loaded, pending.iterator(), pending.size, 1) }
        }
    }

    /** One dialog per anonymous speaker: hear up to 10 s of the voice, then type a name (empty = skip). */
    private fun nameNext(loaded: TrackingSession, queue: Iterator<SpeakerNaming.PendingSpeaker>, total: Int, index: Int) {
        if (!queue.hasNext()) {
            saveAndReport(loaded)
            return
        }
        val pending = queue.next()
        val preview = loaded.previewAudio(pending)

        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            hint = getString(R.string.speaker_name_hint)
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle(getString(R.string.speaker_dialog_title, pending.genericName, pending.seconds.toInt(), index, total))
            .setMessage(R.string.speaker_dialog_message)
            .setView(input)
            .setCancelable(false)
            .setPositiveButton(R.string.save, null)
            .setNeutralButton(R.string.replay, null)
            .setNegativeButton(R.string.skip) { _, _ ->
                player.stop()
                nameNext(loaded, queue, total, index + 1)
            }
            .create()

        dialog.setOnShowListener {
            player.play(preview)
            // Replay keeps the dialog open, so it is wired after show().
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener { player.play(preview) }
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val name = input.text.toString().trim()
                player.stop()
                dialog.dismiss()
                if (name.isEmpty()) {
                    nameNext(loaded, queue, total, index + 1)
                } else {
                    statusView.text = getString(R.string.status_saving_voice, name)
                    loaderExecutor.execute {
                        runCatching { loaded.enrollSpeaker(pending, name) }.onFailure { Log.e(TAG, "Enrolment failed", it) }
                        runOnUiThread { nameNext(loaded, queue, total, index + 1) }
                    }
                }
            }
        }
        dialog.show()
    }

    private fun saveAndReport(loaded: TrackingSession) {
        statusView.text = getString(R.string.status_saving)
        loaderExecutor.execute {
            val message = runCatching { loaded.saveSession() }
                .map { getString(R.string.status_saved, it.transcript.parentFile?.absolutePath ?: it.transcript.absolutePath) }
                .getOrElse {
                    Log.e(TAG, "Saving the session failed", it)
                    getString(R.string.status_save_failed, it.message ?: it.javaClass.simpleName)
                }
            runOnUiThread { statusView.text = message }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        player.stop()
        cameraExecutor.shutdown()
        loaderExecutor.shutdown()
        val running = system
        if (running != null && !finishing && running.transcript.isNotEmpty()) {
            // Closed without finishing: keep the session like the headless Python mode (no naming prompts).
            Thread {
                runCatching { running.stopAudio(); running.saveSession() }
                running.close()
            }.start()
        } else {
            running?.close()
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val MAX_TRANSCRIPT_LINES = 6
    }
}
