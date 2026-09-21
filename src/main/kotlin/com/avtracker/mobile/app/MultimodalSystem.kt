package com.avtracker.mobile.app

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import com.avtracker.mobile.asd.ActiveSpeakerDetector
import com.avtracker.mobile.asd.FaceCropper
import com.avtracker.mobile.asd.FaceSample
import com.avtracker.mobile.asd.LightAsdDetector
import com.avtracker.mobile.asd.LightAsdModel
import com.avtracker.mobile.audio.AudioCapture
import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.detection.YoloDetector
import com.avtracker.mobile.diarization.Diarizer
import com.avtracker.mobile.embedding.EdgeFaceEmbedder
import com.avtracker.mobile.fusion.FaceIdentitySync
import com.avtracker.mobile.fusion.FaceLabel
import com.avtracker.mobile.fusion.FusionConfig
import com.avtracker.mobile.fusion.FusionEngine
import com.avtracker.mobile.fusion.IdentityRegistry
import com.avtracker.mobile.fusion.RegistryFaceBridge
import com.avtracker.mobile.fusion.TranscriptEntry
import com.avtracker.mobile.pipeline.FrameResult
import com.avtracker.mobile.pipeline.HeadTrackerPipeline
import com.avtracker.mobile.profile.VisualProfileDatabase
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.voice.VoiceEmbedder
import com.avtracker.mobile.voice.DirectoryVoiceStore
import com.avtracker.mobile.ner.SpacyNer
import com.avtracker.mobile.whisper.WhisperTranscriber
import java.util.Locale

/** What the overlay draws on top of one camera frame. */
class FrameOverlay(val labels: Map<Int, FaceLabel>, val speaking: Set<Int>)

/**
 * The whole multimodal system of av-tracker on the phone: the face pipeline (YOLO + EdgeFace), the audio pipeline
 * (mic -> diarization -> ECAPA voice ID -> Whisper) and the fusion that ties them together through a shared
 * [IdentityRegistry] and the active-speaker detector.
 */
class MultimodalSystem private constructor(
    val facePipeline: HeadTrackerPipeline,
    private val registry: IdentityRegistry,
    private val faceSync: FaceIdentitySync,
    private val asd: ActiveSpeakerDetector,
    private val useLightAsd: Boolean,
    private val timeline: AudioTimeline,
    private val fusion: FusionEngine,
    private val capture: AudioCapture,
    private val closeables: List<AutoCloseable>
) : AutoCloseable {

    val transcript: List<TranscriptEntry> get() = fusion.transcript

    /** Called from the camera analysis thread after the face pipeline has processed [frame]. */
    fun onFrame(frame: Bitmap, result: FrameResult, timestampSec: Double): FrameOverlay {
        val faces = FaceIdentitySync.fromSnapshots(result.persons)
        val current = faceSync.sync(faces)

        val samples = result.persons.map { p ->
            FaceSample(p.trackId, if (useLightAsd) FaceCropper.asdCrop(frame, p.bbox) else null, FaceCropper.grayFace(frame, p.bbox))
        }
        asd.update(samples, timestampSec)

        return FrameOverlay(
            labels = faceSync.labels(faces, current).associateBy { it.trackId },
            speaking = result.persons.map { it.trackId }.filter { asd.isSpeakingNow(it) }.toSet()
        )
    }

    /** Needs the RECORD_AUDIO permission. */
    fun startAudio() {
        capture.start()
        fusion.start()
    }

    fun stopAudio() {
        capture.stop()
        fusion.stop()
    }

    override fun close() {
        stopAudio()
        closeables.forEach { runCatching { it.close() } }
    }

    companion object {
        /**
         * Loads every model from assets (several seconds and ~200 MB): call off the main thread.
         * [onTranscript] is invoked on the fusion thread.
         */
        fun load(context: Context, onTranscript: (TranscriptEntry) -> Unit, useLightAsd: Boolean = USE_LIGHT_ASD): MultimodalSystem {
            val env = OrtEnvironment.getEnvironment()
            val closeables = mutableListOf<AutoCloseable>()

            val detector = YoloDetector.fromAssets(
                context, "models/yolov8n_face.onnx", confidenceThreshold = 0.3f, nmsThreshold = 0.7f
            )
            val embedder = EdgeFaceEmbedder.fromAssets(context, "models/edgeface_xxs.onnx")
            closeables += AutoCloseable { detector.close() }
            closeables += AutoCloseable { embedder.close() }
            val faceDatabase = VisualProfileDatabase.loadFromAssets(context)
            val facePipeline = HeadTrackerPipeline(detector, embedder, faceDatabase)

            val voice = VoiceEmbedder.fromAssets(context, "models/ecapa_voxceleb.onnx")
            val diarizer = Diarizer.fromAssets(context, "models/pyannote_segmentation.onnx")
            val whisper = WhisperTranscriber.fromAssets(context, "whisper")
            closeables += listOf(voice, diarizer, whisper)

            val timeline = AudioTimeline()
            val registry = IdentityRegistry()
            val lightAsd = if (useLightAsd) LightAsdModel.fromAssets(context, "models/light_asd.onnx").also { closeables += it } else null
            val asd = ActiveSpeakerDetector(lightAsd?.let { LightAsdDetector(it, timeline) }, clock = AudioUtils::nowSec)
            // Voices live in app storage (seeded from the exported av-tracker embeddings on the first run) so they persist.
            val verifier = SpeakerVerifier(DirectoryVoiceStore.open(context))

            // run_multimodal_tracker.py transcribes English (AMI); use Portuguese on Portuguese devices.
            val language = if (Locale.getDefault().language == "pt") "pt" else "en"
            val fusion = FusionEngine(
                timeline = timeline,
                turnDetector = diarizer,
                voice = voice,
                verifier = verifier,
                recognizer = whisper,
                asd = asd,
                registry = registry,
                faceTracker = RegistryFaceBridge(registry, knownFaces = { faceDatabase.names }),
                config = FusionConfig(language = language),
                entities = SpacyNer.fromAssets(context, language),
                onTranscript = onTranscript
            )
            val capture = AudioCapture { chunk, startSec -> timeline.append(chunk, startSec) }
            return MultimodalSystem(facePipeline, registry, FaceIdentitySync(registry), asd, useLightAsd, timeline, fusion, capture, closeables)
        }

        /**
         * The Python system's ASD is the mouth-region pixel difference: LightASDDetector fails to load its checkpoint
         * (GRU key names do not match), so ActiveSpeakerDetector always falls back. Enable this to use the network instead.
         */
        const val USE_LIGHT_ASD = false

        fun nowSec(): Double = AudioUtils.nowSec()
    }
}
