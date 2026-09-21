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
import com.avtracker.mobile.db.AndroidSqlDatabase
import com.avtracker.mobile.db.DbMirror
import com.avtracker.mobile.db.TrackerDb
import com.avtracker.mobile.detection.YoloDetector
import com.avtracker.mobile.diarization.Diarizer
import com.avtracker.mobile.diarization.PyannoteDiarizer
import com.avtracker.mobile.fusion.TurnDetector
import com.avtracker.mobile.embedding.EdgeFaceEmbedder
import com.avtracker.mobile.fusion.FaceIdentitySync
import com.avtracker.mobile.fusion.FaceLabel
import com.avtracker.mobile.fusion.FusionConfig
import com.avtracker.mobile.fusion.FusionEngine
import com.avtracker.mobile.fusion.IdentityRegistry
import com.avtracker.mobile.fusion.LlmClient
import com.avtracker.mobile.fusion.PersonIdTrackerBridge
import com.avtracker.mobile.fusion.SpeakerNaming
import com.avtracker.mobile.fusion.TranscriptEntry
import com.avtracker.mobile.pipeline.FrameResult
import com.avtracker.mobile.pipeline.HeadTrackerPipeline
import com.avtracker.mobile.session.SavedSession
import com.avtracker.mobile.session.SessionSaver
import com.avtracker.mobile.session.WavWriter
import com.avtracker.mobile.tracking.DirectoryFaceStore
import com.avtracker.mobile.tracking.PersonIdTracker
import com.avtracker.mobile.voice.SpeakerVerifier
import com.avtracker.mobile.voice.VoiceEmbedder
import com.avtracker.mobile.voice.DirectoryVoiceStore
import com.avtracker.mobile.ner.SpacyNer
import com.avtracker.mobile.whisper.WhisperTranscriber
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

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
    private val recorder: AtomicReference<WavWriter?>,
    private val faceTracker: PersonIdTracker,
    private val config: FusionConfig,
    private val sessionDir: File,
    private val database: TrackerDb,
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

    /** Needs the RECORD_AUDIO permission. The microphone is also recorded to a WAV file, saved with the session. */
    fun startAudio() {
        sessionDir.mkdirs()
        recorder.getAndSet(WavWriter(File(sessionDir, RECORDING_FILE)))?.close()
        capture.start()
        fusion.start()
    }

    /** Stops listening. The recording is finalised; [saveSession] keeps it. */
    fun stopAudio() {
        capture.stop()
        fusion.stop()
        recorder.getAndSet(null)?.close()
    }

    // ---- end of session: name the unidentified voices, then save (run_multimodal_tracker.run) ----

    /** Anonymous speakers with at least 5 s of audio, worth asking the user to name. */
    fun pendingSpeakers(): List<SpeakerNaming.PendingSpeaker> = fusion.naming.unidentifiedSpeakers()

    /** Up to 10 s of the speaker's voice, normalised for playback (16 kHz mono floats). */
    fun previewAudio(pending: SpeakerNaming.PendingSpeaker): FloatArray = fusion.naming.previewAudio(pending.speakerId)

    /** The user named [pending]: the voice is stored under [name] and the session updated. */
    fun enrollSpeaker(pending: SpeakerNaming.PendingSpeaker, name: String) = fusion.naming.enrollPostSession(pending, name)

    /** Port of save_session: transcript, FN/FP reports, metrics (with the face section) and the recorded audio. */
    fun saveSession(): SavedSession = SessionSaver.save(
        dir = sessionDir,
        nowMillis = System.currentTimeMillis(),
        entries = fusion.transcript,
        log = fusion.log,
        verifierThreshold = fusion.verifier.threshold,
        verifierConfidenceMin = config.verifierConfidenceMin,
        numSpeakers = config.numSpeakers,
        whisperModel = config.whisperModel,
        faceMetrics = faceTracker.sessionFaceMetrics(),
        recordedAudio = File(sessionDir, RECORDING_FILE),
        database = database
    )

    override fun close() {
        stopAudio()
        closeables.forEach { runCatching { it.close() } }
    }

    companion object {
        /**
         * Loads every model from assets (several seconds and ~200 MB): call off the main thread.
         * [onTranscript] is invoked on the fusion thread.
         */
        fun load(
            context: Context,
            onTranscript: (TranscriptEntry) -> Unit,
            useLightAsd: Boolean = USE_LIGHT_ASD,
            numSpeakers: Int? = null,
            /** The LLM for the periodic speaker analysis; null keeps it off, like av-tracker's `use_ai_analysis=False`. */
            llm: LlmClient? = null,
            /**
             * The full pyannote 3.1 diarization (segmentation + WeSpeaker embeddings + clustering) instead of the fast
             * segmentation-only turns. It is what av-tracker runs, but on a phone it takes far longer than the 5 s it has per
             * window, so the transcript falls behind: off by default.
             */
            fullDiarization: Boolean = false
        ): MultimodalSystem {
            val env = OrtEnvironment.getEnvironment()
            val closeables = mutableListOf<AutoCloseable>()

            val detector = YoloDetector.fromAssets(
                context, "models/yolov8n_face.onnx", confidenceThreshold = 0.3f, nmsThreshold = 0.7f
            )
            val embedder = EdgeFaceEmbedder.fromAssets(context, "models/edgeface_xxs.onnx")
            closeables += AutoCloseable { detector.close() }
            closeables += AutoCloseable { embedder.close() }
            // Faces live in app storage (seeded from the exported av-tracker face embeddings on the first run), so
            // faces enrolled in a session persist, like data/face_embeddings does.
            // av_tracker.db in app storage: the SQLite copy of enrolments and sessions that the Python keeps.
            val database = TrackerDb(AndroidSqlDatabase.open(context))
            closeables += database
            val mirror = DbMirror(database)
            val faceTracker = PersonIdTracker(DirectoryFaceStore.open(context), maxIdentities = numSpeakers, onEnrolled = mirror::faceEnrolled)
            faceTracker.loadKnownEmbeddings()
            val facePipeline = HeadTrackerPipeline(detector, embedder, faceTracker, numSpeakers)

            val voice = VoiceEmbedder.fromAssets(context, "models/ecapa_voxceleb.onnx")
            val segmentation = Diarizer.fromAssets(context, "models/pyannote_segmentation.onnx")
            val diarizer: TurnDetector = if (fullDiarization) PyannoteDiarizer.fromAssets(context, segmentation, numSpeakers) else segmentation
            val whisper = WhisperTranscriber.fromAssets(context, "whisper")
            closeables += listOf(voice, diarizer as AutoCloseable, whisper)

            val timeline = AudioTimeline()
            val registry = IdentityRegistry()
            val lightAsd = if (useLightAsd) LightAsdModel.fromAssets(context, "models/light_asd.onnx").also { closeables += it } else null
            val asd = ActiveSpeakerDetector(lightAsd?.let { LightAsdDetector(it, timeline) }, clock = AudioUtils::nowSec)
            // Voices live in app storage (seeded from the exported av-tracker embeddings on the first run) so they persist.
            val verifier = SpeakerVerifier(DirectoryVoiceStore.open(context), onVoiceSaved = mirror::voiceSaved)

            // run_multimodal_tracker.py transcribes English (AMI); use Portuguese on Portuguese devices.
            val language = if (Locale.getDefault().language == "pt") "pt" else "en"
            val fusionConfig = FusionConfig(language = language, numSpeakers = numSpeakers)
            val fusion = FusionEngine(
                timeline = timeline,
                turnDetector = diarizer,
                voice = voice,
                verifier = verifier,
                recognizer = whisper,
                asd = asd,
                registry = registry,
                faceTracker = PersonIdTrackerBridge(faceTracker),
                config = fusionConfig,
                entities = SpacyNer.fromAssets(context, language),
                onTranscript = onTranscript,
                llm = llm
            )
            val recorder = AtomicReference<WavWriter?>(null)
            val capture = AudioCapture { chunk, startSec ->
                timeline.append(chunk, startSec)
                recorder.get()?.write(chunk)
            }
            val sessionDir = context.getExternalFilesDir("realtime_sessions") ?: File(context.filesDir, "realtime_sessions")
            return MultimodalSystem(
                facePipeline, registry, FaceIdentitySync(registry), asd, useLightAsd, timeline, fusion, capture,
                recorder, faceTracker, fusionConfig, sessionDir, database, closeables
            )
        }

        /**
         * The Python system's ASD is the mouth-region pixel difference: LightASDDetector fails to load its checkpoint
         * (GRU key names do not match), so ActiveSpeakerDetector always falls back. Enable this to use the network instead.
         */
        const val USE_LIGHT_ASD = false

        private const val RECORDING_FILE = ".recording.wav"

        fun nowSec(): Double = AudioUtils.nowSec()
    }
}
