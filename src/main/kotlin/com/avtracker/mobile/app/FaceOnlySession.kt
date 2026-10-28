package com.avtracker.mobile.app

import android.content.Context
import android.graphics.Bitmap
import com.avtracker.mobile.db.AndroidSqlDatabase
import com.avtracker.mobile.db.DbMirror
import com.avtracker.mobile.db.TrackerDb
import com.avtracker.mobile.detection.YoloDetector
import com.avtracker.mobile.embedding.EdgeFaceEmbedder
import com.avtracker.mobile.fusion.FaceIdentitySync
import com.avtracker.mobile.fusion.IdentityRegistry
import com.avtracker.mobile.fusion.SessionLog
import com.avtracker.mobile.pipeline.FrameResult
import com.avtracker.mobile.pipeline.HeadTrackerPipeline
import com.avtracker.mobile.session.SavedSession
import com.avtracker.mobile.session.SessionSaver
import com.avtracker.mobile.tracking.DirectoryFaceStore
import com.avtracker.mobile.tracking.PersonIdTracker
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File

/** Loads only face models. It never opens a microphone or starts a voice/transcription worker. */
class FaceOnlySession private constructor(
    override val facePipeline: HeadTrackerPipeline,
    private val sync: FaceIdentitySync,
    private val database: TrackerDb,
    private val sessionDir: File,
    private val closeables: List<AutoCloseable>
) : TrackingSession {
    override val localAudioEnabled = false
    private val log = SessionLog()

    override fun onFrame(frame: Bitmap, result: FrameResult, timestampSec: Double): FrameOverlay {
        val faces = FaceIdentitySync.fromSnapshots(result.persons)
        val current = sync.sync(faces)
        return FrameOverlay(sync.labels(faces, current).associateBy { it.trackId }, emptySet())
    }

    override fun saveSession(): SavedSession {
        val now = System.currentTimeMillis()
        val saved = SessionSaver.save(sessionDir, now, emptyList(), log, 0.8f, 0.7f,
            null, "external-vocalid", facePipeline.tracker.sessionFaceMetrics(), null, database)
        val metrics = File(sessionDir, "metrics_${SessionSaver.sessionId(now)}.json")
        val payload = buildJsonObject {
            put("mode", "faces_with_vocalid")
            put("audio_provider", "com.vocalid.jabra")
            put("face", facePipeline.tracker.sessionFaceMetrics())
        }
        metrics.writeText(Json.encodeToString(JsonObject.serializer(), payload), Charsets.UTF_8)
        return SavedSession(saved.transcript, null, null, metrics, null)
    }

    override fun close() { closeables.forEach { runCatching { it.close() } } }

    companion object {
        fun load(context: Context, numSpeakers: Int? = null): FaceOnlySession {
            val resources = mutableListOf<AutoCloseable>()
            try {
                val detector = YoloDetector.fromAssets(context, "models/yolov8n_face.onnx", 0.3f, 0.7f)
                resources += AutoCloseable { detector.close() }
                val embedder = EdgeFaceEmbedder.fromAssets(context, "models/edgeface_xxs.onnx")
                resources += AutoCloseable { embedder.close() }
                val database = TrackerDb(AndroidSqlDatabase.open(context))
                resources += database
                val mirror = DbMirror(database)
                val tracker = PersonIdTracker(DirectoryFaceStore.open(context), maxIdentities = numSpeakers,
                    onEnrolled = mirror::faceEnrolled)
                tracker.loadKnownEmbeddings()
                val directory = context.getExternalFilesDir("realtime_sessions") ?: File(context.filesDir, "realtime_sessions")
                return FaceOnlySession(HeadTrackerPipeline(detector, embedder, tracker, numSpeakers),
                    FaceIdentitySync(IdentityRegistry()), database, directory, resources)
            } catch (failure: Throwable) {
                resources.forEach { runCatching { it.close() } }
                throw failure
            }
        }
    }
}
