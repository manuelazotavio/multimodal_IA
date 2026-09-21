package com.avtracker.mobile.session

import android.util.Log
import com.avtracker.mobile.db.TrackerDb
import com.avtracker.mobile.fusion.SessionLog
import com.avtracker.mobile.fusion.TranscriptEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** The files one session produced; every entry except the transcript exists only when there was something to write. */
class SavedSession(val transcript: File, val fnReport: File?, val fpReport: File?, val metrics: File?, val audio: File?)

/**
 * Port of RealtimeTranscriber.save_session: `transcript_<ts>.txt`, `fn_report_<ts>.txt`, `fp_report_<ts>.txt`,
 * `metrics_<ts>.json` (with the face tracker's section) and `audio_<ts>.wav`, in the same formats, and mirrors the session
 * (config, summary, transcript, every segment) into the SQLite database when one is given.
 */
object SessionSaver {
    private val prettyJson = Json { prettyPrint = true; prettyPrintIndent = "  " }

    fun sessionId(nowMillis: Long): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date(nowMillis))

    /** `[HH:MM:SS] <verified name or speaker id>: <text>` per entry. */
    fun transcriptText(entries: List<TranscriptEntry>): String = buildString {
        val hms = SimpleDateFormat("HH:mm:ss", Locale.ROOT)
        for (e in entries) {
            val who = e.speakerName?.takeIf { it.isNotEmpty() } ?: e.speakerId
            append("[${hms.format(Date(e.wallMillis))}] $who: ${e.text}\n")
        }
    }

    /**
     * Metrics as the Python writes them, plus the face tracker's `config` thresholds and its `face` section when
     * [faceMetrics] (PersonIdTracker.sessionFaceMetrics) is given. Null when no segment was processed.
     */
    fun metricsPayload(
        sessionId: String,
        log: SessionLog,
        verifierThreshold: Float,
        verifierConfidenceMin: Float,
        numSpeakers: Int?,
        whisperModel: String,
        faceMetrics: JsonObject?
    ): JsonObject? {
        val base = log.metricsJson(sessionId, verifierThreshold, verifierConfidenceMin, numSpeakers, whisperModel) ?: return null
        if (faceMetrics == null) return base

        val config = (base.getValue("config") as JsonObject).toMutableMap<String, JsonElement>()
        config["face_match_threshold"] = faceMetrics.getValue("match_threshold")
        config["face_merge_threshold"] = faceMetrics.getValue("merge_threshold")
        val faceKeys = listOf(
            "total_tracks", "tracks_identified", "tracks_generic", "face_fn_count", "face_fn_events", "enrollments",
            "known_embeddings_count", "total_frames", "match_score_stats", "match_scores_sample"
        )
        return buildJsonObject {
            for ((k, v) in base) put(k, if (k == "config") JsonObject(config) else v)
            put("face", JsonObject(faceKeys.associateWith { faceMetrics[it] ?: JsonNull }))
        }
    }

    /**
     * Writes the session files into [dir]. [recordedAudio] is the WAV recorded while listening; it is renamed to
     * `audio_<ts>.wav` (nothing is written when it is missing or empty).
     */
    fun save(
        dir: File,
        nowMillis: Long,
        entries: List<TranscriptEntry>,
        log: SessionLog,
        verifierThreshold: Float,
        verifierConfidenceMin: Float,
        numSpeakers: Int?,
        whisperModel: String,
        faceMetrics: JsonObject?,
        recordedAudio: File?,
        database: TrackerDb? = null
    ): SavedSession {
        dir.mkdirs()
        val ts = sessionId(nowMillis)

        val transcript = File(dir, "transcript_$ts.txt")
        transcript.writeText(transcriptText(entries), Charsets.UTF_8)

        val fn = log.falseNegativeReport()?.let { File(dir, "fn_report_$ts.txt").apply { writeText(it, Charsets.UTF_8) } }
        val fp = log.falsePositiveReport()?.let { File(dir, "fp_report_$ts.txt").apply { writeText(it, Charsets.UTF_8) } }

        val payload = metricsPayload(ts, log, verifierThreshold, verifierConfidenceMin, numSpeakers, whisperModel, faceMetrics)
        val metrics = payload?.let { p -> File(dir, "metrics_$ts.json").apply { writeText(prettyJson.encodeToString(JsonObject.serializer(), p), Charsets.UTF_8) } }

        var audio: File? = null
        if (recordedAudio != null && recordedAudio.exists() && recordedAudio.length() > 44) {
            val target = File(dir, "audio_$ts.wav")
            if (target.exists()) target.delete()
            if (recordedAudio.renameTo(target)) audio = target
        }

        // `db.save_session(...)` then `db.save_segments(...)`, guarded like the Python's try/except.
        if (database != null) {
            try {
                val pk = database.saveSession(
                    sessionId = ts,
                    startedAt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.ROOT).format(Date(log.sessionStartMillis)),
                    config = payload?.get("config") as? JsonObject,
                    summary = payload?.get("summary") as? JsonObject,
                    transcript = transcript.readText(Charsets.UTF_8),
                    audioFile = audio?.absolutePath
                )
                if (log.segmentMetrics.isNotEmpty()) database.saveSegments(pk, log.segmentMetrics)
            } catch (t: Throwable) {
                Log.w("SessionSaver", "Failed to save session to database", t)
            }
        }
        return SavedSession(transcript, fn, fp, metrics, audio)
    }
}
