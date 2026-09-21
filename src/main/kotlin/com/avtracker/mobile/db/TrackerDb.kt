package com.avtracker.mobile.db

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/** A stored embedding: the table row plus the decoded float32 vector. */
class StoredEmbedding(val row: Map<String, Any?>, val embedding: FloatArray)

/**
 * Port of av-tracker's `src/database.py` (TrackerDB): the single-file SQLite store for speakers, voice and face
 * embeddings, sessions, segments and validations, with the same schema and the same SQL. The application mirrors what it
 * learns into it exactly where the Python does (auto-enrolled faces, saved voices, and the session at its end).
 * `apply_validation_feedback`, the validation GUI's file-editing helper, is not ported.
 */
class TrackerDb(private val db: SqlDatabase) : AutoCloseable {
    private val json = Json { ignoreUnknownKeys = true }

    init {
        db.executeScript(SCHEMA)
    }

    // ---- speakers ------------------------------------------------------------------------------------------------

    @Synchronized
    fun addSpeaker(name: String, gender: String? = null): Long = db.transaction {
        db.execute("INSERT OR IGNORE INTO speakers (name, gender) VALUES (?, ?)", listOf(name, gender))
        (db.query("SELECT id FROM speakers WHERE name = ?", listOf(name)).first()["id"] as Long)
    }

    @Synchronized
    fun getSpeaker(name: String): Map<String, Any?>? = db.query("SELECT * FROM speakers WHERE name = ?", listOf(name)).firstOrNull()

    @Synchronized
    fun listSpeakers(): List<Map<String, Any?>> = db.query(
        """
        SELECT s.*,
               COUNT(DISTINCT ve.id) as voice_count,
               COUNT(DISTINCT fe.id) as face_count
        FROM speakers s
        LEFT JOIN voice_embeddings ve ON ve.speaker_id = s.id AND ve.is_active = 1
        LEFT JOIN face_embeddings fe ON fe.speaker_id = s.id AND fe.is_active = 1
        GROUP BY s.id
        ORDER BY s.name
        """.trimIndent()
    )

    @Synchronized
    fun renameSpeaker(oldName: String, newName: String) = db.transaction {
        db.execute("UPDATE speakers SET name = ? WHERE name = ?", listOf(newName, oldName))
    }

    @Synchronized
    fun deleteSpeaker(name: String) = db.transaction { db.execute("DELETE FROM speakers WHERE name = ?", listOf(name)) }

    // ---- voice embeddings ----------------------------------------------------------------------------------------

    @Synchronized
    fun saveVoiceEmbedding(speakerId: Long, embedding: FloatArray, sourceFile: String? = null): Long =
        saveEmbedding("voice_embeddings", speakerId, embedding, sourceFile)

    @Synchronized
    fun getVoiceEmbeddings(speakerId: Long, activeOnly: Boolean = true): List<StoredEmbedding> =
        getEmbeddings("voice_embeddings", speakerId, activeOnly)

    /** `{speaker name: averaged, normalised embedding}`, what the verifier is built from. */
    @Synchronized
    fun getAllVoiceEmbeddings(activeOnly: Boolean = true): Map<String, FloatArray> = getAllEmbeddings("voice_embeddings", activeOnly)

    @Synchronized
    fun deactivateVoiceEmbedding(id: Long) = db.transaction { db.execute("UPDATE voice_embeddings SET is_active = 0 WHERE id = ?", listOf(id)) }

    @Synchronized
    fun deleteVoiceEmbedding(id: Long) = db.transaction { db.execute("DELETE FROM voice_embeddings WHERE id = ?", listOf(id)) }

    @Synchronized
    fun countVoiceEmbeddings(speakerId: Long): Long =
        db.query("SELECT COUNT(*) as n FROM voice_embeddings WHERE speaker_id = ? AND is_active = 1", listOf(speakerId)).first()["n"] as Long

    // ---- face embeddings -----------------------------------------------------------------------------------------

    @Synchronized
    fun saveFaceEmbedding(speakerId: Long, embedding: FloatArray, sourceFile: String? = null): Long =
        saveEmbedding("face_embeddings", speakerId, embedding, sourceFile)

    @Synchronized
    fun getFaceEmbeddings(speakerId: Long, activeOnly: Boolean = true): List<StoredEmbedding> =
        getEmbeddings("face_embeddings", speakerId, activeOnly)

    @Synchronized
    fun getAllFaceEmbeddings(activeOnly: Boolean = true): Map<String, FloatArray> = getAllEmbeddings("face_embeddings", activeOnly)

    @Synchronized
    fun deactivateFaceEmbedding(id: Long) = db.transaction { db.execute("UPDATE face_embeddings SET is_active = 0 WHERE id = ?", listOf(id)) }

    @Synchronized
    fun deleteFaceEmbedding(id: Long) = db.transaction { db.execute("DELETE FROM face_embeddings WHERE id = ?", listOf(id)) }

    private fun saveEmbedding(table: String, speakerId: Long, embedding: FloatArray, sourceFile: String?): Long = db.transaction {
        db.insert("INSERT INTO $table (speaker_id, embedding, source_file) VALUES (?, ?, ?)", listOf(speakerId, toBlob(embedding), sourceFile))
    }

    private fun getEmbeddings(table: String, speakerId: Long, activeOnly: Boolean): List<StoredEmbedding> {
        var sql = "SELECT * FROM $table WHERE speaker_id = ?"
        if (activeOnly) sql += " AND is_active = 1"
        return db.query(sql, listOf(speakerId)).map { StoredEmbedding(it, fromBlob(it["embedding"] as ByteArray)) }
    }

    private fun getAllEmbeddings(table: String, activeOnly: Boolean): Map<String, FloatArray> {
        var sql = "SELECT s.name, e.embedding FROM $table e JOIN speakers s ON s.id = e.speaker_id"
        if (activeOnly) sql += " WHERE e.is_active = 1"
        val raw = LinkedHashMap<String, MutableList<FloatArray>>()
        for (row in db.query(sql)) raw.getOrPut(row["name"] as String) { mutableListOf() } += fromBlob(row["embedding"] as ByteArray)

        return raw.mapValues { (_, embs) ->
            val avg = FloatArray(embs[0].size)
            for (e in embs) for (i in avg.indices) avg[i] += e[i]
            for (i in avg.indices) avg[i] /= embs.size
            var sum = 0f // numpy: float32 throughout
            for (v in avg) sum += v * v
            val norm = sqrt(sum)
            if (norm > 0) FloatArray(avg.size) { avg[it] / norm } else avg
        }
    }

    // ---- sessions ------------------------------------------------------------------------------------------------

    /** INSERT OR REPLACE by `session_id`; empty [config] / [summary] are stored as NULL, like Python's falsy dicts. */
    @Synchronized
    fun saveSession(
        sessionId: String,
        startedAt: String? = null,
        config: JsonObject? = null,
        summary: JsonObject? = null,
        transcript: String? = null,
        audioFile: String? = null
    ): Long = db.transaction {
        db.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, started_at, config, summary, transcript, audio_file)
            VALUES (?, ?, ?, ?, ?, ?)
            """.trimIndent(),
            listOf(
                sessionId, startedAt,
                config?.takeIf { it.isNotEmpty() }?.let(PyJson::dumps),
                summary?.takeIf { it.isNotEmpty() }?.let(PyJson::dumps),
                transcript, audioFile
            )
        )
        db.query("SELECT id FROM sessions WHERE session_id = ?", listOf(sessionId)).first()["id"] as Long
    }

    @Synchronized
    fun getSession(sessionId: String): Map<String, Any?>? {
        val row = db.query("SELECT * FROM sessions WHERE session_id = ?", listOf(sessionId)).firstOrNull() ?: return null
        return withParsedJson(row)
    }

    @Synchronized
    fun listSessions(): List<Map<String, Any?>> = db.query(
        """
        SELECT s.*,
               COUNT(seg.id) as segment_count,
               v.accuracy as validated_accuracy
        FROM sessions s
        LEFT JOIN segments seg ON seg.session_pk = s.id
        LEFT JOIN validations v ON v.session_pk = s.id
        GROUP BY s.id
        ORDER BY s.session_id DESC
        """.trimIndent()
    ).map(::withParsedJson)

    private fun withParsedJson(row: Map<String, Any?>): Map<String, Any?> = LinkedHashMap(row).also {
        it["config"] = (row["config"] as String?)?.let { s -> json.parseToJsonElement(s) } ?: JsonObject(emptyMap())
        it["summary"] = (row["summary"] as String?)?.let { s -> json.parseToJsonElement(s) } ?: JsonObject(emptyMap())
    }

    // ---- segments ------------------------------------------------------------------------------------------------

    /** One row per segment metrics record (the `_segment_metrics` dicts), numbered from 0. */
    @Synchronized
    fun saveSegments(sessionPk: Long, segments: List<JsonObject>) = db.transaction {
        for ((i, seg) in segments.withIndex()) {
            fun text(key: String): String? = (seg[key] as? JsonPrimitive)?.takeIf { it !is JsonNull }?.contentOrNull
            // `seg.get(key, 0)`: a missing key is 0, an explicit null stays NULL
            fun num(key: String): Any? = when (val v = seg[key]) {
                null -> 0
                is JsonNull -> null
                is JsonPrimitive -> v.doubleOrNull ?: 0.0
                else -> 0.0
            }
            fun flag(key: String): Long = (seg[key] as? JsonPrimitive)?.let { p ->
                p.booleanOrNull?.let { if (it) 1L else 0L } ?: (p.doubleOrNull?.let { if (it != 0.0) 1L else 0L }) ?: 0L
            } ?: 0L

            val finalName = text("final_name")?.takeIf { it.isNotEmpty() }
            val finalSpeaker = text("final_speaker")
            db.execute(
                """
                INSERT INTO segments
                    (session_pk, seg_index, timestamp, speaker_name, text,
                     decision, decision_type, raw_best_name, raw_conf,
                     final_name, final_speaker, is_identified, is_fn,
                     fp_risk, audio_gender, whisper_ms, verifier_ms,
                     segment_ms, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.trimIndent(),
                listOf(
                    sessionPk, i.toLong(),
                    text("timestamp"),
                    finalName ?: finalSpeaker, // seg.get("final_name") or seg.get("final_speaker")
                    text("text") ?: "",
                    text("decision"),
                    text("decision_type"),
                    text("raw_best_name"),
                    num("raw_conf"),
                    text("final_name"),
                    finalSpeaker,
                    flag("is_identified"),
                    flag("is_fn"),
                    text("fp_risk") ?: "none",
                    text("audio_gender"),
                    num("whisper_ms"),
                    num("verifier_ms"),
                    num("segment_ms"),
                    PyJson.dumps(seg)
                )
            )
        }
    }

    @Synchronized
    fun getSegments(sessionPk: Long): List<Map<String, Any?>> =
        db.query("SELECT * FROM segments WHERE session_pk = ? ORDER BY seg_index", listOf(sessionPk)).map { row ->
            LinkedHashMap(row).also {
                it["metrics"] = (row["metrics_json"] as String?)?.let { s -> json.parseToJsonElement(s) } ?: JsonObject(emptyMap())
            }
        }

    // ---- validations ---------------------------------------------------------------------------------------------

    @Synchronized
    fun saveValidation(
        sessionPk: Long, accuracy: Double, namedAccuracy: Double, totalLines: Int, correct: Int, incorrect: Int,
        details: JsonObject? = null, lines: List<Triple<Int, String?, String?>> = emptyList()
    ): Long = db.transaction {
        val id = db.insert(
            """
            INSERT INTO validations
                (session_pk, accuracy, named_accuracy, total_lines,
                 correct, incorrect, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """.trimIndent(),
            listOf(sessionPk, accuracy, namedAccuracy, totalLines.toLong(), correct.toLong(), incorrect.toLong(), details?.takeIf { it.isNotEmpty() }?.let(PyJson::dumps))
        )
        for ((segIndex, predicted, corrected) in lines) {
            db.execute(
                "INSERT INTO validation_lines (validation_id, seg_index, predicted, corrected) VALUES (?, ?, ?, ?)",
                listOf(id, segIndex.toLong(), predicted, corrected)
            )
        }
        id
    }

    @Synchronized
    fun getValidation(sessionPk: Long): Map<String, Any?>? {
        val row = db.query("SELECT * FROM validations WHERE session_pk = ? ORDER BY id DESC LIMIT 1", listOf(sessionPk)).firstOrNull() ?: return null
        return LinkedHashMap(row).also {
            it["details"] = (row["details_json"] as String?)?.let { s -> json.parseToJsonElement(s) } ?: JsonObject(emptyMap())
        }
    }

    /** Chronological sessions with their summary and validated accuracy (for graphs). */
    @Synchronized
    fun getSessionAccuracyTrend(): List<Map<String, Any?>> =
        db.query(
            """
            SELECT s.session_id, s.summary,
                   v.accuracy, v.named_accuracy
            FROM sessions s
            LEFT JOIN validations v ON v.session_pk = s.id
            ORDER BY s.session_id
            """.trimIndent()
        ).map { row ->
            val out = LinkedHashMap<String, Any?>()
            out["session_id"] = row["session_id"]
            ((row["summary"] as String?)?.let { json.parseToJsonElement(it) } as? JsonObject)?.forEach { (k, v) -> out[k] = v }
            out["validated_accuracy"] = row["accuracy"]
            out["validated_named_accuracy"] = row["named_accuracy"]
            out
        }

    override fun close() = db.close()

    companion object {
        private fun toBlob(embedding: FloatArray): ByteArray {
            val buf = ByteBuffer.allocate(embedding.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (v in embedding) buf.putFloat(v)
            return buf.array()
        }

        private fun fromBlob(bytes: ByteArray): FloatArray {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(bytes.size / 4) { buf.getFloat() }
        }

        /** The schema of `_SCHEMA` in database.py, statement for statement. */
        val SCHEMA = """
            CREATE TABLE IF NOT EXISTS speakers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                gender      TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(name)
            );

            CREATE TABLE IF NOT EXISTS voice_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
                embedding   BLOB    NOT NULL,
                source_file TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active   INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
                embedding   BLOB    NOT NULL,
                source_file TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active   INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    UNIQUE NOT NULL,
                started_at  TEXT,
                config      TEXT,
                summary     TEXT,
                transcript  TEXT,
                audio_file  TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS segments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_pk      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                seg_index       INTEGER NOT NULL,
                timestamp       TEXT,
                speaker_name    TEXT,
                text            TEXT,
                decision        TEXT,
                decision_type   TEXT,
                raw_best_name   TEXT,
                raw_conf        REAL,
                final_name      TEXT,
                final_speaker   TEXT,
                is_identified   INTEGER,
                is_fn           INTEGER,
                fp_risk         TEXT,
                audio_gender    TEXT,
                whisper_ms      REAL,
                verifier_ms     REAL,
                segment_ms      REAL,
                metrics_json    TEXT
            );

            CREATE TABLE IF NOT EXISTS validations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_pk      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                validated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                accuracy        REAL,
                named_accuracy  REAL,
                total_lines     INTEGER,
                correct         INTEGER,
                incorrect       INTEGER,
                details_json    TEXT
            );

            CREATE TABLE IF NOT EXISTS validation_lines (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id   INTEGER NOT NULL REFERENCES validations(id) ON DELETE CASCADE,
                seg_index       INTEGER NOT NULL,
                predicted       TEXT,
                corrected       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_pk);
            CREATE INDEX IF NOT EXISTS idx_voice_emb_speaker ON voice_embeddings(speaker_id);
            CREATE INDEX IF NOT EXISTS idx_face_emb_speaker ON face_embeddings(speaker_id);
            CREATE INDEX IF NOT EXISTS idx_validation_lines_val ON validation_lines(validation_id);
        """.trimIndent()
    }
}
