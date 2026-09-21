package com.avtracker.mobile.db

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** TrackerDb against av-tracker's own TrackerDB (scripts/parity/db_reference.py): same operations, same tables. */
class TrackerDbTest {
    private val reference: JsonObject by lazy {
        Json.parseToJsonElement(File("src/test/resources/parity/db_reference.json").readText()).jsonObject
    }

    private val tsColumns = setOf("created_at", "validated_at")
    private val jsonColumns = setOf("config", "summary", "metrics_json", "details_json")

    private fun floats(e: JsonElement) = FloatArray(e.jsonArray.size) { e.jsonArray[it].jsonPrimitive.content.toFloat() }

    /** A value as the reference JSON has it: timestamps masked, blobs as float lists, JSON columns parsed. */
    private fun view(key: String?, v: Any?): JsonElement = when {
        key in tsColumns -> JsonPrimitive("<ts>")
        v == null -> JsonNull
        v is ByteArray -> {
            val buf = ByteBuffer.wrap(v).order(ByteOrder.LITTLE_ENDIAN)
            JsonArray(List(v.size / 4) { JsonPrimitive(buf.getFloat().toDouble()) })
        }
        key in jsonColumns && v is String -> Json.parseToJsonElement(v)
        v is JsonElement -> v
        v is Long -> JsonPrimitive(v)
        v is Double -> JsonPrimitive(v)
        v is Map<*, *> -> JsonObject(v.entries.associate { (k, x) -> k.toString() to view(k.toString(), x) })
        v is List<*> -> JsonArray(v.map { view(null, it) })
        v is FloatArray -> JsonArray(v.map { JsonPrimitive(it.toDouble()) })
        else -> JsonPrimitive(v.toString())
    }

    /** Numbers compare as doubles (SQLite hands back 1 where the JSON has 1.0). */
    private fun normalize(e: JsonElement): JsonElement = when (e) {
        is JsonPrimitive -> if (e.isString) e else e.doubleOrNull?.let { JsonPrimitive(it) } ?: e
        is JsonArray -> JsonArray(e.map(::normalize))
        is JsonObject -> JsonObject(e.mapValues { normalize(it.value) })
        else -> e
    }

    private fun assertSame(label: String, expected: JsonElement, actual: JsonElement) {
        assertEquals(normalize(expected), normalize(actual), label)
    }

    private class Ids(val speakers: Map<String, Long>, val sessions: Map<String, Long>)

    private fun replay(db: TrackerDb): Ids {
        val speakers = HashMap<String, Long>()
        val sessions = HashMap<String, Long>()
        val voiceIds = ArrayList<Long>()
        val faceIds = ArrayList<Long>()
        fun str(e: JsonElement?) = if (e == null || e is JsonNull) null else e.jsonPrimitive.content
        fun obj(e: JsonElement?) = if (e == null || e is JsonNull) null else e.jsonObject

        for (o in reference.getValue("ops").jsonArray.map { it.jsonObject }) {
            when (o.getValue("op").jsonPrimitive.content) {
                "add_speaker" -> speakers[str(o["name"])!!] = db.addSpeaker(str(o["name"])!!, str(o["gender"]))
                "voice" -> voiceIds += db.saveVoiceEmbedding(speakers.getValue(str(o["speaker"])!!), floats(o.getValue("vec")), str(o["source"]))
                "face" -> faceIds += db.saveFaceEmbedding(speakers.getValue(str(o["speaker"])!!), floats(o.getValue("vec")), str(o["source"]))
                "deactivate_voice" -> db.deactivateVoiceEmbedding(voiceIds[o.getValue("index").jsonPrimitive.int])
                "deactivate_face" -> db.deactivateFaceEmbedding(faceIds[o.getValue("index").jsonPrimitive.int])
                "delete_face" -> db.deleteFaceEmbedding(faceIds[o.getValue("index").jsonPrimitive.int])
                "session" -> sessions[str(o["session_id"])!!] = db.saveSession(
                    str(o["session_id"])!!, str(o["started_at"]), obj(o["config"]), obj(o["summary"]), str(o["transcript"]), str(o["audio_file"])
                )
                "segments" -> db.saveSegments(sessions.getValue(str(o["session"])!!), o.getValue("segments").jsonArray.map { it.jsonObject })
                "validation" -> db.saveValidation(
                    sessions.getValue(str(o["session"])!!), o.getValue("accuracy").jsonPrimitive.content.toDouble(),
                    o.getValue("named_accuracy").jsonPrimitive.content.toDouble(), o.getValue("total_lines").jsonPrimitive.int,
                    o.getValue("correct").jsonPrimitive.int, o.getValue("incorrect").jsonPrimitive.int, obj(o["details"]),
                    o.getValue("lines").jsonArray.map { l -> Triple(l.jsonArray[0].jsonPrimitive.int, str(l.jsonArray[1]), str(l.jsonArray[2])) }
                )
                "rename_speaker" -> db.renameSpeaker(str(o["old"])!!, str(o["new"])!!)
                "delete_speaker" -> db.deleteSpeaker(str(o["name"])!!)
                else -> error("unknown op $o")
            }
        }
        return Ids(speakers, sessions)
    }

    private fun rowsView(rows: List<Map<String, Any?>>) = JsonArray(rows.map { r -> JsonObject(r.mapValues { (k, v) -> view(k, v) }) })

    private fun rowView(row: Map<String, Any?>) = JsonObject(row.mapValues { (k, v) -> view(k, v) })

    @Test
    fun everyTableAndQueryMatchesTheOriginalTrackerDb() {
        val sql = JdbcSqlDatabase.inMemory()
        val db = TrackerDb(sql)
        val ids = replay(db)
        val expected = reference.getValue("expected").jsonObject

        for ((table, rows) in expected.getValue("tables").jsonObject) {
            assertSame("table $table", rows, rowsView(sql.query("SELECT * FROM $table ORDER BY id")))
        }

        // averaged and normalised in float32: numpy's BLAS may sum in another order, so compare within a few ulps
        fun assertVectors(label: String, want: JsonElement, got: Map<String, FloatArray>) {
            assertEquals(want.jsonObject.keys, got.keys, "$label names")
            for ((name, vec) in want.jsonObject) {
                val expectedVec = floats(vec)
                for (i in expectedVec.indices) assertTrue(kotlin.math.abs(expectedVec[i] - got.getValue(name)[i]) < 5e-7f, "$label $name[$i]: ${expectedVec[i]} vs ${got.getValue(name)[i]}")
            }
        }
        assertVectors("all_voice", expected.getValue("all_voice"), db.getAllVoiceEmbeddings())
        assertVectors("all_face", expected.getValue("all_face"), db.getAllFaceEmbeddings())
        assertVectors("all_voice_inactive_too", expected.getValue("all_voice_inactive_too"), db.getAllVoiceEmbeddings(activeOnly = false))

        assertSame("list_speakers", expected.getValue("list_speakers"), rowsView(db.listSpeakers()))
        assertSame("list_sessions", expected.getValue("list_sessions"), rowsView(db.listSessions()))
        assertSame("get_session", expected.getValue("get_session"), rowView(db.getSession("20260101_101010")!!))
        assertNull(db.getSession("nope"))
        assertSame("get_speaker", expected.getValue("get_speaker"), rowView(db.getSpeaker("Manuela")!!))
        assertEquals(expected.getValue("count_voice").jsonPrimitive.content.toLong(), db.countVoiceEmbeddings(ids.speakers.getValue("Manuela")))
        assertSame("get_segments", expected.getValue("get_segments"), rowsView(db.getSegments(ids.sessions.getValue("20260101_101010"))))
        assertSame("trend", expected.getValue("trend"), rowsView(db.getSessionAccuracyTrend()))
        db.close()
    }

    @Test
    fun pyJson_matchesPythonsJsonDumps() {
        val value = Json.parseToJsonElement("""{"a":1,"b":[1,2,{"c":null}],"d":"ç\"\\\n\t","e":true,"f":0.5,"g":{}}""")
        // python: json.dumps(value, ensure_ascii=False)
        assertEquals("""{"a": 1, "b": [1, 2, {"c": null}], "d": "ç\"\\\n\t", "e": true, "f": 0.5, "g": {}}""", PyJson.dumps(value))
    }

    @Test
    fun schemaIsIdempotentAndForeignKeysCascade() {
        val sql = JdbcSqlDatabase.inMemory()
        val db = TrackerDb(sql)
        sql.executeScript(TrackerDb.SCHEMA) // CREATE ... IF NOT EXISTS: a second open changes nothing
        val id = db.addSpeaker("Ana")
        db.saveVoiceEmbedding(id, floatArrayOf(1f, 0f))
        db.deleteSpeaker("Ana")
        assertTrue(sql.query("SELECT * FROM voice_embeddings").isEmpty(), "deleting a speaker removes its embeddings")
    }
}
