package com.avtracker.mobile.session

import com.avtracker.mobile.fusion.Decision
import com.avtracker.mobile.fusion.SessionLog
import com.avtracker.mobile.fusion.TranscriptEntry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.File
import java.nio.file.Files
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SessionFilesTest {

    private fun tempDir() = Files.createTempDirectory("session").toFile()

    @Test
    fun wavWriter_producesTheSameBytesAsPythonsWaveModule() {
        val samples = Json.parseToJsonElement(File("src/test/resources/session/samples.json").readText())
            .jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray()
        val dir = tempDir()
        try {
            val file = File(dir, "out.wav")
            WavWriter(file).use { w ->
                // written in uneven chunks, like the microphone delivers them
                w.write(samples.copyOfRange(0, 100))
                w.write(samples.copyOfRange(100, 101))
                w.write(samples.copyOfRange(101, samples.size))
                assertEquals(samples.size.toLong(), w.samples)
            }
            assertContentEquals(File("src/test/resources/session/expected.wav").readBytes(), file.readBytes())
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun entry(id: String, name: String?, text: String, wall: Long) =
        TranscriptEntry(id, name, text, "VERIFIER (x)", 1.0, wall)

    @Test
    fun transcript_usesVerifiedNameOrSpeakerIdInPythonsFormat() {
        val t0 = 1_700_000_000_000L
        val hms = SimpleDateFormat("HH:mm:ss", Locale.ROOT)
        val text = SessionSaver.transcriptText(
            listOf(
                entry("spk_001", "Manuela", "Bom dia a todos.", t0),
                entry("spk_002", null, "Good morning.", t0 + 5000),
                entry("spk_003", "", "Ok.", t0 + 9000)
            )
        )
        assertEquals(
            "[${hms.format(Date(t0))}] Manuela: Bom dia a todos.\n" +
                "[${hms.format(Date(t0 + 5000))}] spk_002: Good morning.\n" +
                "[${hms.format(Date(t0 + 9000))}] spk_003: Ok.\n",
            text
        )
    }

    private fun sampleLog(): SessionLog {
        val log = SessionLog(sessionStartMillis = 1_700_000_000_000L)
        val decision = Decision("spk_001", null, "FACE_ONLY (x)", null, null, listOf("Bob" to 0.5f, "Alice" to 0.48f), "Bob", 0.5f)
        val reasons = listOf("MARGIN_REJECT: Bob=0.500 vs Alice=0.480 (margin=0.020<0.04)")
        log.recordFalseNegative(
            log.newFalseNegative(reasons, decision, "Maria", "um texto", "Bob", 0.5f, decision.filtered, "female", 2, 1_700_000_005_000L)
        )
        log.recordSegment(
            1_700_000_005_000L, decision, "Bob", 0.5f, decision.filtered, "Maria", "female", true, 2, reasons,
            "high", listOf("WEAK_DECISION:FACE_ONLY"), 8, 3.2, 850.0, 120.0, 1400.0, -0.3f, 0.01f
        )
        return log
    }

    @Test
    fun save_writesEveryFileWithFaceSectionAndKeepsTheRecording() {
        val dir = tempDir()
        try {
            val recording = File(dir, ".recording.wav")
            WavWriter(recording).use { it.write(FloatArray(1600) { i -> (i % 50) / 100f }) }

            val face = buildJsonObject {
                put("total_tracks", 2); put("tracks_identified", 1); put("tracks_generic", 1); put("face_fn_count", 0)
                put("enrollments", 1); put("known_embeddings_count", 3); put("total_frames", 500)
                put("match_threshold", 0.65); put("merge_threshold", 0.55)
            }
            val saved = SessionSaver.save(
                dir, 1_700_000_100_000L, listOf(entry("spk_001", "Maria", "um texto", 1_700_000_005_000L)), sampleLog(),
                0.80f, 0.70f, 3, "base", face, recording
            )

            assertTrue(saved.transcript.name.matches(Regex("transcript_\\d{8}_\\d{6}\\.txt")))
            assertEquals(1, saved.transcript.readText().lines().count { it.contains("Maria: um texto") })

            val fn = assertNotNull(saved.fnReport).readText()
            assertTrue(fn.startsWith("FALSE NEGATIVE REPORT — 1 event(s)\n" + "=".repeat(60)))
            val fp = assertNotNull(saved.fpReport).readText()
            assertTrue(fp.startsWith("FALSE POSITIVE RISK REPORT — 1 suspect segment(s)\n  HIGH risk: 1  |  MEDIUM risk: 0\n"))

            val metrics = Json.parseToJsonElement(assertNotNull(saved.metrics).readText(Charsets.UTF_8)).jsonObject
            assertEquals(listOf("session_id", "session_start", "config", "summary", "segments", "face"), metrics.keys.toList())
            val config = metrics.getValue("config").jsonObject
            assertEquals(0.65, config.getValue("face_match_threshold").jsonPrimitive.double)
            assertEquals(0.55, config.getValue("face_merge_threshold").jsonPrimitive.double)
            assertEquals(3, config.getValue("num_speakers").jsonPrimitive.int)
            assertEquals(1, metrics.getValue("summary").jsonObject.getValue("total_segments").jsonPrimitive.int)
            assertEquals(500, metrics.getValue("face").jsonObject.getValue("total_frames").jsonPrimitive.int)
            assertTrue("match_threshold" !in metrics.getValue("face").jsonObject)
            assertTrue(assertNotNull(saved.metrics).readText().contains("\n  \"session_id\""), "indent=2 like json.dump(indent=2)")

            val audio = assertNotNull(saved.audio)
            assertTrue(audio.name.matches(Regex("audio_\\d{8}_\\d{6}\\.wav")))
            assertEquals(44L + 3200, audio.length())
            assertTrue(!recording.exists(), "the recording is renamed, not copied")
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun save_mirrorsTheSessionAndItsSegmentsIntoTheDatabase() {
        val dir = tempDir()
        try {
            val recording = File(dir, ".recording.wav")
            WavWriter(recording).use { it.write(FloatArray(1600)) }
            val sql = com.avtracker.mobile.db.JdbcSqlDatabase.inMemory()
            val database = com.avtracker.mobile.db.TrackerDb(sql)

            val saved = SessionSaver.save(
                dir, 1_700_000_100_000L, listOf(entry("spk_001", "Maria", "um texto", 1_700_000_005_000L)), sampleLog(),
                0.80f, 0.70f, 3, "base", null, recording, database
            )

            val ts = SessionSaver.sessionId(1_700_000_100_000L)
            val session = assertNotNull(database.getSession(ts))
            assertEquals("[", session["transcript"].toString().take(1))
            assertEquals(saved.transcript.readText(), session["transcript"])
            assertEquals(assertNotNull(saved.audio).absolutePath, session["audio_file"])
            assertEquals(1, (session["summary"] as JsonObject).getValue("total_segments").jsonPrimitive.int)
            assertEquals(3, (session["config"] as JsonObject).getValue("num_speakers").jsonPrimitive.int)

            val segments = database.getSegments(session["id"] as Long)
            assertEquals(1, segments.size)
            assertEquals("spk_001", segments[0]["final_speaker"])
            assertEquals("high", segments[0]["fp_risk"])
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun save_withNothingToReportWritesOnlyTheTranscript() {
        val dir = tempDir()
        try {
            val saved = SessionSaver.save(dir, 1_700_000_100_000L, emptyList(), SessionLog(), 0.80f, 0.70f, null, "base", null, null)
            assertTrue(saved.transcript.exists())
            assertEquals("", saved.transcript.readText())
            assertNull(saved.fnReport)
            assertNull(saved.fpReport)
            assertNull(saved.metrics)
            assertNull(saved.audio)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun metricsPayload_isNullWithoutSegments() {
        assertNull(SessionSaver.metricsPayload("x", SessionLog(), 0.8f, 0.7f, null, "base", JsonObject(emptyMap())))
    }
}
