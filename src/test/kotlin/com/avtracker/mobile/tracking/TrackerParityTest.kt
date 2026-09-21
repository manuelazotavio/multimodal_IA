package com.avtracker.mobile.tracking

import com.avtracker.mobile.tracking.bytetrack.ByteTrack
import com.avtracker.mobile.tracking.bytetrack.Detection
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.file.Files
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The face tracker against av-tracker's own PersonIDTracker + boxmot ByteTrack (scripts/parity/tracker_parity.py):
 * the same detections and embeddings must give the same boxes, identities, enrolments, renames, files and metrics
 * on every frame.
 */
class TrackerParityTest {

    private val scenarios: List<JsonObject> by lazy {
        Json.parseToJsonElement(File("src/test/resources/parity/tracker_scenarios.json").readText())
            .jsonObject.getValue("scenarios").jsonArray.map { it.jsonObject }
    }

    private fun JsonElement.doubles() = jsonArray.map { it.jsonPrimitive.double }
    private fun JsonElement.floats() = FloatArray(jsonArray.size) { jsonArray[it].jsonPrimitive.double.toFloat() }
    private fun JsonElement.string() = jsonPrimitive.content
    private fun JsonElement.nullableInt() = if (this is JsonNull) null else jsonPrimitive.int

    private fun detections(frame: JsonObject) = frame.getValue("dets").jsonArray.map {
        val d = it.doubles()
        Detection(d[0], d[1], d[2], d[3], d[4], d[5])
    }

    private fun assertNear(expected: Double, actual: Double, tol: Double, what: String) {
        assertTrue(abs(expected - actual) <= tol, "$what: expected $expected, got $actual")
    }

    private fun assertVec(expected: FloatArray, actual: FloatArray?, tol: Float, what: String) {
        assertTrue(actual != null && actual.size == expected.size, "$what: missing or wrong size")
        for (i in expected.indices) assertTrue(abs(expected[i] - actual!![i]) <= tol, "$what[$i]: expected ${expected[i]}, got ${actual[i]}")
    }

    @Test
    fun byteTrack_matchesBoxmotOnEveryFrame() {
        for ((n, sc) in scenarios.withIndex()) {
            val expected = sc.getValue("expected").jsonObject.getValue("bytetrack").jsonArray
            val tracker = ByteTrack(trackBuffer = PersonIdTracker.TRACK_BUFFER)
            for ((f, frame) in sc.getValue("input").jsonObject.getValue("frames").jsonArray.withIndex()) {
                val dets = detections(frame.jsonObject)
                if (dets.isEmpty()) continue
                val out = tracker.update(dets)
                val rows = expected[f].jsonArray
                val where = "scenario $n frame $f"
                assertEquals(rows.size, out.size, "$where: track count")
                for ((k, row) in rows.withIndex()) {
                    val e = row.doubles()
                    val a = out[k]
                    assertEquals(e[4].toInt(), a.id, "$where row $k id")
                    assertEquals(e[7].toInt(), a.detIndex, "$where row $k det index")
                    for ((j, v) in listOf(a.x1, a.y1, a.x2, a.y2).withIndex()) assertNear(e[j], v, 1e-6, "$where row $k box[$j]")
                    assertNear(e[5], a.confidence, 1e-9, "$where row $k conf")
                }
            }
        }
    }

    @Test
    fun personIdTracker_matchesPythonOnEveryFrame() {
        val generic = Regex("^Person_\\d+$")
        val utc = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }

        for ((n, sc) in scenarios.withIndex()) {
            val input = sc.getValue("input").jsonObject
            val expected = sc.getValue("expected").jsonObject
            val dir = Files.createTempDirectory("faces_$n").toFile()
            try {
                val store = DirectoryFaceStore(dir)
                for (f in input.getValue("files").jsonArray) store.write(f.jsonObject.getValue("file").string(), f.jsonObject.getValue("vec").floats())

                var clock = input.getValue("frames").jsonArray[0].jsonObject.getValue("t").jsonPrimitive.double
                val maxIdentities = input.getValue("max_identities").nullableInt()
                val dbCalls = ArrayList<List<Any>>()
                val tracker = PersonIdTracker(
                    store, maxIdentities, clock = { clock },
                    timestamp = { utc.format(Date((clock * 1000).toLong())) },
                    onEnrolled = { name, embedding, file ->
                        dbCalls += listOf("add_speaker", name)
                        dbCalls += listOf("face", name, file, Math.round(embedding.sumOf { it.toDouble() } * 1000) / 1000.0, embedding.size)
                    }
                )
                tracker.loadKnownEmbeddings()

                val loaded = expected.getValue("loaded").jsonObject
                assertEquals(loaded.keys, tracker.knownNames, "scenario $n: names loaded from disk")
                for ((name, vec) in loaded) assertVec(vec.floats(), tracker.embeddingOf(name), 2e-6f, "scenario $n loaded $name")
                assertEquals(expected.getValue("loaded_files").jsonArray.map { it.string() }, store.list(), "scenario $n: files after load")

                val actionsByFrame = input.getValue("actions").jsonArray.groupBy { it.jsonObject.getValue("frame").jsonPrimitive.int }
                val expectedFrames = expected.getValue("frames").jsonArray

                for ((f, frameEl) in input.getValue("frames").jsonArray.withIndex()) {
                    val frame = frameEl.jsonObject
                    clock = frame.getValue("t").jsonPrimitive.double
                    val dets = detections(frame)
                    val labels = frame.getValue("labels").jsonArray.map { it.jsonPrimitive.int }
                    val embs = frame.getValue("embs").jsonObject

                    // the label painted at the crop centre: the last detection whose box covers that pixel
                    val source = FaceEmbeddingSource { left, top, right, bottom ->
                        val px = left + (right - left) / 2
                        val py = top + (bottom - top) / 2
                        var label = 0
                        for (k in dets.indices.reversed()) {
                            val d = dets[k]
                            if (py >= d.y1.toInt() && py < d.y2.toInt() && px >= d.x1.toInt() && px < d.x2.toInt()) {
                                label = labels[k]
                                break
                            }
                        }
                        embs.getValue(label.toString()).floats()
                    }

                    val results = tracker.update(640, 480, dets, source)
                    val where = "scenario $n frame $f"
                    val exp = expectedFrames[f].jsonObject

                    val expRows = exp.getValue("results").jsonArray
                    assertEquals(expRows.size, results.size, "$where: face count")
                    for ((k, row) in expRows.withIndex()) {
                        val r = row.jsonArray
                        val a = results[k]
                        assertEquals(r[0].jsonPrimitive.int, a.trackId, "$where face $k track id")
                        assertEquals(r[1].string(), a.name, "$where face $k name (track ${a.trackId})")
                        assertNear(r[2].jsonPrimitive.double, a.confidence, 1e-4, "$where face $k confidence")
                        for ((j, v) in listOf(a.x1, a.y1, a.x2, a.y2).withIndex()) assertNear(r[3 + j].jsonPrimitive.double, v, 1e-6, "$where face $k box[$j]")
                    }

                    // renames asked from outside (what the voice side does once it knows a name)
                    val expActions = exp.getValue("actions").jsonArray
                    val requested = actionsByFrame[f].orEmpty()
                    assertEquals(expActions.size, requested.size, "$where: action count")
                    for ((k, req) in requested.withIndex()) {
                        val op = req.jsonObject.getValue("op").string()
                        val newName = req.jsonObject.getValue("new").string()
                        val target = results.firstOrNull {
                            (op.startsWith("generic") && generic.matches(it.name)) ||
                                (op == "real_only_track" && it.name != "Unknown" && !generic.matches(it.name))
                        }
                        val e = expActions[k].jsonObject
                        assertEquals(e.getValue("skipped").jsonPrimitive.boolean, target == null, "$where: action $op skipped?")
                        if (target != null) {
                            val only = if (op.endsWith("only_track")) target.trackId else null
                            assertEquals(e.getValue("old").string(), target.name, "$where: action old name")
                            assertEquals(e.getValue("only").nullableInt(), only, "$where: action track")
                            tracker.renamePerson(target.name, newName, only)
                        }
                    }

                    val expNames = exp.getValue("track_to_name").jsonArray.map { it.jsonArray[0].jsonPrimitive.int to it.jsonArray[1].string() }
                    assertEquals(expNames, tracker.trackNames().map { it.key to it.value }, "$where: track -> name")
                }

                val fin = expected.getValue("final").jsonObject
                val known = fin.getValue("known").jsonObject
                assertEquals(known.keys, tracker.knownNames, "scenario $n: final identities")
                for ((name, vec) in known) assertVec(vec.floats(), tracker.embeddingOf(name), 2e-6f, "scenario $n final $name")

                val files = fin.getValue("files").jsonObject
                assertEquals(files.keys.toList(), store.list(), "scenario $n: final files")
                for ((file, vec) in files) assertVec(vec.floats(), store.read(file), 2e-6f, "scenario $n file $file")

                val expEvents = fin.getValue("events").jsonArray.map { it.jsonObject }
                val events = tracker.events()
                assertEquals(expEvents.size, events.size, "scenario $n: event count")
                for ((k, e) in expEvents.withIndex()) {
                    val a = events[k]
                    for ((key, v) in e) {
                        val actual = a[key]
                        when {
                            v is JsonNull -> assertEquals(null, actual, "scenario $n event $k $key")
                            key == "merge_score" -> assertNear(v.jsonPrimitive.double, actual as Double, 1e-3, "scenario $n event $k $key")
                            v is JsonPrimitive && v.isString -> assertEquals(v.content, actual, "scenario $n event $k $key")
                            else -> assertEquals(v.jsonPrimitive.int, (actual as Number).toInt(), "scenario $n event $k $key")
                        }
                    }
                }

                // the mirror into the database: who was enrolled, in which file, with which embedding
                val expectedDb = expected.getValue("db").jsonArray.map { call -> call.jsonArray.map { it.jsonPrimitive.content } }
                assertEquals(expectedDb.map { it.take(3) }, dbCalls.map { it.take(3).map(Any::toString) }, "scenario $n: database calls")
                for ((k, call) in expectedDb.withIndex()) {
                    if (call[0] == "face") assertNear(call[3].toDouble(), (dbCalls[k][3] as Double), 2e-3, "scenario $n db call $k embedding sum")
                }

                val metrics = tracker.sessionFaceMetrics()
                val em = fin.getValue("metrics").jsonObject
                for (key in listOf("total_tracks", "tracks_identified", "tracks_generic", "face_fn_count", "enrollments", "known_embeddings_count", "total_frames")) {
                    assertEquals(em.getValue(key).jsonPrimitive.int, metrics.getValue(key).jsonPrimitive.int, "scenario $n metrics $key")
                }
                assertEquals(em.getValue("samples").jsonPrimitive.int, metrics.getValue("match_scores_sample").jsonArray.size, "scenario $n sampled scores")
                val stats = metrics.getValue("match_score_stats").jsonObject
                for ((key, v) in em.getValue("stats").jsonObject) {
                    assertNear(v.jsonPrimitive.double, stats.getValue(key).jsonPrimitive.double, 1e-3, "scenario $n stats $key")
                }
            } finally {
                dir.deleteRecursively()
            }
        }
    }

    @Test
    fun npy_roundTripsAndReadsNumpyFiles() {
        val values = floatArrayOf(0.1f, -2.5f, 3.25e-5f, 1e10f)
        val bytes = Npy.write(values)
        assertTrue(values.contentEquals(Npy.read(bytes)))
        assertEquals(0, (bytes.size - values.size * 4) % 64, "header padded to a multiple of 64 like numpy")

        // files written by numpy itself
        assertTrue(floatArrayOf(0.5f, -1.25f, 3.0f).contentEquals(Npy.read(File("src/test/resources/npy/f32.npy").readBytes())))
        assertTrue(floatArrayOf(1.5f, 2.5f).contentEquals(Npy.read(File("src/test/resources/npy/f64.npy").readBytes())))
    }
}
