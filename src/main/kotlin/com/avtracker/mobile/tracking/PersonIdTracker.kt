package com.avtracker.mobile.tracking

import com.avtracker.mobile.tracking.bytetrack.ByteTrack
import com.avtracker.mobile.tracking.bytetrack.Detection
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.math.BigDecimal
import java.math.RoundingMode
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

/** Supplies the (L2-normalised) EdgeFace embedding of the frame crop `[left, right) x [top, bottom)`. */
fun interface FaceEmbeddingSource {
    fun embed(left: Int, top: Int, right: Int, bottom: Int): FloatArray
}

/** One face as `PersonIDTracker.update` reports it: track id, the identity given to it, the match score and the box. */
data class FaceTrack(
    val trackId: Int,
    val name: String,
    val confidence: Double,
    val x1: Double, val y1: Double, val x2: Double, val y2: Double
)

/**
 * Port of av-tracker's PersonIDTracker (src/personid_tracker.py): ByteTrack for the face tracks, EdgeFace embeddings
 * matched against the enrolled faces (cosine >= [MATCH_THRESHOLD]), a per-track confirmation vote, "one name, one face"
 * claims, identity persistence with drop-out, auto-enrolment of unknown faces as `Person_N` (merging into an existing
 * identity above [MERGE_THRESHOLD], honouring the identity cap), the rename cycle driven by the voice side, the
 * consolidation of duplicate identities and the on-disk `.npy` bookkeeping.
 *
 * The Python's SQLite mirror of enrolments (`db.add_speaker`/`save_face_embedding`) is [onEnrolled]. Debug logging
 * goes to [log]. [clock] is `time.time()` in seconds, [timestamp] is `datetime.now().strftime("%Y%m%d_%H%M%S")`.
 * Every public method is synchronised: the camera thread updates while the fusion thread renames.
 */
class PersonIdTracker(
    private val store: FaceEmbeddingStore? = null,
    private val maxIdentities: Int? = null,
    private val clock: () -> Double = { System.currentTimeMillis() / 1000.0 },
    private val timestamp: () -> String = { SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date()) },
    private val log: (String) -> Unit = {},
    /** Called for every newly auto-enrolled face (the Python's `db.add_speaker` + `db.save_face_embedding`). */
    private val onEnrolled: ((name: String, embedding: FloatArray, sourceFile: String) -> Unit)? = null
) {
    private val byteTrack = ByteTrack(trackBuffer = TRACK_BUFFER)

    /** name -> embedding (Python `known_embeddings`). */
    private val knownEmbeddings = LinkedHashMap<String, FloatArray>()
    private val embFiles = LinkedHashMap<String, String>()

    private val trackBuffer = HashMap<Int, ArrayList<FloatArray>>()
    private val trackToName = LinkedHashMap<Int, String>()
    private var personCounter = 0
    private val persistFailCount = HashMap<Int, Int>()
    private val nameConfirm = LinkedHashMap<Int, Confirmation>()
    private val overrideLastFrame = HashMap<Int, Double>()
    private val trackLastSeen = HashMap<Int, Double>()
    private var frameCount = 0

    private val faceEvents = ArrayList<Map<String, Any?>>()
    private val matchScoresSample = ArrayList<Map<String, Any?>>()
    private var sampleCounter = 0

    private class Confirmation(var name: String, var count: Int, var avgScore: Double)

    // ---- read access for the other layers ----

    @get:Synchronized val knownNames: Set<String> get() = LinkedHashSet(knownEmbeddings.keys)

    @Synchronized fun knowsFace(name: String) = name in knownEmbeddings

    /** The identity confirmed for [trackId] (Python `_track_to_name.get`). */
    @Synchronized fun trackName(trackId: Int): String? = trackToName[trackId]

    // ---- loading ----

    /** Python `load_known_embeddings`: every `.npy` of the store, generic names skipped, averaged per person. */
    @Synchronized
    fun loadKnownEmbeddings() {
        val store = store ?: return
        embFiles.clear()
        val raw = LinkedHashMap<String, ArrayList<FloatArray>>()
        for (file in store.list()) {
            var base = file.removeSuffix(".npy")
            if (base.endsWith("_auto")) base = base.substring(0, base.length - 5)
            val parts = base.split('_')
            val name = if (parts.size >= 3 && isDigits(parts.last()) && isDigits(parts[parts.size - 2])) {
                parts.subList(0, parts.size - 2).joinToString("_")
            } else base
            if (GENERIC_LOAD.matches(name)) continue
            val emb = store.read(file) ?: continue
            raw.getOrPut(name) { ArrayList() } += emb
            embFiles[name] = file
        }
        for ((name, embs) in raw) {
            val avg = mean(embs)
            val norm = norm(avg)
            knownEmbeddings[name] = if (norm > 0) FloatArray(avg.size) { avg[it] / norm } else avg
        }
        consolidateEmbeddings()
        for (name in knownEmbeddings.keys) {
            if (name.startsWith("Person_")) {
                name.substringAfterLast('_').toIntOrNull()?.let { personCounter = maxOf(personCounter, it) }
            }
        }
    }

    /**
     * Merges known embeddings that are too similar, only when at least one side is a generic `Person_N`. Two real
     * names are never merged. (The file handling follows the original, which looks the files up by `name_b`/`name_a`.)
     */
    private fun consolidateEmbeddings() {
        val names = knownEmbeddings.keys.toList()
        val mergedInto = HashMap<String, String>()

        for ((i, nameA) in names.withIndex()) {
            if (nameA in mergedInto) continue
            for (nameB in names.subList(i + 1, names.size)) {
                if (nameB in mergedInto) continue
                val aGeneric = GENERIC_PERSON.matches(nameA)
                val bGeneric = GENERIC_PERSON.matches(nameB)
                if (!aGeneric && !bGeneric) continue
                val embA = knownEmbeddings.getValue(nameA)
                val embB = knownEmbeddings.getValue(nameB)
                val sim = dot(embA, embB)
                if (sim >= MERGE_THRESHOLD) {
                    val (canon, dup) = when {
                        bGeneric && !aGeneric -> nameA to nameB
                        aGeneric && !bGeneric -> nameB to nameA
                        else -> nameA to nameB
                    }
                    val canonEmb = knownEmbeddings.getValue(canon)
                    val dupEmb = knownEmbeddings.getValue(dup)
                    val avg = FloatArray(canonEmb.size) { (canonEmb[it] + dupEmb[it]) / 2 }
                    val avgNorm = norm(avg) + 1e-8f
                    val normalised = FloatArray(avg.size) { avg[it] / avgNorm }
                    knownEmbeddings[canon] = normalised
                    mergedInto[dup] = canon
                    log("[PersonIDTracker] Consolidated '$dup' -> '$canon' (sim=${"%.3f".format(Locale.ROOT, sim)})")
                    store?.let { s ->
                        embFiles[nameB]?.let { dupFile -> if (s.exists(dupFile)) s.delete(dupFile) }
                        embFiles[nameA]?.let { canonFile -> s.write(canonFile, normalised) }
                    }
                }
            }
        }
        for (name in mergedInto.keys) {
            knownEmbeddings.remove(name)
            embFiles.remove(name)
        }
    }

    // ---- matching ----

    private fun bestMatch(emb: FloatArray, trackId: Int): Pair<String, Double> {
        val scores = knownEmbeddings.map { (name, known) -> name to dot(emb, known) }.sortedByDescending { it.second }
        val (bestName, bestScore) = scores.firstOrNull() ?: ("Unknown" to -1.0)
        sampleCounter++
        if (sampleCounter % 30 == 0 && scores.isNotEmpty()) {
            matchScoresSample += linkedMapOf(
                "ts" to clock(),
                "track_id" to trackId,
                "best_name" to bestName,
                "best_score" to round4(bestScore),
                "top3" to scores.take(3).map { listOf(it.first, round4(it.second)) },
                "matched" to (bestScore >= MATCH_THRESHOLD)
            )
        }
        return if (bestScore < MATCH_THRESHOLD) "Unknown" to bestScore else bestName to bestScore
    }

    private fun enroll(trackId: Int, embeddings: List<FloatArray>): String {
        val mean = mean(embeddings)
        val meanNorm = norm(mean) + 1e-8f
        val avgEmb = FloatArray(mean.size) { mean[it] / meanNorm }

        // Names held by another LIVE track (seen in the last 2 s): two faces visible at once must not share one identity.
        val now = clock()
        val liveNames = trackToName.filter { (tid, _) -> tid != trackId && (now - (trackLastSeen[tid] ?: 0.0)) < 2.0 }
            .values.toSet()

        var mergeName: String? = null
        var mergeScore = -1.0
        for ((name, known) in knownEmbeddings) {
            val score = dot(avgEmb, known)
            if (score > mergeScore) {
                mergeScore = score
                mergeName = name
            }
        }
        val mergeBlocked = mergeName != null && mergeName in liveNames
        if (mergeName != null && mergeScore >= MERGE_THRESHOLD && !mergeBlocked) {
            trackToName[trackId] = mergeName
            return mergeName
        }

        // Identity cap: never exceed the expected speaker count; force-merge into the closest identity that is not live.
        if (maxIdentities != null) {
            val existing = LinkedHashSet(trackToName.values)
            if (existing.size >= maxIdentities && !mergeBlocked) {
                val cand = existing.filter { it in knownEmbeddings && it !in liveNames }
                if (cand.isNotEmpty()) {
                    val forced = cand.maxByOrNull { dot(avgEmb, knownEmbeddings.getValue(it)) }!!
                    val forcedScore = dot(avgEmb, knownEmbeddings.getValue(forced))
                    trackToName[trackId] = forced
                    faceEvents += linkedMapOf(
                        "ts" to clock(), "event" to "enroll", "track_id" to trackId, "name" to forced,
                        "merge_target" to forced, "merge_score" to round4(forcedScore)
                    )
                    return forced
                }
            }
        }

        personCounter++
        val name = "Person_$personCounter"
        knownEmbeddings[name] = avgEmb
        trackToName[trackId] = name

        // Saved so a later rename can promote it; loading skips generic names, so it never competes across sessions.
        store?.let { s ->
            val filename = "${name}_auto.npy"
            s.write(filename, avgEmb)
            embFiles[name] = filename
            log("[PersonIDTracker] Auto-enrolled: $name (track $trackId)")
            onEnrolled?.invoke(name, avgEmb, filename)
        }

        faceEvents += linkedMapOf(
            "ts" to clock(), "event" to "enroll", "track_id" to trackId, "name" to name,
            "merge_target" to mergeName, "merge_score" to round4(mergeScore)
        )
        return name
    }

    // ---- rename cycle ----

    /**
     * Renames an identity in memory and on disk once the real name is known (from the voice).
     * [onlyTrackId] renames just that track and keeps the old embedding for other tracks that still use it.
     */
    @Synchronized
    fun renamePerson(oldName: String, newName: String, onlyTrackId: Int? = null) {
        if (oldName !in knownEmbeddings) return
        if (onlyTrackId != null) {
            val otherTracksUse = trackToName.any { (tid, tname) -> tid != onlyTrackId && tname == oldName }
            if (otherTracksUse) {
                knownEmbeddings[newName] = knownEmbeddings.getValue(oldName).copyOf()
                trackToName[onlyTrackId] = newName
            } else {
                knownEmbeddings[newName] = knownEmbeddings.remove(oldName)!!
                trackToName[onlyTrackId] = newName
            }
        } else {
            knownEmbeddings[newName] = knownEmbeddings.remove(oldName)!!
            for (tid in trackToName.keys.toList()) if (trackToName[tid] == oldName) trackToName[tid] = newName
        }

        // Timestamped file names accumulate embeddings from different sessions (diversity helps cross-session matching).
        store?.let { s ->
            val oldFile = embFiles.remove(oldName)
            val newFile = "${newName}_${timestamp()}_auto.npy"
            if (oldFile != null) {
                if (s.exists(oldFile)) s.rename(oldFile, newFile)
            } else if (newName in knownEmbeddings) {
                s.write(newFile, knownEmbeddings.getValue(newName))
            }
            embFiles[newName] = newFile
            val allFiles = s.list().filter { it.startsWith(newName) }.sorted()
            if (allFiles.size > MAX_FACE_FILES) {
                for (old in allFiles.subList(0, allFiles.size - MAX_FACE_FILES)) s.delete(old)
            }
        }

        // Orphan Person_N files: nobody uses the generic name any more.
        if (store != null && oldName.isNotEmpty() && GENERIC_PERSON.matches(oldName)) {
            if (trackToName.values.none { it == oldName }) {
                for (f in store.list()) {
                    val base = if ('.' in f) f.substringBeforeLast('.') else f
                    if (base == oldName || base.startsWith(oldName + "_")) store.delete(f)
                }
                knownEmbeddings.remove(oldName)
            }
        }

        val oldGeneric = GENERIC_OR_UNKNOWN.matches(oldName)
        val newReal = !GENERIC_OR_UNKNOWN.matches(newName)
        if (oldGeneric && newReal) {
            faceEvents += linkedMapOf(
                "ts" to clock(), "event" to "face_fn_rename", "old_name" to oldName, "new_name" to newName,
                "track_id" to onlyTrackId
            )
        }
        log("[PersonIDTracker] Renamed '$oldName' -> '$newName'")
        consolidateEmbeddings()
        for (tid in trackToName.keys.toList()) if (trackToName.getValue(tid) !in knownEmbeddings) trackToName.remove(tid)
    }

    // ---- per-frame update ----

    /**
     * Tracks [detections] (x1, y1, x2, y2, conf, class) in a [frameWidth] x [frameHeight] frame and returns every
     * face with its identity. A frame without detections returns nothing and does not advance the tracker.
     */
    @Synchronized
    fun update(frameWidth: Int, frameHeight: Int, detections: List<Detection>, source: FaceEmbeddingSource): List<FaceTrack> {
        if (detections.isEmpty()) return emptyList()

        val tracks = byteTrack.update(detections)
        val results = ArrayList<FaceTrack>()

        // A person cannot appear in two places at once: names already claimed in this frame.
        val claimed = LinkedHashMap<String, Int>()

        // Pre-reserve real names of recently seen tracks so another track does not steal the name while the person is
        // briefly out of view (30 s for names with a stored embedding, 5 s for names enrolled in this session).
        val activeIds = tracks.map { it.id }.toSet()
        val now = clock()
        for ((tid, tname) in trackToName) {
            if (GENERIC_OR_UNKNOWN.matches(tname)) continue
            val lastSeen = trackLastSeen[tid] ?: 0.0
            val protection = if (tname in knownEmbeddings) 30.0 else 5.0
            if (tid in activeIds || (now - lastSeen) < protection) claimed[tname] = tid
        }

        for (track in tracks) {
            val trackId = track.id
            val rows = sliceBounds(track.y1.toInt(), track.y2.toInt(), frameHeight)
            val cols = sliceBounds(track.x1.toInt(), track.x2.toInt(), frameWidth)
            if (rows == null || cols == null) continue

            val current = source.embed(cols.first, rows.first, cols.second, rows.second)
            trackLastSeen[trackId] = clock()
            var (bestName, bestScore) = bestMatch(current, trackId)

            if (bestName != UNKNOWN) {
                if (bestName in claimed && claimed[bestName] != trackId) {
                    // Same name on two faces: this one is unknown.
                    bestName = UNKNOWN
                    bestScore = 0.0
                } else if (trackToName[trackId] == bestName) {
                    claimed[bestName] = trackId
                    trackBuffer.remove(trackId)
                    persistFailCount.remove(trackId)
                    nameConfirm.remove(trackId)
                } else {
                    // Block re-confirmation while the current name is still plausible, unless the new match is much stronger.
                    val currentConfirmed = trackToName[trackId]
                    if (currentConfirmed != null && currentConfirmed in knownEmbeddings) {
                        val persistSim = dot(current, knownEmbeddings.getValue(currentConfirmed))
                        val overrideCooldownOk = clock() - (overrideLastFrame[trackId] ?: 0.0) > 3.0
                        val strongNew = bestScore >= 0.75 && persistSim < 0.35 && overrideCooldownOk
                        // A weak Person_N (< 0.45) may be replaced by a real name at the normal threshold.
                        val genericUpgrade = GENERIC_OR_UNKNOWN.matches(currentConfirmed) && !GENERIC_OR_UNKNOWN.matches(bestName) &&
                            bestScore >= MATCH_THRESHOLD && persistSim < 0.45 && overrideCooldownOk
                        if (persistSim >= 0.20 && !strongNew && !genericUpgrade) {
                            bestName = UNKNOWN
                            bestScore = 0.0
                        } else if (strongNew || genericUpgrade) {
                            overrideLastFrame[trackId] = clock()
                            trackToName.remove(trackId)
                            // Person_N -> real name: the generic embedding was absorbed by the real identity.
                            if (GENERIC_PERSON.matches(currentConfirmed) && !GENERIC_PERSON.matches(bestName)) {
                                knownEmbeddings.remove(currentConfirmed)
                                store?.let { s ->
                                    for (f in s.list()) {
                                        val base = if ('.' in f) f.substringBeforeLast('.') else f
                                        if (base == currentConfirmed || base.startsWith(currentConfirmed + "_")) s.delete(f)
                                    }
                                }
                            }
                        }
                    }

                    // Not yet confirmed: accumulate votes before promoting.
                    if (bestName != UNKNOWN) {
                        // Only the best face may claim a unique identity: block if another pending track with the same
                        // real name has a higher average score.
                        var dominated = false
                        if (!GENERIC_OR_UNKNOWN.matches(bestName)) {
                            for ((other, pending) in nameConfirm) {
                                if (other != trackId && pending.name == bestName && pending.avgScore > bestScore) {
                                    dominated = true
                                    break
                                }
                            }
                        }

                        if (dominated) {
                            bestName = UNKNOWN
                            bestScore = 0.0
                        } else {
                            val buf = nameConfirm[trackId]
                            val vote = if (buf != null && buf.name == bestName) {
                                buf.count++
                                buf.avgScore = (buf.avgScore * (buf.count - 1) + bestScore) / buf.count
                                buf
                            } else {
                                Confirmation(bestName, 1, bestScore).also { nameConfirm[trackId] = it }
                            }

                            if (vote.count >= CONFIRM_FRAMES) {
                                // Evict any other track pending with the same name: they lost the race.
                                for (other in nameConfirm.keys.toList()) {
                                    if (other != trackId && nameConfirm.getValue(other).name == bestName) nameConfirm.remove(other)
                                }
                                claimed[bestName] = trackId
                                trackToName[trackId] = bestName
                                trackBuffer.remove(trackId)
                                persistFailCount.remove(trackId)
                                nameConfirm.remove(trackId)
                            } else {
                                // Still awaiting confirmation.
                                bestName = UNKNOWN
                                bestScore = 0.0
                            }
                        }
                    }
                }
            }

            if (bestName == UNKNOWN && trackId in trackToName) {
                val prevName = trackToName.getValue(trackId)
                // Is the embedding still compatible with the persisted name? Prevents a wrong name from persisting forever.
                var persistOk = true
                val prevEmb = knownEmbeddings[prevName]
                if (prevEmb != null) {
                    val persistSim = dot(current, prevEmb)
                    if (persistSim < 0.20) {
                        val failCount = (persistFailCount[trackId] ?: 0) + 1
                        persistFailCount[trackId] = failCount
                        if (failCount >= 3) {
                            // 3 consecutive bad frames: drop the identity.
                            persistOk = false
                            trackToName.remove(trackId)
                            persistFailCount.remove(trackId)
                            nameConfirm.remove(trackId)
                        }
                    } else {
                        persistFailCount.remove(trackId)
                    }
                }
                // Keep the previous name only if no other track uses it.
                if (persistOk && (prevName !in claimed || claimed[prevName] == trackId)) {
                    bestName = prevName
                    bestScore = 1.0
                    claimed[bestName] = trackId
                }
            }

            if (bestName == UNKNOWN) {
                // Buffer for auto-enrolment.
                val buf = trackBuffer.getOrPut(trackId) { ArrayList() }
                buf += current
                val n = buf.size
                if (n >= ENROLL_FRAMES) {
                    val diverseEnough = bufferDiversity(buf) >= ENROLL_MIN_DIVERSITY
                    val forced = n >= ENROLL_MAX_FRAMES
                    if (diverseEnough || forced) {
                        bestName = enroll(trackId, buf)
                        bestScore = 1.0
                        trackBuffer.remove(trackId)
                        claimed[bestName] = trackId
                    }
                }
            }

            results += FaceTrack(trackId, bestName, bestScore, track.x1, track.y1, track.x2, track.y2)
        }

        frameCount++
        return results
    }

    // ---- session metrics ----

    /** Python `get_session_face_metrics`: structured face-recognition metrics for the session files. */
    @Synchronized
    fun sessionFaceMetrics(): JsonObject {
        val enrollments = faceEvents.filter { it["event"] == "enroll" }
        val faceFns = faceEvents.filter { it["event"] == "face_fn_rename" }

        val realNames = trackToName.filterValues { !GENERIC_OR_UNKNOWN.matches(it) }
        val genericNames = trackToName.filterValues { GENERIC_OR_UNKNOWN.matches(it) }

        val allScores = matchScoresSample.map { it["best_score"] as Double }
        val matched = matchScoresSample.filter { it["matched"] == true }.map { it["best_score"] as Double }
        val unmatched = matchScoresSample.filter { it["matched"] != true }.map { it["best_score"] as Double }

        fun mean(v: List<Double>) = if (v.isEmpty()) 0.0 else round4(v.average())
        fun median(v: List<Double>): Double {
            if (v.isEmpty()) return 0.0
            val s = v.sorted()
            return round4(if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2)
        }

        return anyToJson(
            linkedMapOf(
                "total_tracks" to trackToName.size,
                "tracks_identified" to realNames.size,
                "tracks_generic" to genericNames.size,
                "face_fn_count" to faceFns.size,
                "face_fn_events" to faceFns,
                "enrollments" to enrollments.size,
                "match_threshold" to MATCH_THRESHOLD,
                "merge_threshold" to MERGE_THRESHOLD,
                "known_embeddings_count" to knownEmbeddings.size,
                "total_frames" to frameCount,
                "match_score_stats" to linkedMapOf(
                    "n_samples" to allScores.size,
                    "mean" to mean(allScores),
                    "median" to median(allScores),
                    "max" to if (allScores.isEmpty()) 0.0 else round4(allScores.max()),
                    "min" to if (allScores.isEmpty()) 0.0 else round4(allScores.min()),
                    "pct_matched" to if (allScores.isEmpty()) 0.0 else round4(matched.size.toDouble() / allScores.size),
                    "mean_matched" to mean(matched),
                    "mean_unmatched" to mean(unmatched)
                ),
                "match_scores_sample" to matchScoresSample,
                "events" to faceEvents
            )
        ) as JsonObject
    }

    /** State for tests: identity per track and the enrolled embeddings. */
    @Synchronized fun trackNames(): Map<Int, String> = LinkedHashMap(trackToName)

    @Synchronized fun embeddingOf(name: String): FloatArray? = knownEmbeddings[name]?.copyOf()

    @Synchronized fun events(): List<Map<String, Any?>> = faceEvents.toList()

    companion object {
        const val MATCH_THRESHOLD = 0.65
        const val MERGE_THRESHOLD = 0.55
        const val ENROLL_FRAMES = 40
        const val ENROLL_MAX_FRAMES = 120
        const val ENROLL_MIN_DIVERSITY = 0.05
        const val CONFIRM_FRAMES = 5
        const val MAX_FACE_FILES = 8

        /** ByteTrack(track_buffer=150): 5 s at 30 fps, long enough to survive someone looking down at notes. */
        const val TRACK_BUFFER = 150

        const val UNKNOWN = "Unknown"

        private val GENERIC_LOAD = Regex("^(Person_\\d+|Unknown(_\\d+)?)$")
        private val GENERIC_OR_UNKNOWN = Regex("^(Person_\\d+|Unknown)$")
        private val GENERIC_PERSON = Regex("^Person_\\d+$")

        private fun isDigits(s: String) = s.isNotEmpty() && s.all { it.isDigit() }

        /** numpy slice `a[start:stop]` on an axis of length [n]: negative bounds count from the end, both clamp; null if empty. */
        internal fun sliceBounds(start: Int, stop: Int, n: Int): Pair<Int, Int>? {
            var s = start
            var e = stop
            if (s < 0) { s += n; if (s < 0) s = 0 } else if (s > n) s = n
            if (e < 0) { e += n; if (e < 0) e = 0 } else if (e > n) e = n
            return if (e > s) s to e else null
        }

        private fun dot(a: FloatArray, b: FloatArray): Double {
            var sum = 0.0
            for (i in a.indices) sum += a[i].toDouble() * b[i]
            return sum.toFloat().toDouble() // numpy float32 dot
        }

        private fun norm(a: FloatArray): Float {
            var sum = 0.0
            for (v in a) sum += v.toDouble() * v
            return sqrt(sum).toFloat()
        }

        private fun mean(vs: List<FloatArray>): FloatArray {
            val out = FloatArray(vs[0].size)
            for (v in vs) for (i in out.indices) out[i] += v[i]
            for (i in out.indices) out[i] /= vs.size
            return out
        }

        /** Mean pairwise cosine distance among the buffered (normalised) embeddings: 0 = identical. */
        private fun bufferDiversity(embs: List<FloatArray>): Double {
            val n = embs.size
            if (n < 2) return 0.0
            var total = 0.0
            for (i in 0 until n) for (j in 0 until n) if (i != j) total += dot(embs[i], embs[j])
            return 1.0 - total / (n.toDouble() * (n - 1))
        }

        private fun round4(x: Double): Double = BigDecimal(x).setScale(4, RoundingMode.HALF_EVEN).toDouble()

        private fun anyToJson(v: Any?): JsonElement = when (v) {
            null -> JsonNull
            is JsonElement -> v
            is Boolean -> JsonPrimitive(v)
            is Number -> JsonPrimitive(v)
            is String -> JsonPrimitive(v)
            is Map<*, *> -> JsonObject(v.entries.associate { (k, x) -> k.toString() to anyToJson(x) })
            is Iterable<*> -> JsonArray(v.map { anyToJson(it) })
            else -> JsonPrimitive(v.toString())
        }
    }
}
