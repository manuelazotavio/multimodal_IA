package com.avtracker.mobile.tracking.bytetrack

/** A detection as the tracker takes it: (x1, y1, x2, y2, confidence, class), like the rows YOLOFaceDetector.detect returns. */
data class Detection(val x1: Double, val y1: Double, val x2: Double, val y2: Double, val confidence: Double, val cls: Double = 0.0)

/** One confirmed track in the tracker's output: boxmot returns rows (x1, y1, x2, y2, id, conf, cls, det_ind). */
data class TrackOutput(
    val x1: Double, val y1: Double, val x2: Double, val y2: Double,
    val id: Int, val confidence: Double, val cls: Double, val detIndex: Int
)

private enum class TrackState { New, Tracked, Lost, Removed }

private class STrack(det: Detection, detIndex: Int) {
    // xyxy -> xywh -> tlwh -> xyah, with boxmot's exact arithmetic
    private val xywh = doubleArrayOf((det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2, det.x2 - det.x1, det.y2 - det.y1)
    val xyah: DoubleArray = run {
        val t = xywh[0] - xywh[2] / 2.0
        val l = xywh[1] - xywh[3] / 2.0
        doubleArrayOf(t + xywh[2] / 2, l + xywh[3] / 2, xywh[2] / xywh[3], xywh[3])
    }

    var conf = det.confidence
    var cls = det.cls
    var detIndex = detIndex
    var id = 0
    var state = TrackState.New
    var isActivated = false
    var frameId = 0
    var startFrame = 0
    var mean: DoubleArray? = null
    var covariance: Array<DoubleArray>? = null
    private var kalman: KalmanFilterXyah? = null

    val endFrame: Int get() = frameId

    fun xyxy(): DoubleArray {
        val m = mean
        val r = if (m == null) xywh.copyOf() else doubleArrayOf(m[0], m[1], m[2] * m[3], m[3])
        return doubleArrayOf(r[0] - r[2] / 2, r[1] - r[3] / 2, r[0] + r[2] / 2, r[1] + r[3] / 2)
    }

    fun predict(filter: KalmanFilterXyah) {
        val m = mean!!.copyOf()
        if (state != TrackState.Tracked) m[7] = 0.0
        val (nm, nc) = filter.predict(m, covariance!!)
        mean = nm
        covariance = nc
    }

    fun activate(filter: KalmanFilterXyah, frame: Int, newId: Int) {
        kalman = filter
        id = newId
        val (m, c) = filter.initiate(xyah)
        mean = m
        covariance = c
        state = TrackState.Tracked
        if (frame == 1) isActivated = true
        frameId = frame
        startFrame = frame
    }

    fun reActivate(newTrack: STrack, frame: Int) {
        val (m, c) = kalman!!.update(mean!!, covariance!!, newTrack.xyah)
        mean = m
        covariance = c
        state = TrackState.Tracked
        isActivated = true
        frameId = frame
        conf = newTrack.conf
        cls = newTrack.cls
        detIndex = newTrack.detIndex
    }

    fun update(newTrack: STrack, frame: Int) {
        frameId = frame
        val (m, c) = kalman!!.update(mean!!, covariance!!, newTrack.xyah)
        mean = m
        covariance = c
        state = TrackState.Tracked
        isActivated = true
        conf = newTrack.conf
        cls = newTrack.cls
        detIndex = newTrack.detIndex
    }

    fun markLost() { state = TrackState.Lost }

    fun markRemoved() { state = TrackState.Removed }
}

/**
 * Port of boxmot's ByteTrack (the tracker PersonIDTracker builds with `track_buffer=150`): Kalman prediction,
 * a first association of high-confidence detections (IoU fused with the score, threshold 0.8), a second one of the
 * remaining tracks with low-confidence detections (0.5), unconfirmed tracks (0.7), and a lost-track buffer of
 * `track_buffer` frames. The bookkeeping (including how lost and removed lists are merged) follows the original literally.
 */
class ByteTrack(
    private val minConf: Double = 0.1,
    private val trackThresh: Double = 0.45,
    private val matchThresh: Double = 0.8,
    trackBuffer: Int = 25,
    frameRate: Int = 30
) {
    private val kalman = KalmanFilterXyah()
    private val maxTimeLost = (frameRate / 30.0 * trackBuffer).toInt()
    private var frameCount = 0
    private var nextId = 0

    private var activeTracks = ArrayList<STrack>()
    private var lostTracks = ArrayList<STrack>()
    private val removedTracks = ArrayList<STrack>()

    /** One tracker step. [detections] must not be empty (the caller skips frames without detections, as the Python does). */
    fun update(detections: List<Detection>): List<TrackOutput> {
        frameCount++
        val activated = ArrayList<STrack>()
        val refind = ArrayList<STrack>()
        val lost = ArrayList<STrack>()
        val removed = ArrayList<STrack>()

        // boxmot's setup_decorator casts the detections to float32 before tracking
        val all = detections.mapIndexed { i, d ->
            Detection(
                d.x1.toFloat().toDouble(), d.y1.toFloat().toDouble(), d.x2.toFloat().toDouble(), d.y2.toFloat().toDouble(),
                d.confidence.toFloat().toDouble(), d.cls.toFloat().toDouble()
            ) to i
        }
        val high = all.filter { it.first.confidence > trackThresh }
        val second = all.filter { it.first.confidence > minConf && it.first.confidence < trackThresh }

        var dets = high.map { STrack(it.first, it.second) }

        val unconfirmed = ArrayList<STrack>()
        val tracked = ArrayList<STrack>()
        for (t in activeTracks) if (!t.isActivated) unconfirmed += t else tracked += t

        // Step 2: first association with the high-confidence detections
        val pool = jointStracks(tracked, lostTracks)
        for (t in pool) t.predict(kalman)
        var dists = fuseScore(iouDistance(pool, dets), dets)
        var assignment = linearAssignment(dists, pool.size, dets.size, matchThresh)

        for ((it, id) in assignment.matches) {
            val track = pool[it]
            val det = dets[id]
            if (track.state == TrackState.Tracked) {
                track.update(det, frameCount)
                activated += track
            } else {
                track.reActivate(det, frameCount)
                refind += track
            }
        }

        // Step 3: second association of the still unmatched tracked tracks with the low-confidence detections
        val detsSecond = second.map { STrack(it.first, it.second) }
        val rTracked = assignment.unmatchedRows.map { pool[it] }.filter { it.state == TrackState.Tracked }
        dists = iouDistance(rTracked, detsSecond)
        val second2 = linearAssignment(dists, rTracked.size, detsSecond.size, 0.5)
        for ((it, id) in second2.matches) {
            val track = rTracked[it]
            val det = detsSecond[id]
            if (track.state == TrackState.Tracked) {
                track.update(det, frameCount)
                activated += track
            } else {
                track.reActivate(det, frameCount)
                refind += track
            }
        }
        for (it in second2.unmatchedRows) {
            val track = rTracked[it]
            if (track.state != TrackState.Lost) {
                track.markLost()
                lost += track
            }
        }

        // Unconfirmed tracks (usually only one frame old) against the detections left over
        dets = assignment.unmatchedCols.map { dets[it] }
        dists = fuseScore(iouDistance(unconfirmed, dets), dets)
        assignment = linearAssignment(dists, unconfirmed.size, dets.size, 0.7)
        for ((it, id) in assignment.matches) {
            unconfirmed[it].update(dets[id], frameCount)
            activated += unconfirmed[it]
        }
        for (it in assignment.unmatchedRows) {
            val track = unconfirmed[it]
            track.markRemoved()
            removed += track
        }

        // Step 4: new tracks
        for (it in assignment.unmatchedCols) {
            val track = dets[it]
            if (track.conf < trackThresh) continue
            track.activate(kalman, frameCount, ++nextId)
            activated += track
        }

        // Step 5: state
        for (track in lostTracks) {
            if (frameCount - track.endFrame > maxTimeLost) {
                track.markRemoved()
                removed += track
            }
        }

        activeTracks = ArrayList(activeTracks.filter { it.state == TrackState.Tracked })
        activeTracks = jointStracks(activeTracks, activated)
        activeTracks = jointStracks(activeTracks, refind)
        lostTracks = subStracks(lostTracks, activeTracks)
        lostTracks.addAll(lost)
        lostTracks = subStracks(lostTracks, removedTracks)
        removedTracks.addAll(removed)
        val (a, b) = removeDuplicateStracks(activeTracks, lostTracks)
        activeTracks = a
        lostTracks = b

        return activeTracks.filter { it.isActivated }.map {
            val box = it.xyxy()
            TrackOutput(box[0], box[1], box[2], box[3], it.id, it.conf, it.cls, it.detIndex)
        }
    }

    private fun iouDistance(a: List<STrack>, b: List<STrack>): Array<DoubleArray> {
        val boxesA = a.map { it.xyxy() }
        val boxesB = b.map { it.xyxy() }
        return Array(a.size) { i ->
            DoubleArray(b.size) { j -> 1 - iou(boxesA[i], boxesB[j]) }
        }
    }

    private fun iou(p: DoubleArray, q: DoubleArray): Double {
        val xx1 = maxOf(p[0], q[0])
        val yy1 = maxOf(p[1], q[1])
        val xx2 = minOf(p[2], q[2])
        val yy2 = minOf(p[3], q[3])
        val w = maxOf(0.0, xx2 - xx1)
        val h = maxOf(0.0, yy2 - yy1)
        val wh = w * h
        return wh / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - wh)
    }

    private fun fuseScore(cost: Array<DoubleArray>, dets: List<STrack>): Array<DoubleArray> {
        if (cost.isEmpty() || dets.isEmpty()) return cost
        return Array(cost.size) { i -> DoubleArray(dets.size) { j -> 1 - (1 - cost[i][j]) * dets[j].conf } }
    }

    private fun jointStracks(a: List<STrack>, b: List<STrack>): ArrayList<STrack> {
        val exists = HashSet<Int>()
        val res = ArrayList<STrack>()
        for (t in a) { exists += t.id; res += t }
        for (t in b) if (exists.add(t.id)) res += t
        return res
    }

    /** dict semantics: keyed by id (a repeated id keeps its first position, with the last value). */
    private fun subStracks(a: List<STrack>, b: List<STrack>): ArrayList<STrack> {
        val map = LinkedHashMap<Int, STrack>()
        for (t in a) map[t.id] = t
        for (t in b) map.remove(t.id)
        return ArrayList(map.values)
    }

    private fun removeDuplicateStracks(a: List<STrack>, b: List<STrack>): Pair<ArrayList<STrack>, ArrayList<STrack>> {
        val dist = iouDistance(a, b)
        val dupA = HashSet<Int>()
        val dupB = HashSet<Int>()
        for (p in a.indices) for (q in b.indices) {
            if (dist[p][q] < 0.15) {
                val timeP = a[p].frameId - a[p].startFrame
                val timeQ = b[q].frameId - b[q].startFrame
                if (timeP > timeQ) dupB += q else dupA += p
            }
        }
        return ArrayList(a.filterIndexed { i, _ -> i !in dupA }) to ArrayList(b.filterIndexed { i, _ -> i !in dupB })
    }
}
