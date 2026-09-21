package com.avtracker.mobile.diarization

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.fusion.TurnDetector
import com.avtracker.mobile.voice.AssetFiles
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * The WeSpeaker ResNet34 speaker embedding of pyannote 3.1 (scripts/export_wespeaker.py): 10 s of audio and the activity
 * mask of one local speaker (589 frames) give a 256-d embedding. Fbank, mean subtraction and the weighted statistics
 * pooling are inside the graph.
 */
class WespeakerEmbedder(private val env: OrtEnvironment, private val session: OrtSession) : AutoCloseable {

    fun embed(waveform: FloatArray, mask: FloatArray): FloatArray {
        val wave = OnnxTensor.createTensor(env, FloatBuffer.wrap(waveform), longArrayOf(1, waveform.size.toLong()))
        val weights = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), longArrayOf(1, mask.size.toLong()))
        try {
            session.run(mapOf("waveform" to wave, "weights" to weights)).use { result ->
                @Suppress("UNCHECKED_CAST")
                return (result[0].value as Array<FloatArray>)[0]
            }
        } finally {
            wave.close()
            weights.close()
        }
    }

    override fun close() = session.close()

    companion object {
        fun fromFile(file: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): WespeakerEmbedder =
            WespeakerEmbedder(env, env.createSession(file.absolutePath, OrtSession.SessionOptions()))

        fun fromAssets(context: Context, assetPath: String): WespeakerEmbedder = fromFile(AssetFiles.materialize(context, assetPath))
    }
}

/** What one diarization run produced, for tests: the per-frame speaker count, the discrete diarization and the turns. */
class DiarizationResult(val count: IntArray, val discrete: Array<IntArray>, val turns: List<SpeakerTurn>)

/**
 * pyannote/speaker-diarization-3.1 as av-tracker runs it (`self.pipeline(...)` on every 8-15 s window): powerset
 * segmentation on 10 s windows sliding by 1 s, the instantaneous speaker count, WeSpeaker embeddings of every active
 * (window, local speaker) with the overlapped speech excluded, agglomerative clustering (centroid linkage, threshold 0.6,
 * minimum cluster size 12) constrained to [numSpeakers] when known, and the reconstruction of the discrete diarization.
 *
 * Embeddings of speakers that are not active in a window are not computed: the pipeline discards their clusters anyway.
 * Turn `slot` is the rank of the cluster among those present, i.e. SPEAKER_00 -> 0, SPEAKER_01 -> 1.
 */
class PyannoteDiarizer(
    private val segmentation: Diarizer,
    private val embedder: WespeakerEmbedder,
    private val numSpeakers: Int? = null,
    private val clustering: AgglomerativeClustering = AgglomerativeClustering(threshold = CLUSTERING_THRESHOLD, minClusterSize = MIN_CLUSTER_SIZE),
    private val minDurationOff: Double = 0.0
) : TurnDetector, AutoCloseable {

    override fun turns(audio: FloatArray): List<SpeakerTurn> = diarize(audio).turns

    fun diarize(audio: FloatArray): DiarizationResult {
        val empty = DiarizationResult(IntArray(0), emptyArray(), emptyList())
        if (audio.isEmpty()) return empty

        // ---- segmentation: 10 s windows every 1 s, the last one zero-padded ----
        val numComplete = if (audio.size >= WINDOW) (audio.size - WINDOW) / STEP + 1 else 0
        val hasLast = audio.size < WINDOW || (audio.size - WINDOW) % STEP > 0
        val numChunks = numComplete + if (hasLast) 1 else 0

        val chunks = Array(numChunks) { c -> crop(audio, c * STEP) }
        val seg = Array(numChunks) { c -> multilabel(segmentation.logProbabilities(chunks[c])) } // [chunk][frame][speaker]

        // ---- instantaneous speaker count ----
        val numFrames = closestFrame(WINDOW_SEC + (numChunks - 1) * STEP_SEC) + 1
        val sum = FloatArray(numFrames)
        val weight = FloatArray(numFrames)
        for (c in 0 until numChunks) {
            val start = closestFrame(c * STEP_SEC)
            for (f in 0 until FRAMES) {
                var s = 0f
                for (k in 0 until SPEAKERS) s += seg[c][f][k]
                sum[start + f] += s
                weight[start + f] += 1f
            }
        }
        val count = IntArray(numFrames) { Math.rint((sum[it] / max(weight[it], EPSILON)).toDouble()).toInt() }
        if ((count.maxOrNull() ?: 0) == 0) return DiarizationResult(count, emptyArray(), emptyList())

        // ---- embeddings of the active (window, speaker) pairs ----
        val minNumFrames = ceil(FRAMES * MIN_NUM_SAMPLES.toDouble() / WINDOW).toInt()
        val embeddings = ArrayList<Array<FloatArray?>>()
        val active = Array(numChunks) { BooleanArray(SPEAKERS) }
        for (c in 0 until numChunks) {
            val row = arrayOfNulls<FloatArray>(SPEAKERS)
            for (s in 0 until SPEAKERS) {
                var total = 0f
                for (f in 0 until FRAMES) total += seg[c][f][s]
                active[c][s] = total > 0
                if (!active[c][s]) continue

                // exclude_overlap: frames with two speakers are zeroed, unless too little speech is left
                val mask = FloatArray(FRAMES) { f -> seg[c][f][s] }
                val clean = FloatArray(FRAMES) { f ->
                    var others = 0f
                    for (k in 0 until SPEAKERS) others += seg[c][f][k]
                    if (others < 2) seg[c][f][s] else 0f
                }
                val used = if (clean.sum() > minNumFrames) clean else mask
                row[s] = embedder.embed(chunks[c], used)
            }
            embeddings += row
        }

        // ---- clustering, constrained to the number of speakers when the user gave it ----
        val hard = clustering.cluster(embeddings, numClusters = numSpeakers, minClusters = numSpeakers, maxClusters = numSpeakers)
        val numClusters = (hard.maxOf { it.maxOrNull() ?: -2 }) + 1

        val cap = numSpeakers
        if (cap != null) for (i in count.indices) count[i] = min(count[i], cap)

        // ---- reconstruct: max over the local speakers of a cluster, summed over the windows, top-`count` per frame ----
        val columns = max(numClusters, count.maxOrNull() ?: 0)
        val activations = Array(numFrames) { FloatArray(columns) }
        for (c in 0 until numChunks) {
            val start = closestFrame(c * STEP_SEC)
            for (k in hard[c].filter { it != -2 }.toSet()) {
                for (f in 0 until FRAMES) {
                    var best = Float.NEGATIVE_INFINITY
                    for (s in 0 until SPEAKERS) if (hard[c][s] == k) best = max(best, seg[c][f][s])
                    activations[start + f][k] += best
                }
            }
        }
        val discrete = Array(numFrames) { IntArray(columns) }
        for (t in 0 until numFrames) {
            val order = (0 until columns).sortedBy { -activations[t][it] } // stable: equal activations keep the lower index
            for (i in 0 until count[t]) discrete[t][order[i]] = 1
        }

        return DiarizationResult(count, discrete, toTurns(discrete, columns))
    }

    /** `Binarize(onset=0.5, offset=0.5, min_duration_off)` over the frame midpoints, then SPEAKER_NN ranks by cluster. */
    private fun toTurns(discrete: Array<IntArray>, columns: Int): List<SpeakerTurn> {
        val frames = discrete.size
        val timestamps = DoubleArray(frames) { it * FRAME_SEC + FRAME_SEC / 2 }
        val regions = ArrayList<Triple<Double, Double, Int>>()

        for (k in 0 until columns) {
            val perSpeaker = ArrayList<DoubleArray>()
            var start = timestamps[0]
            var isActive = discrete[0][k] > 0.5
            var t = timestamps[0]
            for (i in 1 until frames) {
                t = timestamps[i]
                val y = discrete[i][k].toDouble()
                if (isActive) {
                    if (y < 0.5) { perSpeaker += doubleArrayOf(start, t); start = t; isActive = false }
                } else if (y > 0.5) {
                    start = t
                    isActive = true
                }
            }
            if (isActive) perSpeaker += doubleArrayOf(start, t)

            // Annotation.support(collar): merge regions of one speaker that are closer than min_duration_off
            val merged = ArrayList<DoubleArray>()
            for (r in perSpeaker.sortedBy { it[0] }) {
                val last = merged.lastOrNull()
                if (minDurationOff > 0 && last != null && r[0] - last[1] <= minDurationOff) last[1] = max(last[1], r[1]) else merged += r
            }
            for (r in merged) if (r[1] - r[0] > SEGMENT_PRECISION) regions += Triple(r[0], r[1], k)
        }

        val present = regions.map { it.third }.toSortedSet().toList()
        return regions.map { (s, e, k) -> SpeakerTurn(s, e, present.indexOf(k)) }
            .sortedWith(compareBy({ it.startSec }, { it.endSec }, { it.slot }))
    }

    override fun close() {
        segmentation.close()
        embedder.close()
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val WINDOW = 160_000
        const val STEP = 16_000
        const val WINDOW_SEC = 10.0
        const val STEP_SEC = 1.0
        const val FRAMES = 589
        const val SPEAKERS = 3
        const val FRAME_SEC = WINDOW_SEC / FRAMES
        const val CLUSTERING_THRESHOLD = 0.6
        const val MIN_CLUSTER_SIZE = 12
        private const val MIN_NUM_SAMPLES = 400
        private const val EPSILON = 1e-12f
        private const val SEGMENT_PRECISION = 1e-6

        /** SlidingWindow.closest_frame for frames of `FRAME_SEC` starting at 0. */
        internal fun closestFrame(t: Double): Int = Math.rint((t - FRAME_SEC / 2) / FRAME_SEC).toInt()

        private fun crop(audio: FloatArray, start: Int): FloatArray {
            val out = FloatArray(WINDOW)
            val n = min(WINDOW, audio.size - start)
            if (n > 0) System.arraycopy(audio, start, out, 0, n)
            return out
        }

        /** Powerset log-probabilities -> hard multi-label activity (`to_multilabel(soft=False)`): [frame][speaker]. */
        internal fun multilabel(logProbs: Array<FloatArray>): Array<FloatArray> = Array(logProbs.size) { f ->
            var best = 0
            for (c in 1 until Diarizer.POWERSET.size) if (logProbs[f][c] > logProbs[f][best]) best = c
            FloatArray(SPEAKERS) { s -> if (Diarizer.POWERSET[best][s]) 1f else 0f }
        }

        fun fromAssets(context: Context, segmentation: Diarizer, numSpeakers: Int?): PyannoteDiarizer =
            PyannoteDiarizer(segmentation, WespeakerEmbedder.fromAssets(context, "models/wespeaker_resnet34.onnx"), numSpeakers)
    }
}
