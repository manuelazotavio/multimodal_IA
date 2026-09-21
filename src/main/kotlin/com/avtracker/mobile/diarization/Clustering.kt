package com.avtracker.mobile.diarization

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * pyannote.audio's `AgglomerativeClustering` (metric cosine, method centroid) as the speaker-diarization-3.1 pipeline uses
 * it, with scipy's `linkage(method="centroid")`, `fcluster(criterion="distance")` and `cdist` reproduced.
 *
 * `embeddings[c][s]` is the embedding of local speaker `s` in chunk `c`, or null when that pair is inactive: the
 * pipeline overwrites the result of inactive speakers with -2, so their embeddings never matter and are not computed.
 */
class AgglomerativeClustering(
    private val threshold: Double = 0.6,
    private val minClusterSize: Int = 12,
    private val metricCosine: Boolean = true
) {

    /** Hard cluster of every (chunk, speaker): -2 for inactive ones. Mirrors `BaseClustering.__call__`. */
    fun cluster(
        embeddings: List<Array<FloatArray?>>,
        numClusters: Int?,
        minClusters: Int?,
        maxClusters: Int?
    ): Array<IntArray> {
        val numChunks = embeddings.size
        val numSpeakers = embeddings.firstOrNull()?.size ?: 0

        val trainIdx = ArrayList<IntArray>() // (chunk, speaker) of the embeddings that take part
        for (c in 0 until numChunks) for (s in 0 until numSpeakers) if (embeddings[c][s] != null) trainIdx += intArrayOf(c, s)
        val train = trainIdx.map { embeddings[it[0]][it[1]]!! }
        val n = train.size

        // set_num_clusters
        var minK = numClusters ?: minClusters ?: 1
        minK = max(1, min(n, minK))
        var maxK = numClusters ?: maxClusters ?: n
        maxK = max(1, min(n, maxK))
        require(minK <= maxK) { "min_clusters must be smaller than (or equal to) max_clusters" }
        val wanted = if (minK == maxK) minK else numClusters

        val hard = Array(numChunks) { IntArray(numSpeakers) { -2 } }
        if (n == 0) return hard

        if (maxK < 2) {
            for ((c, s) in trainIdx.map { it[0] to it[1] }) hard[c][s] = 0
            return hard
        }

        val trainClusters = clusterEmbeddings(train, minK, maxK, wanted)
        val k = (trainClusters.maxOrNull() ?: 0) + 1

        // assign_embeddings: centroid = mean of the (un-normalised) train embeddings, argmax of 2 - cosine distance
        val dim = train[0].size
        val centroids = Array(k) { DoubleArray(dim) }
        val counts = IntArray(k)
        for ((i, e) in train.withIndex()) {
            counts[trainClusters[i]]++
            for (d in 0 until dim) centroids[trainClusters[i]][d] += e[d].toDouble()
        }
        for (cl in 0 until k) for (d in 0 until dim) centroids[cl][d] /= counts[cl]

        for ((c, s) in trainIdx.map { it[0] to it[1] }) {
            val e = embeddings[c][s]!!
            var best = 0
            var bestScore = Double.NEGATIVE_INFINITY
            for (cl in 0 until k) {
                val score = 2 - distance(e, centroids[cl])
                if (score > bestScore) { bestScore = score; best = cl }
            }
            hard[c][s] = best
        }
        return hard
    }

    /** `AgglomerativeClustering.cluster`: clusters of the train embeddings, 0-indexed. */
    internal fun clusterEmbeddings(embeddings: List<FloatArray>, minClusters: Int, maxClusters: Int, numClusters: Int?): IntArray {
        val n = embeddings.size
        // heuristic to reduce min_cluster_size when there are very few embeddings: Python's round() is half-to-even
        val minSize = min(minClusterSize, max(1, Math.rint(0.1 * n).toInt()))

        if (n == 1) return intArrayOf(0)

        // centroid linkage only supports euclidean: unit-normalise (in float32, like numpy) and use euclid
        // (numpy normalises the array in place, so the later small-cluster re-assignment also sees unit vectors)
        val unitF = embeddings.map { e ->
            var sum = 0f
            for (v in e) sum += v * v
            val norm = sqrt(sum)
            FloatArray(e.size) { e[it] / norm }
        }
        val unit = Array(n) { i -> DoubleArray(unitF[i].size) { unitF[i][it].toDouble() } }
        val z = centroidLinkage(unit)

        var clusters = fcluster(z, threshold, n)
        var (unique, counts) = uniqueCounts(clusters)
        var large = unique.filterIndexed { i, _ -> counts[i] >= minSize }
        var numLarge = large.size

        var target = numClusters
        if (numLarge < minClusters) target = minClusters else if (numLarge > maxClusters) target = maxClusters

        if (target != null && numLarge != target) {
            // switch the stopping criterion from distance to iteration index and look for the closest candidate
            val zIter = Array(z.size) { r -> z[r].copyOf().also { it[2] = r.toDouble() } }
            var bestIteration = n - 1
            var bestNumLarge = 1

            val order = z.indices.sortedWith(compareBy({ abs(z[it][2] - threshold) }, { it }))
            for (iteration in order) {
                if (zIter[iteration][3] < minSize) continue
                clusters = fcluster(zIter, iteration.toDouble(), n)
                val uc = uniqueCounts(clusters)
                large = uc.first.filterIndexed { i, _ -> uc.second[i] >= minSize }
                numLarge = large.size

                if (abs(numLarge - target) < abs(bestNumLarge - target)) {
                    bestIteration = iteration
                    bestNumLarge = numLarge
                }
                if (numLarge == target) break
            }

            if (bestNumLarge != target) {
                clusters = fcluster(zIter, bestIteration.toDouble(), n)
                val uc = uniqueCounts(clusters)
                unique = uc.first; counts = uc.second
                large = unique.filterIndexed { i, _ -> counts[i] >= minSize }
                numLarge = large.size
            } else {
                val uc = uniqueCounts(clusters)
                unique = uc.first; counts = uc.second
            }
        }

        if (numLarge == 0) return IntArray(n)

        val small = unique.filterIndexed { i, _ -> counts[i] < minSize }
        if (small.isEmpty()) return clusters

        // re-assign every small cluster to the most similar large cluster, by centroid
        val largeCentroids = large.map { centroidOf(unitF, clusters, it) }
        val smallCentroids = small.map { centroidOf(unitF, clusters, it) }
        val reassigned = clusters.copyOf()
        for ((si, sc) in small.withIndex()) {
            var best = 0
            var bestDistance = Double.POSITIVE_INFINITY
            for (li in large.indices) {
                val d = distance(largeCentroids[li], smallCentroids[si])
                if (d < bestDistance) { bestDistance = d; best = li }
            }
            for (i in reassigned.indices) if (clusters[i] == sc) reassigned[i] = large[best]
        }

        // re-number from 0 (np.unique(return_inverse))
        val sorted = reassigned.toSortedSet().toList()
        return IntArray(n) { sorted.indexOf(reassigned[it]) }
    }

    private fun centroidOf(embeddings: List<FloatArray>, clusters: IntArray, k: Int): DoubleArray {
        val dim = embeddings[0].size
        val sum = DoubleArray(dim)
        var count = 0
        for ((i, e) in embeddings.withIndex()) if (clusters[i] == k) { count++; for (d in 0 until dim) sum[d] += e[d].toDouble() }
        for (d in 0 until dim) sum[d] /= count
        return sum
    }

    private fun uniqueCounts(labels: IntArray): Pair<List<Int>, IntArray> {
        val unique = labels.toSortedSet().toList()
        return unique to IntArray(unique.size) { i -> labels.count { it == unique[i] } }
    }

    /** scipy `cdist(metric="cosine")` for one pair (metric cosine) or euclidean. */
    private fun distance(a: FloatArray, b: DoubleArray): Double = distance(DoubleArray(a.size) { a[it].toDouble() }, b)

    private fun distance(a: DoubleArray, b: DoubleArray): Double {
        if (!metricCosine) {
            var s = 0.0
            for (i in a.indices) s += (a[i] - b[i]) * (a[i] - b[i])
            return sqrt(s)
        }
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return 1 - dot / (sqrt(na) * sqrt(nb))
    }

    companion object {
        /**
         * scipy `linkage(X, method="centroid", metric="euclidean")`: every step merges the closest pair of clusters and
         * updates distances with the Lance-Williams centroid formula. Rows are (left id, right id, distance, size), the
         * merged cluster getting id `n + row`.
         */
        internal fun centroidLinkage(points: Array<DoubleArray>): Array<DoubleArray> {
            val n = points.size
            val d = Array(2 * n - 1) { DoubleArray(2 * n - 1) }
            for (i in 0 until n) for (j in i + 1 until n) {
                var s = 0.0
                for (k in points[i].indices) s += (points[i][k] - points[j][k]) * (points[i][k] - points[j][k])
                d[i][j] = sqrt(s)
                d[j][i] = d[i][j]
            }
            val size = IntArray(2 * n - 1) { if (it < n) 1 else 0 }
            val active = ArrayList<Int>((0 until n).toList())
            val z = Array(n - 1) { DoubleArray(4) }

            for (step in 0 until n - 1) {
                var bi = -1
                var bj = -1
                var best = Double.POSITIVE_INFINITY
                for (ai in active.indices) for (aj in ai + 1 until active.size) {
                    val i = active[ai]
                    val j = active[aj]
                    if (d[i][j] < best) { best = d[i][j]; bi = i; bj = j }
                }
                val id = n + step
                z[step][0] = min(bi, bj).toDouble()
                z[step][1] = max(bi, bj).toDouble()
                z[step][2] = best
                z[step][3] = (size[bi] + size[bj]).toDouble()
                size[id] = size[bi] + size[bj]

                for (k in active) {
                    if (k == bi || k == bj) continue
                    val sx = size[bi].toDouble()
                    val sy = size[bj].toDouble()
                    val updated = sqrt(((sx * d[bi][k] * d[bi][k] + sy * d[bj][k] * d[bj][k]) - (sx * sy * best * best / (sx + sy))) / (sx + sy))
                    d[id][k] = updated
                    d[k][id] = updated
                }
                active.remove(bi)
                active.remove(bj)
                active += id
            }
            return z
        }

        /**
         * scipy `fcluster(Z, t, criterion="distance")`: a subtree becomes one flat cluster when the largest merge distance
         * inside it is at most [cutoff]. Labels start at 1, numbered in scipy's traversal order.
         */
        internal fun fcluster(z: Array<DoubleArray>, cutoff: Double, n: Int): IntArray {
            val maxDist = DoubleArray(n - 1)
            for (i in 0 until n - 1) {
                var m = z[i][2]
                for (child in listOf(z[i][0].toInt(), z[i][1].toInt())) if (child >= n) m = max(m, maxDist[child - n])
                maxDist[i] = m
            }

            val labels = IntArray(n)
            val visited = BooleanArray(n - 1)
            val stack = IntArray(n)
            var k = 0
            stack[0] = 2 * n - 2
            var clusters = 0
            var leader = -1
            while (k >= 0) {
                val root = stack[k] - n
                val left = z[root][0].toInt()
                val right = z[root][1].toInt()

                if (leader == -1 && maxDist[root] <= cutoff) {
                    leader = root
                    clusters++
                }
                if (left >= n && !visited[left - n]) {
                    visited[left - n] = true
                    stack[++k] = left
                    continue
                }
                if (right >= n && !visited[right - n]) {
                    visited[right - n] = true
                    stack[++k] = right
                    continue
                }
                if (left < n) {
                    if (leader == -1) clusters++
                    labels[left] = clusters
                }
                if (right < n) {
                    if (leader == -1) clusters++
                    labels[right] = clusters
                }
                if (leader == root) leader = -1
                k--
            }
            return IntArray(n) { labels[it] - 1 } // the pipeline works with fcluster(...) - 1
        }
    }
}
