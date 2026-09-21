package com.avtracker.mobile.diarization

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The agglomerative clustering against pyannote's own on random embedding sets (scripts/parity/diarization_reference.py). */
class ClusteringTest {

    @Test
    fun clusteringMatchesPyannoteOnEveryCase() {
        val cases = Json.parseToJsonElement(File("src/test/resources/diarization/expected.json").readText())
            .jsonObject.getValue("clustering").jsonArray
        var failures = 0
        val report = StringBuilder()

        for ((n, c) in cases.withIndex()) {
            val case = c.jsonObject
            val embeddings = case.getValue("embeddings").jsonArray.mapIndexed { i, row ->
                Array(row.jsonArray.size) { s ->
                    if (case.getValue("active").jsonArray[i].jsonArray[s].jsonPrimitive.content.toBoolean())
                        FloatArray(row.jsonArray[s].jsonArray.size) { d -> row.jsonArray[s].jsonArray[d].jsonPrimitive.content.toFloat() }
                    else null
                }
            }
            fun optional(key: String) = case.getValue(key).let { if (it is JsonNull) null else it.jsonPrimitive.int }

            val hard = AgglomerativeClustering(threshold = 0.6, minClusterSize = 12).cluster(
                embeddings, numClusters = optional("num"), minClusters = optional("min"), maxClusters = optional("max")
            )
            val expected = case.getValue("hard").jsonArray.map { row -> row.jsonArray.map { it.jsonPrimitive.int } }
            val actual = hard.map { it.toList() }
            if (expected != actual) {
                failures++
                if (report.length < 2500) report.append("case $n: num=${optional("num")} min=${optional("min")} max=${optional("max")}\n  pyannote: $expected\n  kotlin:   $actual\n")
            }
        }
        assertTrue(failures == 0, "$failures of ${cases.size} clusterings differ from pyannote\n$report")
    }

    @Test
    fun fclusterCutsTheTreeLikeScipy() {
        // scipy: linkage([[0],[1],[10],[11],[30]], "centroid") then fcluster(Z, 3, "distance") - 1
        val z = AgglomerativeClustering.centroidLinkage(arrayOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(10.0), doubleArrayOf(11.0), doubleArrayOf(30.0)))
        assertEquals(listOf(0.0, 1.0, 1.0, 2.0), z[0].toList())
        val labels = AgglomerativeClustering.fcluster(z, 3.0, 5)
        assertEquals(labels[0], labels[1])
        assertEquals(labels[2], labels[3])
        assertTrue(labels[0] != labels[2] && labels[2] != labels[4] && labels[0] != labels[4])
    }
}
