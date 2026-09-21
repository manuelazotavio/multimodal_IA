package com.avtracker.mobile.fusion

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.round

/** Everything the Python version records per segment for post-session analysis (false negatives/positives, metrics). */
class SessionLog(
    val sessionStartMillis: Long = System.currentTimeMillis(),
    private val config: Map<String, JsonElement> = emptyMap()
) {
    class FalseNegative(
        val timestamp: String,
        val reasons: List<String>,
        val decision: String,
        val finalName: String,
        val text: String,
        val rawBest: String,
        val candidates: List<Pair<String, Double>>,
        val audioGender: String?,
        val faces: Int
    )

    val segmentMetrics = ArrayList<JsonObject>()
    val falseNegatives = ArrayList<FalseNegative>()

    private fun hms(millis: Long) = SimpleDateFormat("HH:mm:ss", Locale.ROOT).format(Date(millis))
    private fun r(v: Double, digits: Int): Double { val f = Math.pow(10.0, digits.toDouble()); return round(v * f) / f }
    private fun f3(v: Float) = "%.3f".format(Locale.ROOT, v)

    /**
     * Voice candidates that were plausible but not used: the verifier had a match below its thresholds, the gender
     * filter removed the best one, or two candidates were too close to call. Returns the reasons (empty = none).
     */
    fun falseNegativeReasons(
        decision: Decision,
        rawBestName: String,
        rawConf: Float,
        allCandidates: List<Pair<String, Float>>,
        audioGender: String?,
        verifierThreshold: Float,
        verifierConfidenceMin: Float
    ): List<String> {
        if (decision.label.startsWith("VERIFIER")) return emptyList()
        val awareness = FN_AWARENESS
        val reasons = ArrayList<String>()
        val filtered = decision.filtered

        if (rawBestName != "Unknown" && rawConf >= awareness) {
            if (rawConf < verifierThreshold) {
                reasons += "BELOW_THRESHOLD: $rawBestName=${f3(rawConf)} (threshold=$verifierThreshold)"
            } else if (rawConf < verifierConfidenceMin) {
                reasons += "BELOW_CONF_MIN: $rawBestName=${f3(rawConf)} (conf_min=$verifierConfidenceMin)"
            }
        }
        if (rawBestName != "Unknown" && rawConf >= awareness) {
            val wasCandidate = allCandidates.any { it.first == rawBestName }
            if (wasCandidate && rawBestName !in filtered.map { it.first }) {
                reasons += "GENDER_FILTERED: $rawBestName=${f3(rawConf)} removed (audio_gender=$audioGender)"
            }
        }
        if (filtered.size >= 2) {
            val (n0, s0) = filtered[0]
            val (n1, s1) = filtered[1]
            val margin = s0 - s1
            if (s0 >= awareness && margin < 0.04f && audioGender == null) {
                reasons += "MARGIN_REJECT: $n0=${f3(s0)} vs $n1=${f3(s1)} (margin=${f3(margin)}<0.04)"
            }
        }
        return reasons
    }

    /** Port of the FALSE POSITIVE RISK block: (risk, reasons) for a named final decision. */
    fun falsePositiveRisk(
        decision: Decision,
        finalName: String?,
        rawConf: Float,
        audioGender: String?,
        verifierNames: Set<String>,
        verifierThreshold: Float,
        numSpeakers: Int?,
        identifiedCount: Int
    ): Pair<String, List<String>> {
        var risk = "none"
        val reasons = ArrayList<String>()
        val isIdentified = finalName != null && !IdentityRegistry.isGeneric(finalName)
        if (isIdentified) {
            val type = decisionType(decision.label)
            if (type in setOf("VERIFIER_WEAK", "FACE_ONLY", "SINGLE_FACE")) {
                reasons += "WEAK_DECISION:$type"
                risk = "medium"
            }
            val nameInVerifier = finalName!! in verifierNames || verifierNames.any { finalName in it }
            if (nameInVerifier && rawConf < verifierThreshold) {
                reasons += "UNCONFIRMED_KNOWN:$finalName raw=${f3(rawConf)}<thr=$verifierThreshold"
                risk = "high"
            }
            if (audioGender != null) {
                val nameGender = com.avtracker.mobile.gender.NameGender.genderOf(finalName)
                if (nameGender != null && nameGender != audioGender) {
                    reasons += "GENDER_MISMATCH:audio=$audioGender name=$nameGender($finalName)"
                    risk = "high"
                }
            }
            if (numSpeakers != null && numSpeakers > 0 && identifiedCount > numSpeakers) {
                reasons += "SPEAKER_OVERFLOW:$identifiedCount>$numSpeakers"
                if (risk != "high") risk = "medium"
            }
        }
        return risk to reasons
    }

    fun recordFalseNegative(entry: FalseNegative) { falseNegatives += entry }

    fun newFalseNegative(
        reasons: List<String>, decision: Decision, finalName: String, text: String, rawBestName: String, rawConf: Float,
        allCandidates: List<Pair<String, Float>>, audioGender: String?, faces: Int, nowMillis: Long
    ) = FalseNegative(
        timestamp = hms(nowMillis),
        reasons = reasons,
        decision = decision.label,
        finalName = finalName,
        text = if (text.length > 60) text.take(60) + "..." else text,
        rawBest = if (rawBestName != "Unknown" || rawConf > 0) "$rawBestName=${f3(rawConf)}" else "none",
        candidates = allCandidates.take(4).map { it.first to r(it.second.toDouble(), 3) },
        audioGender = audioGender,
        faces = faces
    )

    /** One structured metrics record per processed segment (same keys as the Python `_segment_metrics`). */
    fun recordSegment(
        nowMillis: Long,
        decision: Decision,
        rawBestName: String,
        rawConf: Float,
        allCandidates: List<Pair<String, Float>>,
        finalName: String?,
        audioGender: String?,
        asdActive: Boolean,
        faces: Int,
        fnReasons: List<String>,
        fpRisk: String,
        fpReasons: List<String>,
        textLength: Int,
        durationSec: Double,
        whisperMs: Double,
        verifierMs: Double,
        segmentMs: Double,
        avgLogProb: Float,
        avgNoSpeech: Float
    ) {
        val filtered = decision.filtered
        segmentMetrics += buildJsonObject {
            put("timestamp", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.ROOT).format(Date(nowMillis)))
            put("decision", decision.label)
            put("decision_type", decisionType(decision.label))
            put("raw_best_name", if (rawBestName != "Unknown") JsonPrimitive(rawBestName) else JsonNull)
            put("raw_conf", r(rawConf.toDouble(), 4))
            put("chosen_name", if (decision.chosenName != "Unknown") JsonPrimitive(decision.chosenName) else JsonNull)
            put("chosen_conf", r(decision.chosenConf.toDouble(), 4))
            put("final_name", finalName?.let(::JsonPrimitive) ?: JsonNull)
            put("final_speaker", decision.speakerId)
            put("is_identified", finalName != null && !IdentityRegistry.isGeneric(finalName))
            put("audio_gender", audioGender?.let(::JsonPrimitive) ?: JsonNull)
            put("n_candidates", allCandidates.size)
            put("n_filtered", filtered.size)
            put("margin", if (filtered.size >= 2) JsonPrimitive(r((filtered[0].second - filtered[1].second).toDouble(), 4)) else JsonNull)
            put("asd_active", asdActive)
            put("n_faces", faces)
            put("is_fn", fnReasons.isNotEmpty())
            putJsonArray("fn_reasons") { fnReasons.forEach { add(JsonPrimitive(it.substringBefore(":"))) } }
            put("fp_risk", fpRisk)
            putJsonArray("fp_reasons") { fpReasons.forEach { add(JsonPrimitive(it)) } }
            put("text_len", textLength)
            put("segment_duration_s", r(durationSec, 2))
            put("whisper_ms", r(whisperMs, 1))
            put("verifier_ms", r(verifierMs, 1))
            put("segment_ms", r(segmentMs, 1))
            put("avg_logprob", r(avgLogProb.toDouble(), 3))
            put("avg_no_speech", r(avgNoSpeech.toDouble(), 3))
            putJsonArray("candidates") {
                allCandidates.take(5).forEach { add(JsonArray(listOf(JsonPrimitive(it.first), JsonPrimitive(r(it.second.toDouble(), 4))))) }
            }
        }
    }

    // ---- report files -----------------------------------------------------------------------------------------------

    fun falseNegativeReport(): String? {
        if (falseNegatives.isEmpty()) return null
        return buildString {
            append("FALSE NEGATIVE REPORT — ${falseNegatives.size} event(s)\n")
            append("=".repeat(60)).append("\n\n")
            for (ev in falseNegatives) {
                append("[${ev.timestamp}] ${ev.reasons.joinToString(" | ")}\n")
                append("  decision=${ev.decision} final=${ev.finalName} raw_best=${ev.rawBest} gender=${ev.audioGender} faces=${ev.faces}\n")
                append("  text=\"${ev.text}\"\n")
                if (ev.candidates.isNotEmpty()) {
                    append("  candidates=[${ev.candidates.joinToString(" | ") { "${it.first}:${f3(it.second.toFloat())}" }}]\n")
                }
                append("\n")
            }
        }
    }

    fun falsePositiveReport(): String? {
        val suspects = segmentMetrics.filter { (it["fp_risk"] as JsonPrimitive).content in setOf("high", "medium") }
        if (suspects.isEmpty()) return null
        val high = suspects.count { (it["fp_risk"] as JsonPrimitive).content == "high" }
        val medium = suspects.count { (it["fp_risk"] as JsonPrimitive).content == "medium" }
        return buildString {
            append("FALSE POSITIVE RISK REPORT — ${suspects.size} suspect segment(s)\n")
            append("  HIGH risk: $high  |  MEDIUM risk: $medium\n")
            append("=".repeat(60)).append("\n\n")
            for (m in suspects) {
                fun s(key: String) = (m[key] as? JsonPrimitive)?.takeIf { it !is JsonNull }?.content ?: "None"
                append("[${s("timestamp")}] risk=${s("fp_risk").uppercase()}\n")
                append("  decision=${s("decision")}  final=${s("final_name")}  speaker=${s("final_speaker")}\n")
                append("  raw_best=${s("raw_best_name")}:${"%.3f".format(Locale.ROOT, s("raw_conf").toDouble())}  gender=${s("audio_gender")}\n")
                (m["fp_reasons"] as JsonArray).forEach { append("  reason: ${(it as JsonPrimitive).content}\n") }
                val cands = m["candidates"] as JsonArray
                if (cands.isNotEmpty()) {
                    append("  candidates=[${cands.joinToString(" | ") { c -> val a = c as JsonArray; "${(a[0] as JsonPrimitive).content}:${"%.3f".format(Locale.ROOT, (a[1] as JsonPrimitive).content.toDouble())}" }}]\n")
                }
                append("\n")
            }
        }
    }

    /** The `metrics_<ts>.json` payload: config, summary and every segment. */
    fun metricsJson(sessionId: String, verifierThreshold: Float, verifierConfidenceMin: Float, numSpeakers: Int?, whisperModel: String): JsonObject? {
        if (segmentMetrics.isEmpty()) return null
        fun bool(m: JsonObject, key: String) = (m[key] as JsonPrimitive).content == "true"
        fun num(m: JsonObject, key: String) = (m[key] as JsonPrimitive).content.toDouble()

        val decisionCounts = LinkedHashMap<String, Int>()
        segmentMetrics.forEach { decisionCounts.merge((it["decision_type"] as JsonPrimitive).content, 1, Int::plus) }
        val rawConfs = segmentMetrics.map { num(it, "raw_conf") }.filter { it > 0 }
        val total = segmentMetrics.size
        val fn = segmentMetrics.count { bool(it, "is_fn") }
        val identified = segmentMetrics.count { bool(it, "is_identified") }
        val fpHigh = segmentMetrics.count { (it["fp_risk"] as JsonPrimitive).content == "high" }
        val fpMedium = segmentMetrics.count { (it["fp_risk"] as JsonPrimitive).content == "medium" }
        fun median(v: List<Double>): Double { val s = v.sorted(); return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2 }

        return buildJsonObject {
            put("session_id", sessionId)
            put("session_start", SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.ROOT).format(Date(sessionStartMillis)))
            put("config", buildJsonObject {
                put("verifier_threshold", verifierThreshold)
                put("verifier_confidence_min", verifierConfidenceMin)
                put("num_speakers", numSpeakers?.let(::JsonPrimitive) ?: JsonNull)
                put("whisper_size", whisperModel)
                put("sample_rate", 16000)
                config.forEach { (k, v) -> put(k, v) }
            })
            put("summary", buildJsonObject {
                put("total_segments", total)
                put("total_fn", fn)
                put("total_identified", identified)
                put("unique_speakers", segmentMetrics.map { (it["final_speaker"] as JsonPrimitive).content }.toSet().size)
                put("decision_counts", JsonObject(decisionCounts.mapValues { JsonPrimitive(it.value) }))
                put("fn_rate", r(fn.toDouble() / total, 4))
                put("identification_rate", r(identified.toDouble() / total, 4))
                put("avg_raw_conf", if (rawConfs.isNotEmpty()) r(rawConfs.average(), 4) else 0.0)
                put("median_raw_conf", if (rawConfs.isNotEmpty()) r(median(rawConfs), 4) else 0.0)
                put("avg_whisper_ms", r(segmentMetrics.map { num(it, "whisper_ms") }.average(), 1))
                put("avg_verifier_ms", r(segmentMetrics.map { num(it, "verifier_ms") }.average(), 1))
                put("avg_segment_ms", r(segmentMetrics.map { num(it, "segment_ms") }.average(), 1))
                put("total_fp_high", fpHigh)
                put("total_fp_medium", fpMedium)
                put("fp_rate", r((fpHigh + fpMedium).toDouble() / total, 4))
            })
            put("segments", JsonArray(segmentMetrics))
        }
    }

    companion object {
        private const val FN_AWARENESS = 0.45f

        /** `decision.split("(")[0].split(" ")[0].strip()` */
        fun decisionType(label: String): String = label.split("(")[0].split(" ")[0].trim()
    }
}
