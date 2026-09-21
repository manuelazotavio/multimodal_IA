package com.avtracker.mobile.fusion

import com.avtracker.mobile.pipeline.TrackedPersonSnapshot

/** What the face side reports per tracked face and frame. */
data class FaceResult(val trackId: Int, val name: String, val confidence: Float)

/** A face box label after the "one name, one face" rule. */
data class FaceLabel(val trackId: Int, val text: String, val unknown: Boolean)

/**
 * Port of the identity bookkeeping in run_multimodal_tracker.video_loop: maps each face track to a
 * stable person_id in the [IdentityRegistry] so that a voice and a face recognised separately end up
 * as the same person.
 */
class FaceIdentitySync(private val registry: IdentityRegistry) {
    private val trackToPerson = HashMap<Int, String>()

    fun sync(results: List<FaceResult>): Map<Int, String> {
        val current = HashMap<Int, String>()
        val usedNames = HashMap<String, Int>() // real name -> track it labelled this frame

        for (res in results) {
            val faceName = res.name
            val trackId = res.trackId
            val personId: String

            val known = trackToPerson[trackId]
            if (known != null) {
                var pid = known
                val mapped = registry.personFor(faceName)
                if (mapped != null) {
                    pid = mapped
                    trackToPerson[trackId] = pid
                } else if (!isGenericFace(faceName) && faceName != UNKNOWN) {
                    // A real name the registry has not seen: adopt it if the current display name is still a placeholder.
                    val display = registry.nameOf(pid).orEmpty()
                    if ((isGenericFace(display) || display.isEmpty() || display == UNKNOWN) && faceName !in usedNames) {
                        registry.setName(pid, faceName)
                        registry.bind(faceName, pid)
                    }
                } else if (isGenericFace(faceName)) {
                    // Auto-enrolled face (Person_N): upgrade "Unknown" to the generic name, unless another person already holds it.
                    val display = registry.nameOf(pid).orEmpty()
                    if (display.isEmpty() || display == UNKNOWN) {
                        val inUse = trackToPerson.values.any { other -> other != pid && registry.nameOf(other) == faceName }
                        if (!inUse) registry.setName(pid, faceName)
                    }
                }
                personId = pid
            } else {
                val mapped = registry.personFor(faceName)
                if (mapped != null) {
                    personId = mapped
                    trackToPerson[trackId] = personId
                } else {
                    personId = registry.newPersonId()
                    trackToPerson[trackId] = personId
                    registry.setName(personId, faceName)
                    if (!isGenericFace(faceName) && faceName != UNKNOWN) registry.bind(faceName, personId)
                }
            }

            current[trackId] = personId
            val displayCheck = registry.nameOf(personId) ?: faceName
            if (!isGenericFace(displayCheck) && displayCheck != UNKNOWN) usedNames[displayCheck] = trackId
        }

        registry.setActiveFaces(current, results.associate { it.trackId to it.name })
        return current
    }

    /**
     * Labels for drawing: the most confident face gets a name first, and a name already shown on another
     * face this frame is a duplicate, so the weaker face falls back to "Unknown".
     */
    fun labels(results: List<FaceResult>, current: Map<Int, String>): List<FaceLabel> {
        val shown = HashSet<String>()
        return results.sortedByDescending { it.confidence }.map { res ->
            val pid = current[res.trackId]
            var display = if (pid != null) registry.nameOf(pid) ?: res.name else res.name
            if (display != UNKNOWN) {
                if (display in shown) display = UNKNOWN else shown += display
            }
            FaceLabel(res.trackId, "$display (%.2f)".format(java.util.Locale.ROOT, res.confidence), display == UNKNOWN)
        }
    }

    companion object {
        const val UNKNOWN = "Unknown"
        private val GENERIC_FACE = Regex("^(Person_\\d+|spk_\\d+|Desconhecido_\\d+)$")

        private fun isGenericFace(name: String) = GENERIC_FACE.matches(name)

        fun fromSnapshots(persons: List<TrackedPersonSnapshot>): List<FaceResult> =
            persons.map { FaceResult(it.trackId, it.name, it.confidence) }
    }
}
