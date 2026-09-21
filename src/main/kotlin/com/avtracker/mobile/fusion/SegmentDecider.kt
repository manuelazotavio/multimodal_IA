package com.avtracker.mobile.fusion

import com.avtracker.mobile.gender.NameGender
import com.avtracker.mobile.voice.SessionSpeakerTracker
import java.util.Locale

/** Everything the attribution cascade looks at for one speech segment. */
class SegmentEvidence(
    /** Best speaker by voice, even if below the verifier threshold ("Unknown" when there are no voice profiles). */
    val rawBestName: String,
    val rawConf: Float,
    /** Voice candidates at or above the verifier threshold, best first. */
    val candidates: List<Pair<String, Float>>,
    /** Face track that the active-speaker detector says was speaking, if any. */
    val asdTrack: Int?,
    /** Lazily asks the ASD for a weaker "best guess" face; only the ASD_GUESS branch evaluates it. */
    val asdGuess: () -> Int?,
    val asdAvailable: Boolean,
    val sessionHint: String?,
    /** Unit-norm voice embedding of the segment, for the session-speaker tracker. */
    val voiceEmbedding: FloatArray?,
    /** "male" / "female" from the pitch of the segment, or null. */
    val audioGender: String? = null
)

class Decision(
    val speakerId: String,
    val verifiedName: String?,
    val label: String,
    val asdPersonId: String?,
    val asdTrackId: Int?,
    /** Voice candidates after the gender filter (the Python `filtered`). */
    val filtered: List<Pair<String, Float>> = emptyList(),
    /** Name and score chosen from [filtered] by the margin rule ("Unknown"/0 if none). */
    val chosenName: String = "Unknown",
    val chosenConf: Float = 0f
) {
    val type: String get() = label.substringBefore(" (")
    val isVerifier: Boolean get() = label.startsWith("VERIFIER")
}

/** Mutable identity bookkeeping shared by the decision cascade and the naming heuristics. */
class SpeakerBook {
    /** Person ids with a confirmed real name (Python `identified_speakers`). */
    val identifiedSpeakers = HashSet<String>()

    /** voice session id -> face person id it was confirmed to be (Python `_session_to_face`). */
    // Insertion-ordered like the Python dict: `next(s for s, p in _session_to_face.items() ...)` takes the first inserted.
    val sessionToFace = LinkedHashMap<String, String>()

    /** Names mentioned in the conversation (Python `_context_names`). */
    val contextNames = HashSet<String>()
}

/**
 * Port of the speaker-attribution cascade in RealtimeTranscriber._process_segment, in the same priority order:
 *
 *   1. VERIFIER_HIGH  voice match at or above verifier_confidence_min
 *   2. VERIFIER_MOD   voice match at or above the verifier threshold
 *   3. VERIFIER_WEAK  raw voice score >= 0.65 corroborated by the active-speaker detector and/or the voice gender
 *   4. ASD            the face whose mouth was moving
 *   5. FACE_ONLY / SINGLE_FACE   a single visible face (only when at most one speaker is expected)
 *   6. ASD_GUESS      weaker mouth-movement guess, unless that face is already bound to another voice session
 *   7. SESSION_FACE / SESSION_HINT / SESSION_TRACKER   voice-only session identity
 *
 * followed by the gender passes: voice candidates whose name contradicts the voice pitch are dropped up front, a
 * session-only decision is upgraded to the one visible face of the right gender (GENDER_FACE), and a visual decision
 * that contradicts the voice pitch is rerouted (GENDER_REROUTE / GENDER_VERIFIER_HINT / GENDER_OVERRIDE).
 */
class SegmentDecider(
    private val registry: IdentityRegistry,
    private val sessionTracker: SessionSpeakerTracker,
    val book: SpeakerBook = SpeakerBook(),
    private val numSpeakers: Int? = null,
    private val verifierThreshold: Float = 0.80f,
    private val verifierConfidenceMin: Float = 0.70f
) {
    val identifiedSpeakers: MutableSet<String> get() = book.identifiedSpeakers
    val sessionToFace: MutableMap<String, String> get() = book.sessionToFace

    private var speakerId = ""
    private var verifiedName: String? = null
    private var label = ""

    private fun names(pid: String?): String = pid?.let(registry::nameOf).orEmpty()
    private fun fmt(v: Float) = "%.2f".format(Locale.ROOT, v)

    @Synchronized
    fun decide(evidence: SegmentEvidence): Decision {
        val activeFaces = registry.activeFacesSnapshot()
        val visualPeople = activeFaces.values.toList()
        val audioGender = evidence.audioGender

        var asdPersonId: String? = null
        var asdTrackId: Int? = null
        evidence.asdTrack?.let { track ->
            activeFaces[track]?.let { asdPersonId = it; asdTrackId = track }
        }

        // ---- gender-aware selection + margin between verifier candidates ------------------------------------------
        var filtered = evidence.candidates
        if (audioGender != null && filtered.isNotEmpty()) {
            val kept = filtered.filter { (name, _) ->
                val g = NameGender.genderOf(name)
                g == null || g == audioGender
            }
            filtered = kept.ifEmpty { evidence.candidates } // everything contradicted -> keep the original list
        }

        var realName = UNKNOWN
        var conf = 0f
        if (filtered.isNotEmpty()) {
            val (bestName, bestScore) = filtered[0]
            val margin = if (filtered.size >= 2) bestScore - filtered[1].second else 1f
            if (margin >= MIN_MARGIN || audioGender != null) { // low margin but gender determined: trust the gender filter
                realName = bestName
                conf = bestScore
            }
        }

        // Reject a verifier match whose name contradicts the audio gender, unless the voice match is very strong.
        val verifierNameGender = if (realName != UNKNOWN) NameGender.genderOf(realName) else null
        val verifierGenderOk = conf >= 0.90f || !(audioGender != null && verifierNameGender != null && audioGender != verifierNameGender)
        val rawName = evidence.rawBestName
        val rawConf = evidence.rawConf
        val rawNameGender = if (rawName != UNKNOWN) NameGender.genderOf(rawName) else null
        val rawGenderOk = !(audioGender != null && rawNameGender != null && audioGender != rawNameGender)

        val singleSpeakerExpected = numSpeakers == null || numSpeakers <= 1

        // ---- the cascade ----------------------------------------------------------------------------------------
        if (realName != UNKNOWN && conf >= verifierConfidenceMin && verifierGenderOk) {
            take(personForVoice(realName), realName, "VERIFIER_HIGH (${fmt(conf)})")
        } else if (realName != UNKNOWN && conf >= verifierThreshold && verifierGenderOk) {
            take(personForVoice(realName), realName, "VERIFIER_MOD (${fmt(conf)})")
        } else if (rawName != UNKNOWN && rawConf >= WEAK_MIN_SCORE && rawGenderOk) {
            // Weak voice match: needs strong corroboration (mouth movement or voice gender). A single visible face is
            // not corroboration, and a gender contradiction vetoes the ASD.
            val nameGender = NameGender.genderOf(rawName)
            val contradicts = audioGender != null && nameGender != null && audioGender != nameGender
            val corroboration = mutableListOf<String>()
            if (asdPersonId != null && !contradicts) corroboration += "ASD"
            if (audioGender != null && nameGender == audioGender) corroboration += "gender"
            if (corroboration.isNotEmpty()) {
                val name = rawName.replace(HOMONYM_SUFFIX, "")
                take(personForVoice(name), name, "VERIFIER_WEAK (${fmt(rawConf)} +${corroboration.joinToString("+")})")
            } else if (asdPersonId != null) {
                take(asdPersonId!!, registry.nameOf(asdPersonId!!), "ASD (${names(asdPersonId).ifEmpty { asdPersonId!! }})")
            } else if (visualPeople.size == 1 && !IdentityRegistry.isGeneric(names(visualPeople[0]))) {
                take(visualPeople[0], registry.nameOf(visualPeople[0]), "FACE_ONLY")
            } else if (visualPeople.size == 1) {
                take(visualPeople[0], registry.nameOf(visualPeople[0]), "SINGLE_FACE (${fmt(rawConf)})")
            } else {
                sessionOutcome(evidence)
            }
        } else if (asdPersonId != null) {
            take(asdPersonId!!, registry.nameOf(asdPersonId!!), "ASD (${names(asdPersonId).ifEmpty { asdPersonId!! }})")
        } else if (visualPeople.size == 1 && !IdentityRegistry.isGeneric(names(visualPeople[0])) && singleSpeakerExpected) {
            // Only when at most one speaker is expected: with 2+ speakers, one visible face is not the speaker.
            take(visualPeople[0], registry.nameOf(visualPeople[0]), "FACE_ONLY")
        } else if (visualPeople.size == 1 && rawConf >= SINGLE_FACE_MIN_SCORE && singleSpeakerExpected) {
            take(visualPeople[0], registry.nameOf(visualPeople[0]), "SINGLE_FACE (${fmt(rawConf)})")
        } else if (visualPeople.isNotEmpty() && evidence.asdAvailable) {
            // ASD_GUESS: weaker relative mouth-movement guess, rejected if that face is already bound to a DIFFERENT
            // voice session (face A bound to SPEAKER_0 must not be given to SPEAKER_1's segment).
            val guessTrack = evidence.asdGuess()
            var guessOk = false
            var guessPerson: String? = null
            if (guessTrack != null && guessTrack in activeFaces) {
                guessPerson = activeFaces.getValue(guessTrack)
                val boundSession = sessionToFace.entries.firstOrNull { it.value == guessPerson }?.key
                guessOk = boundSession == null || evidence.sessionHint == null || boundSession == evidence.sessionHint
            }
            if (guessOk && guessPerson != null) {
                asdPersonId = guessPerson
                asdTrackId = guessTrack
                take(guessPerson, registry.nameOf(guessPerson), "ASD_GUESS (${names(guessPerson).ifEmpty { guessPerson }})")
            } else {
                sessionOutcome(evidence)
            }
        } else {
            sessionOutcome(evidence)
        }

        // ---- gender passes --------------------------------------------------------------------------------------
        val allIdentified = numSpeakers == null || numSpeakers <= 0 || identifiedSpeakers.size >= numSpeakers
        var genderUnique = false
        if (!allIdentified && audioGender != null) {
            val identifiedNames = identifiedSpeakers.map { names(it) }.toSet()
            val sameGenderIdentified = identifiedSpeakers.count { NameGender.genderOf(names(it)) == audioGender }
            val sameGenderContext = book.contextNames.count { it !in identifiedNames && NameGender.genderOf(it) == audioGender }
            // Only safe if exactly one person of this gender is in the whole meeting.
            genderUnique = sameGenderIdentified == 1 && sameGenderIdentified + sameGenderContext <= 1
        }

        // Upgrade a session-only decision to the single visible identified face of the right gender.
        if ((allIdentified || genderUnique) && (label.startsWith("SESSION_HINT") || label.startsWith("SESSION_TRACKER")) &&
            audioGender != null && visualPeople.size >= 2
        ) {
            var match: String? = null
            var count = 0
            for (vp in visualPeople) {
                val name = names(vp)
                if (!IdentityRegistry.isGeneric(name) && NameGender.genderOf(name) == audioGender) {
                    match = vp
                    count++
                }
            }
            if (count == 1 && match != null) take(match, registry.nameOf(match), "GENDER_FACE ($audioGender->${names(match)})")
        }

        // Reject a visual decision when the voice pitch contradicts the assigned name.
        if ((allIdentified || genderUnique) && audioGender != null &&
            listOf("ASD", "FACE_ONLY", "SINGLE_FACE", "SESSION_HINT").any(label::startsWith)
        ) {
            val assigned = verifiedName ?: registry.nameOf(speakerId)
            if (assigned != null && IdentityRegistry.isGeneric(assigned) && label.startsWith("ASD")) {
                // ASD pointed at a generic face: reroute to the single named face matching the voice gender.
                var match: String? = null
                var count = 0
                for (pid in activeFaces.values) {
                    val name = names(pid)
                    if (!IdentityRegistry.isGeneric(name) && NameGender.genderOf(name) == audioGender) {
                        match = pid
                        count++
                    }
                }
                if (count == 1 && match != null) take(match, registry.nameOf(match), "GENDER_REROUTE ($audioGender->${names(match)})")
            } else if (assigned != null && !IdentityRegistry.isGeneric(assigned)) {
                val nameGender = NameGender.genderOf(assigned)
                if (nameGender != null && nameGender != audioGender) {
                    val other = activeFaces.values.firstOrNull { pid ->
                        val name = names(pid)
                        pid != speakerId && !IdentityRegistry.isGeneric(name) && NameGender.genderOf(name) == audioGender
                    }
                    if (other != null) {
                        take(other, registry.nameOf(other), "GENDER_REROUTE ($audioGender->${names(other)})")
                    } else if (rawName != UNKNOWN && rawConf >= 0.15f && NameGender.genderOf(rawName) == audioGender) {
                        take(personForVoice(rawName), rawName, "GENDER_VERIFIER_HINT ($rawName ${fmt(rawConf)})")
                    } else {
                        val rejected = speakerId
                        val id = evidence.voiceEmbedding
                            ?.let { sessionTracker.getOrCreate(it, setOf(rejected), audioGender) }
                            ?: registry.newPersonId()
                        take(id, null, "GENDER_OVERRIDE ($audioGender!=$nameGender)")
                    }
                }
            }
        }

        val decision = Decision(
            speakerId = speakerId,
            verifiedName = verifiedName,
            label = label,
            asdPersonId = asdPersonId,
            asdTrackId = asdTrackId,
            filtered = filtered,
            chosenName = realName,
            chosenConf = conf
        )
        bindSessionToFace(evidence.sessionHint, decision)

        verifiedName?.let { name ->
            registry.setName(speakerId, name)
            if (!IdentityRegistry.isGeneric(name)) identifiedSpeakers += speakerId
        }
        return decision
    }

    private fun take(id: String, name: String?, decisionLabel: String) {
        speakerId = id
        verifiedName = name
        label = decisionLabel
    }

    /** Voice-only outcome: the face bound to this session, else the session hint, else a tracked session id. */
    private fun sessionOutcome(evidence: SegmentEvidence) {
        val hint = evidence.sessionHint
        if (hint != null) {
            val face = sessionToFace[hint]
            if (face != null && registry.nameOf(face) != null) {
                take(face, registry.nameOf(face), "SESSION_FACE (${registry.nameOf(face)})")
            } else {
                take(hint, null, "SESSION_HINT")
            }
            return
        }
        val id = evidence.voiceEmbedding?.let { sessionTracker.getOrCreate(it, emptySet(), evidence.audioGender) } ?: registry.newPersonId()
        take(id, null, "SESSION_TRACKER ($id)")
    }

    /** The person id that owns the voice profile [name], creating it the first time the voice is heard. */
    private fun personForVoice(name: String): String {
        registry.personFor(name)?.let { return it }
        val pid = registry.newPersonId()
        registry.bind(name, pid)
        registry.setName(pid, name)
        return pid
    }

    /**
     * A decision that rests on a face or a voice match confirms who this voice session is; remember it so the
     * next segment of the same session resolves to the same person instead of a fresh spk_N.
     */
    private fun bindSessionToFace(sessionHint: String?, decision: Decision) {
        if (sessionHint == null || FACE_DECISIONS.none { decision.label.startsWith(it) }) return
        sessionToFace[sessionHint] = decision.speakerId
        sessionTracker.mirror(sessionHint, decision.speakerId)
    }

    companion object {
        const val UNKNOWN = "Unknown"
        private const val MIN_MARGIN = 0.04f
        private const val WEAK_MIN_SCORE = 0.65f
        private const val SINGLE_FACE_MIN_SCORE = 0.35f
        private val HOMONYM_SUFFIX = Regex("\\s+\\d+$")
        private val FACE_DECISIONS = listOf(
            "ASD", "ASD_GUESS", "FACE_ONLY", "SINGLE_FACE", "GENDER_FACE", "GENDER_REROUTE",
            "VERIFIER_HIGH", "VERIFIER_MOD", "VERIFIER_WEAK"
        )
    }
}
