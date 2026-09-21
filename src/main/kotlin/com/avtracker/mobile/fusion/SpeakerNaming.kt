package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.gender.NameGender
import com.avtracker.mobile.gender.PitchGender
import com.avtracker.mobile.voice.SessionSpeakerTracker
import com.avtracker.mobile.voice.SpeakerVerifier

/** Person-name recognition on transcript text (spaCy NER in the Python version). */
interface EntityExtractor {
    /** False when no model is loaded: name rules then fall back to the blocklist alone, like `self.nlp is None`. */
    val available: Boolean

    /** Port of `_extract_names_with_ner`: PERSON entities, honorifics stripped, longer than 2 characters. */
    fun personNames(text: String): List<String>
}

/** Nothing recognised, model absent. */
object NoEntities : EntityExtractor {
    override val available = false
    override fun personNames(text: String) = emptyList<String>()
}

/** What the voice side needs from the face tracker (Python `shared_state["face_tracker"]`). */
interface FaceTrackerBridge {
    /** The name the tracker gave the face on [trackId] (`_track_to_name`). */
    fun trackName(trackId: Int): String?

    /** `name in face_tracker.known_embeddings`. */
    fun knowsFace(name: String): Boolean

    /** `face_tracker.rename_person(old, new, only_track_id=...)`. */
    fun renamePerson(oldName: String, newName: String, onlyTrackId: Int?)
}

/**
 * The name-inference half of RealtimeTranscriber: who a speaker is from what they and others say, and saving
 * their voice once they have a name.
 *
 *  - [updateNamesIncremental]  self-introductions ("I'm Laura", "meu nome é ...")
 *  - [detectContextNames] + [applyContextNaming]  vocatives and mentions vote for the next speaker's name
 *  - [saveLiveEmbedding]  persists the accumulated audio as a voice profile and renames generic ones
 *  - [tryFaceVoiceBinding]  a recognised face that is speaking lends its name to the voice
 *  - [deduceLastSpeaker]  with N-1 identified people, the last generic speaker gets the one unused name
 */
class SpeakerNaming(
    private val registry: IdentityRegistry,
    private val verifier: SpeakerVerifier,
    private val book: SpeakerBook,
    private val voice: VoiceEncoder,
    private val entities: EntityExtractor = NoEntities,
    private val face: FaceTrackerBridge? = null,
    private val numSpeakers: Int? = null,
    private val clock: () -> Double = AudioUtils::nowSec,
    private val genderDetector: (FloatArray) -> String? = PitchGender::detect
) {
    private class Addressee(val name: String, val fromSpeaker: String, val ts: Double)

    private var pendingAddressee: Addressee? = null
    // Insertion-ordered like the Python dict, so ties between names resolve the same way.
    private val addresseeVotes = HashMap<String, LinkedHashMap<String, Int>>()
    private val fvBindLast = HashMap<String, Double>()

    /** Audio kept per speaker (about 60 s) to build a voice profile once the speaker is identified. */
    val unknownSpeakersAudio = LinkedHashMap<String, MutableList<FloatArray>>()
    private val voiceEmbSaved = HashSet<String>()

    private val identified get() = book.identifiedSpeakers

    // ---- audio bookkeeping ------------------------------------------------------------------------------------------

    /** Accumulates every segment's audio per speaker and auto-saves a voice for speakers that have none yet. */
    fun accumulateAndAutoSave(speakerId: String, audio: FloatArray) {
        val buf = unknownSpeakersAudio.getOrPut(speakerId) { mutableListOf() }
        buf += audio.copyOf()
        if (buf.size > MAX_CHUNKS_PER_SPEAKER) buf.removeAt(0)

        // Generic names (spk_N, Person_N): always save so validation can rename them. Named speakers that already
        // have a voice are skipped: auto-enrol handles them with stricter checks and a false positive here would
        // contaminate their centroid.
        val current = PyText.stripHomonymSuffix(registry.nameOf(speakerId) ?: speakerId)
        val hasExisting = !IdentityRegistry.isGeneric(current) && current in verifier
        if (speakerId !in voiceEmbSaved && buf.size >= MIN_CHUNKS_FOR_EMB && !hasExisting) {
            saveLiveEmbedding(speakerId, current, null)
            voiceEmbSaved += speakerId
        }
    }

    // ---- gender from text -------------------------------------------------------------------------------------------

    /** Port of `_detect_gender_from_text`: NER names first, then Portuguese gender markers in speech. */
    fun detectGenderFromText(text: String): String? {
        val lower = text.lowercase()
        for (name in entities.personNames(text)) {
            val key = NameGender.normalize(name)
            if (NameGender.isFemaleName(key)) return NameGender.FEMALE
            if (NameGender.isMaleName(key)) return NameGender.MALE
        }
        val female = FEMALE_MARKERS.count { it.containsMatchIn(lower) }
        val male = MALE_MARKERS.count { it.containsMatchIn(lower) }
        return when {
            female > male -> NameGender.FEMALE
            male > female -> NameGender.MALE
            else -> null
        }
    }

    // ---- self-introductions -----------------------------------------------------------------------------------------

    /** Port of `_update_speaker_names_incremental`. */
    fun updateNamesIncremental(speakerId: String, text: String, audio: FloatArray?) {
        val currentName = registry.nameOf(speakerId)
        val isRename = IdentityRegistry.isGeneric(currentName)
        if (currentName != null && !isRename) return // already has a real name
        // The speaker limit only blocks adding a new speaker, not renaming a known one.
        if (!isRename && numSpeakers != null && numSpeakers > 0 && identified.size >= numSpeakers) return

        val gender = detectGenderFromText(text) ?: audio?.let(genderDetector)

        for (pattern in INTRODUCTION_PATTERNS) {
            val match = pattern.find(text) ?: continue
            val raw = match.groupValues[1]
            // Must start upper-case in the original text and not be a common word.
            if (!raw[0].isUpperCase() || raw.lowercase().split(WHITESPACE).first() in NOT_NAMES) continue

            val name = NameGender.correctForGender(PyText.title(raw), gender)
            // A name that already exists is assumed to be the same person introducing themselves again: merge by
            // pointing it at this speaker rather than creating "Jose Marques 2".
            registry.bind(name, speakerId)
            registry.setName(speakerId, name)
            identified += speakerId
            saveLiveEmbedding(speakerId, name, currentName)
            return
        }
    }

    // ---- contextual naming ------------------------------------------------------------------------------------------

    private fun validName(n: String?): Boolean =
        n != null && n.length >= 2 && n[0].isUpperCase() &&
            n.lowercase() !in VOCATIVE_BLOCKLIST && !GENERIC_FACE_OR_SPK.matches(n)

    /**
     * Port of `_detect_context_names`: vocatives ("Arthur, pode falar"), invitations, thanks and third-person mentions.
     * Returns (name, role) with role "addressee" or "mentioned"; also registers the names and the pending addressee.
     */
    fun detectContextNames(text: String, currentSpeakerId: String): List<Pair<String, String>> {
        val results = mutableListOf<Pair<String, String>>()

        val nerEntities = entities.personNames(text)
        val nerSet = HashSet<String>()
        for (e in nerEntities) {
            nerSet += e.lowercase()
            nerSet += e.lowercase().split(WHITESPACE)
        }
        // NER must confirm the candidate; when NER is unavailable fall back to the blocklist alone.
        fun isPerson(n: String): Boolean {
            if (!entities.available) return true
            val nl = n.lowercase()
            return nl in nerSet || nl.split(WHITESPACE).any { it in nerSet }
        }

        // 1) Vocative at the start: "Arthur, pode falar". NER-gated: rejects "Yeah,", "So,", "Well,".
        VOCATIVE_START.find(text)?.let { m ->
            val name = m.groupValues[1]
            if (validName(name) && isPerson(name)) results += PyText.title(name) to ADDRESSEE
        }
        // 2) Vocative at the end: "Pode falar, Arthur".
        VOCATIVE_END.find(text)?.let { m ->
            val name = m.groupValues[1]
            if (validName(name) && isPerson(name)) results += PyText.title(name) to ADDRESSEE
        }
        // 3) Invitation to speak / thanks: "Fala, Arthur", "Obrigado, Manu".
        for (pattern in INVITATION_PATTERNS) {
            pattern.find(text)?.let { m ->
                val raw = m.groupValues[1]
                if (raw[0].isUpperCase() && validName(raw)) results += PyText.title(raw) to ADDRESSEE
            }
        }
        // 4) Third-person references: "a parte do Kauan".
        for (pattern in MENTION_PATTERNS) {
            for (m in pattern.findAll(text)) {
                val raw = m.groupValues[1]
                if (validName(raw)) results += PyText.title(raw) to MENTIONED
            }
        }
        // 5) NER fallback: any PERSON entity not already found.
        val found = results.map { it.first.lowercase() }.toSet()
        for (n in nerEntities) {
            if (n.lowercase() !in found && validName(n)) results += PyText.title(n) to MENTIONED
        }

        for ((name, _) in results) book.contextNames += name

        // If an addressee was found, the next speaker is probably that person (one addressee per segment).
        results.firstOrNull { it.second == ADDRESSEE }?.let { (name, _) ->
            pendingAddressee = Addressee(name, currentSpeakerId, clock())
        }
        return results
    }

    /** Port of `_apply_context_naming`: names an unidentified speaker once the votes reach 3. */
    fun applyContextNaming(speakerId: String) {
        val currentName = registry.nameOf(speakerId)
        if (currentName != null && !IdentityRegistry.isGeneric(currentName)) return // already has a real name

        val pending = pendingAddressee
        val now = clock()
        // 1) A vocative from the previous turn is a strong signal (+2) for whoever speaks next.
        if (pending != null && pending.fromSpeaker != speakerId && now - pending.ts < ADDRESSEE_WINDOW_SEC) {
            val owner = registry.personFor(pending.name)
            if (owner == null || owner == speakerId) {
                addresseeVotes.getOrPut(speakerId) { LinkedHashMap() }.merge(pending.name, 2, Int::plus)
            }
        }

        // 2) Accumulated votes: assign when confident (>= 3).
        val votes = addresseeVotes[speakerId]
        if (!votes.isNullOrEmpty()) {
            val (bestName, bestCount) = votes.maxByOrNull { it.value }!!.toPair()
            if (bestCount >= VOTES_NEEDED) {
                val owner = registry.personFor(bestName)
                if (owner == null || owner == speakerId) {
                    val speakersFull = numSpeakers != null && numSpeakers > 0 && identified.size >= numSpeakers
                    if (!speakersFull) {
                        registry.setName(speakerId, bestName)
                        registry.bind(bestName, speakerId)
                        identified += speakerId
                        saveLiveEmbedding(speakerId, bestName, currentName)
                        addresseeVotes.remove(speakerId)
                    }
                }
            }
        }

        if (pending != null && now - pending.ts > ADDRESSEE_WINDOW_SEC) pendingAddressee = null
    }

    /**
     * Last-speaker deduction: with `numSpeakers` set and all but one identified, the single remaining generic speaker
     * must be the only context name not already taken.
     */
    fun deduceLastSpeaker(transcriptSpeakers: Collection<String>) {
        val limit = numSpeakers ?: return
        if (limit <= 0 || identified.size != limit - 1) return

        val generic = HashSet<String>()
        for (pid in transcriptSpeakers) if (IdentityRegistry.isGeneric(registry.nameOf(pid) ?: "")) generic += pid
        for (pid in registry.activeFacesSnapshot().values) if (IdentityRegistry.isGeneric(registry.nameOf(pid) ?: "")) generic += pid
        if (generic.size != 1) return

        val last = generic.first()
        val used = identified.mapNotNull { registry.nameOf(it) }.filter { !IdentityRegistry.isGeneric(it) }.toSet()
        val candidates = book.contextNames - used
        if (candidates.size != 1) return

        val deduced = candidates.first()
        val old = registry.nameOf(last)
        registry.setName(last, deduced)
        registry.bind(deduced, last)
        identified += last
        saveLiveEmbedding(last, deduced, old)
    }

    // ---- persistence ------------------------------------------------------------------------------------------------

    /** Port of `_save_live_embedding`: builds a voice profile from the speaker's audio and renames generic ones. */
    fun saveLiveEmbedding(speakerId: String, name: String?, oldName: String?) {
        // Never persist a voice under an implausible name (discourse markers, "Unknown", empty). Generic ids are
        // allowed: they exist for the rename cycle.
        val base = PyText.stripHomonymSuffix(name.orEmpty()).trim()
        if (!IdentityRegistry.isGeneric(base) && (base.isEmpty() || base.lowercase() in VOCATIVE_BLOCKLIST || base.lowercase() == "unknown")) return
        val realName = name ?: return

        val chunks = unknownSpeakersAudio[speakerId].orEmpty()
        val activeFaces = registry.activeFacesSnapshot()

        if (oldName != null && IdentityRegistry.isGeneric(oldName)) {
            verifier.renameEmbedding(oldName, realName)
            face?.let { tracker ->
                // If old_name is "Unknown" the face may be enrolled under another name (e.g. "Person_1").
                var faceOld: String = oldName
                var targetTrack: Int? = null
                for ((trackId, pid) in activeFaces) {
                    if (pid == speakerId) {
                        if (!tracker.knowsFace(oldName)) {
                            val enrolled = tracker.trackName(trackId)
                            if (enrolled != null && tracker.knowsFace(enrolled)) faceOld = enrolled
                        }
                        targetTrack = trackId
                        break
                    }
                }
                tracker.renamePerson(faceOld, realName, targetTrack)
            }
        } else if (oldName == null && face != null && !IdentityRegistry.isGeneric(realName)) {
            // Came from session tracking: bind to the one visible generic face, if there is exactly one.
            val genericPids = activeFaces.values.toSet().filter { it != speakerId && IdentityRegistry.isGeneric(registry.nameOf(it) ?: "") }
            if (genericPids.size == 1) {
                val target = genericPids[0]
                val oldFaceName = registry.nameOf(target).orEmpty()
                if (oldFaceName.isNotEmpty()) {
                    val track = activeFaces.entries.firstOrNull { it.value == target }?.key
                    face.renamePerson(oldFaceName, realName, track)
                    registry.setName(target, realName)
                    registry.bind(realName, target)
                }
            }
        }

        if (chunks.isEmpty()) return
        // Skip if a voice already exists for this name, unless we just renamed from a generic one (newer audio wins).
        val renamedFromGeneric = oldName != null && IdentityRegistry.isGeneric(oldName)
        if (!renamedFromGeneric && verifier.hasStoredFileFor(realName)) return

        val combined = FloatArray(chunks.sumOf { it.size })
        var pos = 0
        for (c in chunks) { System.arraycopy(c, 0, combined, pos, c.size); pos += c.size }
        val peak = AudioUtils.peak(combined)
        val scaled = if (peak > 0f) FloatArray(combined.size) { combined[it] / peak } else combined
        val embedding = voice.embed(scaled) ?: return
        verifier.save(realName, embedding)
        verifier.loadEmbeddings()
    }

    /**
     * Port of `_try_face_voice_binding`: when the ASD confirms a recognised face is speaking and we have no voice
     * for that name yet, create one from this audio (face identity bootstraps voice identity).
     */
    fun tryFaceVoiceBinding(trackId: Int, audio: FloatArray) {
        val tracker = face ?: return
        val faceName = tracker.trackName(trackId)
        if (faceName == null || IdentityRegistry.isGeneric(faceName)) return // face not recognised as a known person
        if (faceName in verifier) return

        val now = clock()
        if (now - (fvBindLast[faceName] ?: 0.0) < FACE_VOICE_COOLDOWN_SEC) return
        fvBindLast[faceName] = now

        if (audio.size < 3 * AudioUtils.SAMPLE_RATE) return
        var sumSq = 0.0
        for (v in audio) sumSq += v.toDouble() * v
        if (Math.sqrt(sumSq / audio.size) < 1e-4) return // near-silent

        // A voice of the wrong gender for this name is not this person.
        val audioGender = genderDetector(audio)
        val nameGender = NameGender.genderOf(faceName)
        if (audioGender != null && nameGender != null && nameGender != audioGender) return

        val peak = AudioUtils.peak(audio)
        val scaled = if (peak > 0f) FloatArray(audio.size) { audio[it] / peak } else audio
        val embedding = voice.embed(scaled) ?: return
        verifier.enrollFromFace(faceName, embedding)

        if (registry.personFor(faceName) == null) {
            for ((tid, pid) in registry.activeFacesSnapshot()) {
                if (tracker.trackName(tid) == faceName) {
                    registry.bind(faceName, pid)
                    registry.setName(pid, faceName)
                    identified += pid
                    break
                }
            }
        }
    }

    // ---- post-session enrolment -------------------------------------------------------------------------------------

    class PendingSpeaker(val speakerId: String, val genericName: String, val seconds: Double)

    /** Anonymous speakers with at least [minAudioSec] of audio: the ones worth asking the user to name. */
    fun unidentifiedSpeakers(minAudioSec: Double = 5.0): List<PendingSpeaker> =
        unknownSpeakersAudio.entries.mapNotNull { (spk, chunks) ->
            if (chunks.isEmpty()) return@mapNotNull null
            val current = registry.nameOf(spk) ?: spk
            if (!IdentityRegistry.isGeneric(current)) return@mapNotNull null // identified during the session
            val seconds = chunks.sumOf { it.size }.toDouble() / AudioUtils.SAMPLE_RATE
            if (seconds < minAudioSec) null else PendingSpeaker(spk, current, seconds)
        }

    /** Up to [maxSec] of the speaker's audio, peak-normalised to 0.8, for the "who is this?" preview. */
    fun previewAudio(speakerId: String, maxSec: Int = 10): FloatArray {
        val chunks = unknownSpeakersAudio[speakerId].orEmpty()
        val all = FloatArray(chunks.sumOf { it.size })
        var pos = 0
        for (c in chunks) { System.arraycopy(c, 0, all, pos, c.size); pos += c.size }
        val clip = all.copyOf(minOf(all.size, maxSec * AudioUtils.SAMPLE_RATE))
        val peak = AudioUtils.peak(clip)
        return if (peak > 0f) FloatArray(clip.size) { clip[it] / peak * 0.8f } else clip
    }

    /** The user named [pending]: save the voice under that name and update the session (Python: end of the loop). */
    fun enrollPostSession(pending: PendingSpeaker, name: String) {
        saveLiveEmbedding(pending.speakerId, name, pending.genericName)
        registry.setName(pending.speakerId, name)
        registry.bind(name, pending.speakerId)
    }

    companion object {
        private const val MAX_CHUNKS_PER_SPEAKER = 30
        private const val MIN_CHUNKS_FOR_EMB = 1
        private const val ADDRESSEE_WINDOW_SEC = 30.0
        private const val VOTES_NEEDED = 3
        private const val FACE_VOICE_COOLDOWN_SEC = 120.0
        private const val ADDRESSEE = "addressee"
        private const val MENTIONED = "mentioned"

        private val WHITESPACE = PyText.regex("\\s+")
        private val GENERIC_FACE_OR_SPK = PyText.regex("^(Person_\\d+|spk_\\d+)$")

        // Portuguese gender markers in speech (they match transcribed PT-BR text).
        private val FEMALE_MARKERS = listOf(
            PyText.regex("\\beu sou a\\b"), PyText.regex("\\bsou a\\b"), PyText.regex("\\bsou mulher"),
            PyText.regex("\\bmeu nome é\\s+([a-zà-ú]+a)\\b")
        )
        private val MALE_MARKERS = listOf(
            PyText.regex("\\beu sou o\\b"), PyText.regex("\\bsou o\\b"), PyText.regex("\\bsou homem"),
            PyText.regex("\\bmeu nome é\\s+([a-zà-ú]+o)\\b")
        )

        private const val NAME = "[A-ZÀ-Ú][a-zà-ú]+"

        // The keyword is case-insensitive via (?i:...) while the captured name stays case-sensitive, so only
        // Capitalized tokens are taken as names ("and" is never captured).
        private val INTRODUCTION_PATTERNS = listOf(
            PyText.regex("(?i:meu nome é|me chamo|eu sou|sou o|sou a)\\s+($NAME(?:\\s+$NAME)?)"),
            PyText.regex("(?i:aqui é|aqui quem fala é)\\s+(?:o|a)?\\s*($NAME(?:\\s+$NAME)?)"),
            // English: "I'm Laura", "I am David", "my name is X", "this is X", "call me X"
            PyText.regex("\\b(?i:i'?m|i am|my name(?:'s| is)|this is|call me)\\s+($NAME(?:\\s+$NAME)?)")
        )

        private val VOCATIVE_START = PyText.regex("^($NAME(?:\\s+$NAME)?)\\s*,")
        private val VOCATIVE_END = PyText.regex(",\\s*($NAME(?:\\s+$NAME)?)\\s*[?.!]?\\s*$")
        private val INVITATION_PATTERNS = listOf(
            PyText.regex("(?:fala|vai|pode falar|sua vez|manda)\\s*,?\\s*($NAME)", ignoreCase = true),
            PyText.regex("(?:obrigad[oa]|valeu)\\s*,?\\s*($NAME)", ignoreCase = true)
        )
        private val MENTION_PATTERNS = listOf(
            PyText.regex("(?:d[oa]|com o|com a|que o|que a)\\s+($NAME(?:\\s+$NAME)?)"),
            PyText.regex("(?:parte|vez|turno|projeto|trabalho)\\s+d[oa]\\s+($NAME)")
        )

        /** Words that match "eu sou X" but are not names. */
        private val NOT_NAMES = setOf(
            "lésbica", "lésbico", "gay", "trans", "travesti", "hétero", "hetero",
            "bissexual", "homem", "mulher", "pessoa", "alguém", "ninguém",
            "professor", "professora", "doutor", "doutora", "engenheiro", "engenheira",
            "advogado", "advogada", "médico", "médica", "estudante", "aluno", "aluna",
            "brasileiro", "brasileira", "casado", "casada", "solteiro", "solteira",
            "cristão", "cristã", "ateu", "ateia", "católico", "católica",
            "tímido", "tímida", "ansioso", "ansiosa", "feliz", "triste",
            "novo", "nova", "velho", "velha", "gordo", "gorda", "magro", "magra",
            // Common words Whisper capitalizes / NER misidentifies
            "claro", "ou", "mas", "porque", "porém", "pois", "logo",
            "talvez", "nunca", "sempre", "ainda", "também", "aliás",
            "legal", "verdade", "exato", "beleza", "tranquilo", "show",
            "viado", "meu", "minha", "nosso", "nossa", "simplesmente",
            "maravilha", "perfeito", "exatamente", "piloto", "obrigado",
            // English: capitalized words that follow "I'm ..." but are not names
            "american", "british", "brazilian", "english", "canadian", "irish",
            "australian", "german", "french", "spanish", "italian", "indian",
            "chinese", "japanese", "scottish", "welsh", "european",
            "marketing", "sorry", "sure", "fine", "okay", "afraid", "glad",
            "ready", "done", "here", "good", "happy", "gay", "straight"
        )

        /**
         * Words that must never be treated as person names. Whisper capitalises sentence-initial words, so
         * "Yeah, ..." / "Olha, ..." look like a vocative to the regex rules.
         */
        val VOCATIVE_BLOCKLIST = setOf(
            // Portuguese discourse markers / fillers
            "pessoal", "gente", "galera", "turma", "cara", "mano", "brother",
            "professor", "professora", "doutor", "doutora",
            "obrigado", "obrigada", "desculpa", "tchau", "oi", "olá", "olha", "olhe",
            "sim", "não", "bom", "boa", "tudo", "certo", "pronto",
            "então", "agora", "aqui", "assim", "tipo", "enfim", "viu", "sabe", "entendeu",
            "claro", "ou", "mas", "porque", "porém", "pois", "logo",
            "talvez", "nunca", "sempre", "ainda", "também", "aliás",
            "legal", "verdade", "exato", "beleza", "tranquilo", "show",
            "viado", "meu", "minha", "nosso", "nossa", "dele", "dela",
            "isso", "esse", "essa", "aquele", "aquela", "qual", "quem",
            "onde", "como", "quando", "quanto", "vamos", "bora",
            "hein", "né", "pô", "putz", "caramba", "caraca",
            "maravilha", "perfeito", "exatamente", "simplesmente",
            // English discourse markers / fillers
            "yeah", "yep", "yes", "no", "nope", "ok", "okay", "so", "well",
            "right", "now", "look", "hey", "hi", "hello", "um", "uh", "hmm",
            "oh", "ah", "anyway", "alright", "actually", "basically", "like",
            "sure", "maybe", "please", "thanks", "thank", "sorry", "exactly",
            "totally", "honestly", "obviously", "listen", "wait", "and", "but",
            "or", "because", "then", "also", "just", "really", "here", "there",
            "this", "that", "what", "who", "when", "where", "why", "how",
            "which", "guys", "everyone", "folks", "man", "dude", "mean"
        )
    }
}
