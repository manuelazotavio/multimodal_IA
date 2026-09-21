package com.avtracker.mobile.fusion

/**
 * The `shared_state` dict of run_multimodal_tracker.py, shared by the video and audio threads:
 * one stable person_id per human, whichever modality noticed them first.
 */
class IdentityRegistry {
    private var counter = 0
    private val personNames = LinkedHashMap<String, String>()     // person_id -> display name (insertion-ordered like the Python dict)
    private val embeddingToPerson = HashMap<String, String>()     // known name -> person_id
    private var activeFaces: Map<Int, String> = emptyMap()        // face track_id -> person_id (insertion-ordered)
    private var faceNames: Map<Int, String> = emptyMap()          // face track_id -> name the face tracker gave it

    @Synchronized
    fun newPersonId(): String {
        counter++
        return "spk_%03d".format(java.util.Locale.ROOT, counter)
    }

    @Synchronized fun nameOf(personId: String): String? = personNames[personId]

    @Synchronized fun setName(personId: String, name: String) { personNames[personId] = name }

    @Synchronized fun personFor(name: String): String? = embeddingToPerson[name]

    @Synchronized fun bind(name: String, personId: String) { embeddingToPerson[name] = personId }

    @Synchronized fun knownPersonIds(): Set<String> = personNames.keys.toSet()

    @Synchronized fun namesSnapshot(): Map<String, String> = LinkedHashMap(personNames)

    /** name -> person_id bindings (Python `_emb_to_pid`). */
    @Synchronized fun bindingsSnapshot(): Map<String, String> = HashMap(embeddingToPerson)

    @Synchronized fun setActiveFaces(faces: Map<Int, String>, names: Map<Int, String> = emptyMap()) {
        activeFaces = LinkedHashMap(faces)
        faceNames = LinkedHashMap(names)
    }

    @Synchronized fun activeFacesSnapshot(): Map<Int, String> = LinkedHashMap(activeFaces)

    /** The name the face tracker currently gives [trackId] (Python: face_tracker._track_to_name). */
    @Synchronized fun faceNameOf(trackId: Int): String? = faceNames[trackId]

    /** Registers a voice-known name up front (Python: pre-registering verifier.embeddings before the transcriber starts). */
    @Synchronized
    fun registerKnown(name: String): String {
        embeddingToPerson[name]?.let { return it }
        val pid = newPersonId()
        embeddingToPerson[name] = pid
        personNames[pid] = name
        return pid
    }

    companion object {
        private val GENERIC = Regex("^(Person_\\d+|spk_\\d+|Desconhecido_\\d+|Unknown(_\\d+)?)$")

        /** Port of _is_generic_name: placeholder names that are not real identities. */
        fun isGeneric(name: String?): Boolean = name != null && GENERIC.matches(name)
    }
}
