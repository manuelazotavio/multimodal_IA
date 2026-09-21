package com.avtracker.mobile.fusion

import com.avtracker.mobile.tracking.PersonIdTracker

/**
 * [FaceTrackerBridge] over the shared [IdentityRegistry]: the name the face side last gave each track lives there.
 * [knownFaces] is the set of names with an enrolled face embedding; [onRename] forwards renames to the face side.
 * Used where there is no real face tracker (tests); the app uses [PersonIdTrackerBridge].
 */
class RegistryFaceBridge(
    private val registry: IdentityRegistry,
    private val knownFaces: () -> Set<String> = { emptySet() },
    private val onRename: (oldName: String, newName: String, onlyTrackId: Int?) -> Unit = { _, _, _ -> }
) : FaceTrackerBridge {
    override fun trackName(trackId: Int): String? = registry.faceNameOf(trackId)

    override fun knowsFace(name: String): Boolean = name in knownFaces()

    override fun renamePerson(oldName: String, newName: String, onlyTrackId: Int?) = onRename(oldName, newName, onlyTrackId)
}

/**
 * [FaceTrackerBridge] straight onto the face tracker, like the Python transcriber, which reads
 * `face_tracker._track_to_name` / `known_embeddings` and calls `face_tracker.rename_person`.
 */
class PersonIdTrackerBridge(private val tracker: PersonIdTracker) : FaceTrackerBridge {
    override fun trackName(trackId: Int): String? = tracker.trackName(trackId)

    override fun knowsFace(name: String): Boolean = tracker.knowsFace(name)

    override fun renamePerson(oldName: String, newName: String, onlyTrackId: Int?) = tracker.renamePerson(oldName, newName, onlyTrackId)
}
