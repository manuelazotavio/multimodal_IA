package com.avtracker.mobile.fusion

/**
 * [FaceTrackerBridge] over the shared [IdentityRegistry]: the name the face side last gave each track lives there.
 * [knownFaces] is the set of names with an enrolled face embedding; [onRename] forwards renames to the face side.
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
