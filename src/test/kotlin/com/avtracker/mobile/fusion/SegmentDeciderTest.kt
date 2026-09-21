package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.voice.SessionSpeakerTracker
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SegmentDeciderTest {

    private val registry = IdentityRegistry()
    private val sessionTracker = SessionSpeakerTracker(newId = registry::newPersonId)
    private fun decider(numSpeakers: Int? = null) = SegmentDecider(registry, sessionTracker, numSpeakers = numSpeakers)

    private fun voice(vararg v: Float) = AudioUtils.l2Normalize(floatArrayOf(*v))

    private fun evidence(
        rawName: String = "Unknown",
        rawConf: Float = 0f,
        candidates: List<Pair<String, Float>> = emptyList(),
        asdTrack: Int? = null,
        guess: Int? = null,
        asdAvailable: Boolean = true,
        hint: String? = null,
        embedding: FloatArray? = voice(1f, 0f, 0f)
    ) = SegmentEvidence(rawName, rawConf, candidates, asdTrack, { guess }, asdAvailable, hint, embedding)

    /** Puts faces on screen: trackId -> (personId, displayName). */
    private fun faces(vararg faces: Triple<Int, String, String>) {
        faces.forEach { (_, pid, name) -> registry.setName(pid, name) }
        registry.setActiveFaces(faces.associate { it.first to it.second })
    }

    // ---- voice ------------------------------------------------------------------------------

    @Test
    fun strongVoiceMatchWins_andRegistersThePerson() {
        val d = decider().decide(evidence("Alice", 0.9f, listOf("Alice" to 0.9f, "Bob" to 0.5f)))

        assertEquals("VERIFIER_HIGH", d.type)
        assertEquals("Alice", d.verifiedName)
        assertEquals(registry.personFor("Alice"), d.speakerId)
        assertEquals("Alice", registry.nameOf(d.speakerId))
    }

    @Test
    fun nearTieBetweenTwoVoicesIsNotTrusted() {
        val d = decider().decide(evidence("Alice", 0.86f, listOf("Alice" to 0.86f, "Bob" to 0.84f)))
        assertTrue(d.type.startsWith("SESSION"), "margin 0.02 < 0.04, no faces -> falls to the session tracker, got ${d.label}")
    }

    @Test
    fun weakVoiceMatchNeedsTheActiveSpeakerToCorroborateIt() {
        faces(Triple(1, "spk_010", "Person_1"))

        val withAsd = decider().decide(evidence("Alice 2", 0.70f, asdTrack = 1))
        assertEquals("VERIFIER_WEAK", withAsd.type)
        assertEquals("Alice", withAsd.verifiedName, "homonym suffix is stripped")

        val withoutAsd = decider().decide(evidence("Alice", 0.70f, asdTrack = null))
        assertNotEquals("VERIFIER_WEAK", withoutAsd.type)
    }

    // ---- faces ------------------------------------------------------------------------------

    @Test
    fun activeSpeakerFaceIsUsedWhenTheVoiceIsUnknown() {
        faces(Triple(1, "spk_010", "Manuela"), Triple(2, "spk_011", "Heitor"))

        val d = decider().decide(evidence(asdTrack = 2))

        assertEquals("ASD", d.type)
        assertEquals("spk_011", d.speakerId)
        assertEquals("Heitor", d.verifiedName)
        assertEquals(2, d.asdTrackId)
    }

    @Test
    fun aSingleNamedFaceIsTheSpeakerWhenOnlyOnePersonIsExpected() {
        faces(Triple(1, "spk_010", "Manuela"))

        assertEquals("FACE_ONLY", decider().decide(evidence()).type)
        assertTrue(decider(numSpeakers = 3).decide(evidence()).type.startsWith("SESSION"), "with 3 people, one visible face is not proof")
    }

    @Test
    fun aSingleAnonymousFaceNeedsSomeVoiceSimilarity() {
        faces(Triple(1, "spk_010", "Person_1"))

        assertEquals("SINGLE_FACE", decider().decide(evidence(rawName = "Alice", rawConf = 0.4f)).type)
        assertTrue(decider().decide(evidence(rawName = "Alice", rawConf = 0.1f, asdAvailable = false)).type.startsWith("SESSION"))
    }

    @Test
    fun weakerAsdGuessIsUsedWhenThereIsNoConfirmedSpeaker() {
        faces(Triple(1, "spk_010", "Manuela"), Triple(2, "spk_011", "Heitor"))

        val d = decider().decide(evidence(guess = 2))
        assertEquals("ASD_GUESS", d.type)
        assertEquals("spk_011", d.speakerId)
    }

    @Test
    fun asdGuessIsRejectedWhenThatFaceIsAlreadyBoundToAnotherVoiceSession() {
        faces(Triple(1, "spk_010", "Manuela"), Triple(2, "spk_011", "Heitor"))
        val decider = decider()
        decider.sessionToFace["session_A"] = "spk_011" // Heitor's face was already confirmed for session_A

        val d = decider.decide(evidence(guess = 2, hint = "session_B"))

        assertNotEquals("ASD_GUESS", d.type)
        assertEquals("SESSION_HINT", d.type)
        assertEquals("session_B", d.speakerId)
    }

    // ---- voice sessions ---------------------------------------------------------------------

    @Test
    fun sameUnknownVoiceKeepsTheSameSessionIdWithoutAnyFace() {
        val decider = decider()
        val first = decider.decide(evidence(embedding = voice(1f, 0f, 0f)))
        val second = decider.decide(evidence(embedding = voice(0.95f, 0.1f, 0f)))
        val other = decider.decide(evidence(embedding = voice(0f, 1f, 0f)))

        assertEquals("SESSION_TRACKER", first.type)
        assertEquals(first.speakerId, second.speakerId)
        assertNotEquals(first.speakerId, other.speakerId)
    }

    @Test
    fun aConfirmedFaceBindsTheVoiceSession_soLaterSegmentsResolveToThatPerson() {
        faces(Triple(1, "spk_010", "Manuela"))
        val decider = decider()

        val confirmed = decider.decide(evidence(asdTrack = 1, hint = "session_A"))
        assertEquals("ASD", confirmed.type)
        assertEquals("spk_010", decider.sessionToFace["session_A"])

        registry.setActiveFaces(emptyMap()) // Manuela leaves the frame
        val later = decider.decide(evidence(hint = "session_A"))
        assertEquals("SESSION_FACE", later.type)
        assertEquals("spk_010", later.speakerId)
        assertEquals("Manuela", later.verifiedName)
    }

    @Test
    fun namedDecisionsMarkThePersonAsIdentified_genericOnesDoNot() {
        faces(Triple(1, "spk_010", "Manuela"), Triple(2, "spk_011", "Person_2"))
        val decider = decider()

        decider.decide(evidence(asdTrack = 1))
        decider.decide(evidence(asdTrack = 2))

        assertEquals(setOf("spk_010"), decider.identifiedSpeakers)
    }

    @Test
    fun noEvidenceAtAllStillYieldsASpeaker() {
        val d = decider().decide(evidence(embedding = null))
        assertNull(d.verifiedName)
        assertTrue(d.speakerId.startsWith("spk_"))
    }
}
