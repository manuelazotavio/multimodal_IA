package com.avtracker.mobile

import com.avtracker.mobile.profile.VisualProfileDatabase
import com.avtracker.mobile.tracking.Roi
import com.avtracker.mobile.tracking.SimpleTracker
import com.avtracker.mobile.tracking.TrackedPerson
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TrackingLogicTest {

    @Test
    fun simpleTracker_keepsSameIdAcrossOverlappingFrames() {
        val tracker = SimpleTracker()

        val first = tracker.update(listOf(Roi(10, 10, 60, 60)))
        assertEquals(1, first.size)
        val id = first[0].trackId

        // Slightly moved box next frame, still heavily overlapping -> same id.
        val second = tracker.update(listOf(Roi(12, 12, 62, 62)))
        assertEquals(id, second[0].trackId)
    }

    @Test
    fun simpleTracker_assignsNewIdWhenTrackDisappearsAndReappears() {
        val tracker = SimpleTracker()

        val first = tracker.update(listOf(Roi(0, 0, 50, 50)))
        val firstId = first[0].trackId

        tracker.update(emptyList()) // detection missed a frame -> track dropped

        val third = tracker.update(listOf(Roi(0, 0, 50, 50)))
        assertTrue(third[0].trackId != firstId)
    }

    @Test
    fun trackedPerson_averagesEmbeddingHistory() {
        val person = TrackedPerson(1, Roi(0, 0, 10, 10), floatArrayOf(1f, 0f))
        person.update(Roi(0, 0, 10, 10), floatArrayOf(0f, 1f))

        val avg = person.averageEmbedding()!!
        assertEquals(0.5f, avg[0])
        assertEquals(0.5f, avg[1])
    }

    @Test
    fun trackedPerson_displayName_showsIdWhenUnidentified() {
        val person = TrackedPerson(7, Roi(0, 0, 10, 10))
        assertEquals("Person 7", person.displayName())
    }

    @Test
    fun trackedPerson_displayName_showsNameAndConfidenceWhenIdentified() {
        val person = TrackedPerson(7, Roi(0, 0, 10, 10))
        person.setIdentity("alice", 0.87f)
        assertEquals("alice (0.87)", person.displayName())
    }

    @Test
    fun visualProfileDatabase_identifiesBestMatchAboveThreshold() {
        val db = VisualProfileDatabase.fromMap(
            mapOf(
                "alice" to floatArrayOf(1f, 0f, 0f),
                "bob" to floatArrayOf(0f, 1f, 0f)
            )
        )

        val (name, confidence) = db.identifyPerson(floatArrayOf(0.9f, 0.1f, 0f), threshold = 0.6f)
        assertEquals("alice", name)
        assertTrue(confidence > 0.9f)
    }

    @Test
    fun visualProfileDatabase_returnsNullBelowThreshold() {
        val db = VisualProfileDatabase.fromMap(mapOf("alice" to floatArrayOf(1f, 0f, 0f)))

        val (name, _) = db.identifyPerson(floatArrayOf(0f, 1f, 0f), threshold = 0.6f)
        assertNull(name)
    }
}
