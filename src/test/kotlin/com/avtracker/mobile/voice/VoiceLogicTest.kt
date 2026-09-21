package com.avtracker.mobile.voice

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class VoiceLogicTest {

    private fun unit(vararg v: Float) = AudioUtils.l2Normalize(floatArrayOf(*v))

    // ---- AudioUtils -------------------------------------------------------------------------

    @Test
    fun normalize_scalesToMinus20dbRms() {
        val audio = FloatArray(16_000) { if (it % 2 == 0) 0.5f else -0.5f } // RMS 0.5
        val out = AudioUtils.normalize(audio)
        val rms = sqrt(out.map { it.toDouble() * it }.average())
        assertEquals(0.1, rms, 1e-4) // -20 dBFS
    }

    @Test
    fun normalize_limitsPeakToOne() {
        // Tiny RMS but one huge spike: scaling to -20 dBFS would push the spike above 1.0.
        val audio = FloatArray(16_000) { 0.001f }.also { it[100] = 0.9f }
        assertTrue(AudioUtils.peak(AudioUtils.normalize(audio)) <= 1f + 1e-6f)
    }

    @Test
    fun normalize_leavesSilenceUntouched() {
        val silence = FloatArray(1000)
        assertContentEquals(silence, AudioUtils.normalize(silence))
    }

    // ---- AudioTimeline ----------------------------------------------------------------------

    @Test
    fun timeline_slicesByTime() {
        val timeline = AudioTimeline(capacitySamples = 16_000 * 10)
        // 2 s of audio starting at t=100 s where sample value == its index
        timeline.append(FloatArray(32_000) { it.toFloat() }, chunkStartSec = 100.0)

        val slice = assertNotNull(timeline.slice(100.5, 101.0))
        assertEquals(8_000, slice.size)
        assertEquals(8_000f, slice.first())
        assertEquals(15_999f, slice.last())
    }

    @Test
    fun timeline_dropsOldestSamplesWhenFull() {
        val timeline = AudioTimeline(capacitySamples = 16_000) // 1 s ring
        timeline.append(FloatArray(48_000) { it.toFloat() }, chunkStartSec = 0.0)

        assertNull(timeline.slice(0.0, 1.0)) // fully overwritten
        val recent = assertNotNull(timeline.slice(2.0, 3.0))
        assertEquals(32_000f, recent.first())
        assertEquals(47_999f, recent.last())
    }

    // ---- SpeakerVerifier --------------------------------------------------------------------

    @Test
    fun verifier_returnsBestAndCandidatesAboveThreshold() {
        val verifier = SpeakerVerifier.fromMap(
            mapOf("alice" to unit(1f, 0f, 0f), "bob" to unit(0f, 1f, 0f), "carol" to unit(0.9f, 0.4f, 0f))
        )
        val result = verifier.identify(unit(1f, 0.05f, 0f))

        assertEquals("alice", result.bestName)
        assertTrue(result.bestScore > 0.99f)
        assertEquals(listOf("alice", "carol"), result.candidates.map { it.first }) // bob is below 0.80
    }

    @Test
    fun verifier_belowThresholdKeepsBestNameButNoCandidates() {
        val verifier = SpeakerVerifier.fromMap(mapOf("alice" to unit(1f, 0f)))
        val result = verifier.identify(unit(0f, 1f))

        assertEquals("alice", result.bestName)
        assertTrue(result.candidates.isEmpty())
    }

    @Test
    fun verifier_emptyDatabaseIsUnknown() {
        val result = SpeakerVerifier.fromMap(emptyMap()).identify(unit(1f, 0f))
        assertEquals("Unknown", result.bestName)
        assertEquals(-1f, result.bestScore)
    }

    @Test
    fun autoEnroll_respectsSimilarityBoundsAndCooldown() {
        var now = 1_000.0
        val verifier = SpeakerVerifier.fromMap(mapOf("alice" to unit(1f, 0f, 0f)), nowSec = { now })

        assertFalse(verifier.autoEnroll("alice", unit(0f, 1f, 0f)), "different voice must be rejected (sim < 0.65)")
        assertFalse(verifier.autoEnroll("alice", unit(1f, 0f, 0f)), "identical embedding adds no diversity (sim > 0.96)")
        assertTrue(verifier.autoEnroll("alice", unit(1f, 0.5f, 0f)))

        now += 10.0
        assertFalse(verifier.autoEnroll("alice", unit(1f, 0.6f, 0f)), "cooldown of 300 s not elapsed")
        now += 400.0
        assertTrue(verifier.autoEnroll("alice", unit(1f, 0.6f, 0f)))
    }

    @Test
    fun autoEnroll_shiftsCentroidTowardNewVoice() {
        val verifier = SpeakerVerifier.fromMap(mapOf("alice" to unit(1f, 0f)))
        val probe = unit(1f, 0.5f)
        val before = verifier.identify(probe).bestScore

        assertTrue(verifier.autoEnroll("alice", probe))
        assertTrue(verifier.identify(probe).bestScore > before)
    }

    // ---- SessionSpeakerTracker --------------------------------------------------------------

    @Test
    fun sessionTracker_sameVoiceGetsSameId_differentVoiceGetsNewId() {
        var n = 0
        val tracker = SessionSpeakerTracker(newId = { "spk_${++n}" })

        val a = tracker.getOrCreate(unit(1f, 0f, 0f))
        assertEquals(a, tracker.getOrCreate(unit(0.95f, 0.1f, 0f)))
        assertNotEquals(a, tracker.getOrCreate(unit(0f, 1f, 0f)))
    }

    @Test
    fun sessionTracker_excludedIdIsNeverReturned() {
        var n = 0
        val tracker = SessionSpeakerTracker(newId = { "spk_${++n}" })
        val a = tracker.getOrCreate(unit(1f, 0f))

        assertNotEquals(a, tracker.getOrCreate(unit(1f, 0f), excludeIds = setOf(a)))
    }

    @Test
    fun sessionTracker_speakerLimitForcesMergeWhenMinimallySimilar() {
        var n = 0
        val tracker = SessionSpeakerTracker(newId = { "spk_${++n}" }, numSpeakers = 2)
        val a = tracker.getOrCreate(unit(1f, 0f, 0f))
        tracker.getOrCreate(unit(0f, 1f, 0f)) // second speaker: limit reached

        // sim with a = 0.3: below the 0.45 match threshold but >= 0.20 -> forced into the closest speaker
        assertEquals(a, tracker.getOrCreate(unit(0.3f, 0f, 0.95f)))
    }
}
