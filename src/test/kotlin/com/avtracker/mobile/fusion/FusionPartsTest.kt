package com.avtracker.mobile.fusion

import com.avtracker.mobile.audio.AudioTimeline
import com.avtracker.mobile.audio.AudioUtils
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FusionPartsTest {

    // ---- TextSimilarity: expected values come from Python's difflib.SequenceMatcher(None, a, b).ratio() ----

    @Test
    fun textSimilarity_matchesPythonDifflib() {
        val cases = listOf(
            Triple("hello world", "hello there world", 0.785714),
            Triple("i think so", "i think so.", 0.952381),
            Triple("abc", "xyz", 0.0),
            Triple("", "", 1.0),
            Triple("the quick brown fox", "the quick brown fox jumps", 0.863636),
            Triple("bom dia a todos", "bom dia todos", 0.928571),
            Triple("night", "nacht", 0.6),
            Triple("abcabc", "abc", 0.666667),
            Triple("i think maybe mario would have covered us", "i think maru would have covered us", 0.88),
            Triple("no i said no", "no respond no", 0.64),
            Triple("ab", "ba", 0.5),
            Triple("abxcd", "cdxab", 0.4)
        )
        for ((a, b, expected) in cases) {
            assertEquals(expected, TextSimilarity.ratio(a, b), 1e-6, "ratio('$a', '$b')")
        }
    }

    // ---- TranscriptFilter -------------------------------------------------------------------

    @Test
    fun filter_acceptsNormalSpeech() {
        assertTrue(TranscriptFilter.accept("I think we should start the meeting now.", -0.3f, 0.05f))
    }

    @Test
    fun filter_rejectsWhenWhisperDoubtsThereIsSpeech() {
        assertFalse(TranscriptFilter.accept("I think we should start the meeting now.", -0.3f, 0.6f))
    }

    @Test
    fun filter_rejectsSubtitleAndYoutubeArtifactsAndTinyText() {
        assertFalse(TranscriptFilter.accept("Subtitles by the Amara.org community", -0.2f, 0.01f))
        assertFalse(TranscriptFilter.accept("Não se esqueça de se inscrever no canal", -0.2f, 0.01f))
        assertFalse(TranscriptFilter.accept("ok", -0.2f, 0.01f))
    }

    @Test
    fun filter_rejectsRepetitionHallucination() {
        assertFalse(TranscriptFilter.accept("Thank you. Thank you. Thank you.", -0.2f, 0.01f))
        assertTrue(TranscriptFilter.accept("Thank you. Thanks a lot. Thank you.", -0.2f, 0.01f))
    }

    @Test
    fun filter_rejectsCaptionWordsOnlyWhenTheyAreTheWholeText() {
        assertFalse(TranscriptFilter.accept("Música.", -0.2f, 0.01f))
        assertTrue(TranscriptFilter.accept("Vamos ouvir a música agora.", -0.2f, 0.01f))
    }

    @Test
    fun filter_shortLowConfidenceAndSingleWordOnNoise() {
        assertFalse(TranscriptFilter.accept("Yes, right.", -1.2f, 0.01f))      // short + low logprob
        assertFalse(TranscriptFilter.accept("Yes, right.", -0.2f, 0.4f))       // short + moderate no-speech
        assertFalse(TranscriptFilter.accept("Hmmmmm...", -0.2f, 0.2f))         // one short word, elevated no-speech
        assertTrue(TranscriptFilter.accept("Yes, that is right.", -0.2f, 0.2f))
    }

    // ---- AudioChunker -----------------------------------------------------------------------

    private fun timelineOf(seconds: Double, start: Double = 100.0) = AudioTimeline().also {
        it.append(FloatArray((seconds * AudioUtils.SAMPLE_RATE).toInt()), start)
    }

    @Test
    fun chunker_followsThe8s_15s_then5sStepSchedule() {
        val chunker = AudioChunker()
        assertNull(chunker.next(AudioTimeline()), "nothing before any audio")
        assertNull(chunker.next(timelineOf(7.9)))

        val first = assertNotNull(chunker.next(timelineOf(8.0)))
        assertEquals(100.0, first.startSec, 1e-9); assertEquals(108.0, first.endSec, 1e-9); assertEquals(0, first.newZoneStartSamples)

        val second = assertNotNull(chunker.next(timelineOf(23.0)))
        assertEquals(108.0, second.startSec, 1e-9); assertEquals(123.0, second.endSec, 1e-9); assertEquals(0, second.newZoneStartSamples)

        val third = assertNotNull(chunker.next(timelineOf(28.0)))
        assertEquals(113.0, third.startSec, 1e-9); assertEquals(128.0, third.endSec, 1e-9)
        assertEquals(10 * AudioUtils.SAMPLE_RATE, third.newZoneStartSamples) // 10 s of context, 5 s new

        val fourth = assertNotNull(chunker.next(timelineOf(33.0)))
        assertEquals(118.0, fourth.startSec, 1e-9)
    }

    // ---- FaceIdentitySync -------------------------------------------------------------------

    @Test
    fun faceSync_newTrackGetsAPersonAndKeepsItAcrossFrames() {
        val registry = IdentityRegistry()
        val sync = FaceIdentitySync(registry)

        val first = sync.sync(listOf(FaceResult(1, "Person_1", 0.9f)))
        val pid = first.getValue(1)
        assertEquals("Person_1", registry.nameOf(pid))
        assertEquals(mapOf(1 to pid), sync.sync(listOf(FaceResult(1, "Person_1", 0.9f))))
    }

    @Test
    fun faceSync_realNameUpgradesAGenericFace() {
        val registry = IdentityRegistry()
        val sync = FaceIdentitySync(registry)
        val pid = sync.sync(listOf(FaceResult(1, "Person_1", 0.5f))).getValue(1)

        sync.sync(listOf(FaceResult(1, "Manuela", 0.9f)))

        assertEquals("Manuela", registry.nameOf(pid))
        assertEquals(pid, registry.personFor("Manuela"))
    }

    @Test
    fun faceSync_aVoiceKnownNameReusesItsPerson() {
        val registry = IdentityRegistry()
        val voicePerson = registry.registerKnown("Heitor") // heard before it was ever seen
        val sync = FaceIdentitySync(registry)

        assertEquals(voicePerson, sync.sync(listOf(FaceResult(7, "Heitor", 0.9f))).getValue(7))
    }

    @Test
    fun faceSync_oneNameCannotLabelTwoFaces() {
        val registry = IdentityRegistry()
        val sync = FaceIdentitySync(registry)
        val current = sync.sync(listOf(FaceResult(1, "Isabel", 0.9f), FaceResult(2, "Isabel", 0.6f)))

        val labels = sync.labels(listOf(FaceResult(1, "Isabel", 0.9f), FaceResult(2, "Isabel", 0.6f)), current)
        assertEquals(listOf(false, true), labels.map { it.unknown })
        assertTrue(labels[0].text.startsWith("Isabel"))
        assertTrue(labels[1].text.startsWith("Unknown"))
    }

    @Test
    fun faceSync_publishesFacesAndTheirTrackerNames() {
        val registry = IdentityRegistry()
        val current = FaceIdentitySync(registry).sync(listOf(FaceResult(3, "Gustavo", 0.8f)))

        assertEquals(current, registry.activeFacesSnapshot())
        assertEquals("Gustavo", registry.faceNameOf(3))
    }
}
