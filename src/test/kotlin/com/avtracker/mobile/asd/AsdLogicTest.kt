package com.avtracker.mobile.asd

import com.avtracker.mobile.audio.AudioTimeline
import java.util.concurrent.Executor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AsdLogicTest {

    /** Scores each crop set by its first byte: the "speaking" faces are the ones whose crops start with 200. */
    private class FakeScorer(private val speaking: Map<Int, Float>) : LightAsdScorer {
        val calls = mutableListOf<Pair<List<ByteArray>, Int>>()
        override fun score(crops: List<ByteArray>, audio: FloatArray): FloatArray {
            calls += crops to audio.size
            val trackTag = crops.first()[0].toInt() and 0xFF
            return FloatArray(25) { speaking[trackTag] ?: 0f }
        }
    }

    private fun crop(tag: Int) = ByteArray(112 * 112).also { it[0] = tag.toByte() }

    private fun timelineWithAudio(fromSec: Double, seconds: Int): AudioTimeline =
        AudioTimeline().also { it.append(FloatArray(16_000 * seconds) { 0.01f }, fromSec) }

    private val direct = Executor { it.run() }

    private fun feed(detector: LightAsdDetector, tags: Map<Int, Int>, from: Double, to: Double, step: Double = 0.1) {
        var t = from
        while (t <= to + 1e-9) {
            detector.update(tags.map { (track, tag) -> FaceSample(track, crop(tag), null) }, t)
            t += step
        }
    }

    @Test
    fun picksTheTrackWithTheHigherScore() {
        val scorer = FakeScorer(mapOf(1 to 0.9f, 2 to 0.1f))
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })

        feed(detector, mapOf(10 to 1, 20 to 2), from = 101.0, to = 102.4)

        assertTrue(detector.hasScores(100.0, 103.0))
        assertEquals(10, detector.getActiveSpeaker(100.0, 103.0))
    }

    @Test
    fun returnsNullWhenTwoFacesAreTooClose() {
        val scorer = FakeScorer(mapOf(1 to 0.80f, 2 to 0.75f))
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })

        feed(detector, mapOf(10 to 1, 20 to 2), from = 101.0, to = 102.4)

        assertNull(detector.getActiveSpeaker(100.0, 103.0)) // margin 0.05 < 0.15
        // best-guess uses the relaxed margin (0.10) -> still ambiguous, but the min score is looser
        assertNull(detector.getActiveSpeaker(100.0, 103.0, minScore = 0.30f, margin = 0.10f))
    }

    @Test
    fun aSingleFaceBelowTheMinimumScoreIsNotTheSpeaker() {
        val scorer = FakeScorer(mapOf(1 to 0.40f))
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })

        feed(detector, mapOf(10 to 1), from = 101.0, to = 102.4)

        assertNull(detector.getActiveSpeaker(100.0, 103.0)) // 0.40 < 0.50
        assertEquals(10, detector.getActiveSpeaker(100.0, 103.0, minScore = 0.30f, margin = 0.10f))
    }

    @Test
    fun resamplesToExactly25FramesAndPassesTheAudioWindow() {
        val scorer = FakeScorer(mapOf(1 to 0.9f))
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })

        feed(detector, mapOf(10 to 1), from = 101.0, to = 102.4)

        assertTrue(scorer.calls.isNotEmpty())
        scorer.calls.forEach { (crops, audioSamples) ->
            assertEquals(25, crops.size)
            assertTrue(audioSamples in 15_900..16_100, "audio window should be ~1 s, was $audioSamples")
        }
    }

    @Test
    fun doesNotRunWithoutAudioOrWithTooFewCrops() {
        val scorer = FakeScorer(mapOf(1 to 0.9f))

        val noAudio = LightAsdDetector(scorer, AudioTimeline(), direct, clock = { 102.0 })
        feed(noAudio, mapOf(10 to 1), from = 101.0, to = 102.4)
        assertTrue(scorer.calls.isEmpty())
        assertFalse(noAudio.hasScores(0.0, 1e9))

        val fewCrops = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })
        fewCrops.update(listOf(FaceSample(10, crop(1), null)), 101.0)
        fewCrops.update(listOf(FaceSample(10, crop(1), null)), 101.7) // only 2 crops in the last second
        assertTrue(scorer.calls.isEmpty())
    }

    @Test
    fun overlappingWindowsDoNotDuplicateScoredFrames() {
        val scorer = FakeScorer(mapOf(1 to 0.9f))
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 6), direct, clock = { 103.0 })

        feed(detector, mapOf(10 to 1), from = 101.0, to = 103.0)

        // Frames in [101.4, 102.0] were covered by at least two overlapping windows; the buffer must hold one score each.
        val mean = detector.getScore(10)
        assertTrue(mean in 0.89f..0.91f)
        assertEquals(10, detector.getActiveSpeaker(100.0, 104.0))
    }

    @Test
    fun speakingHysteresisTurnsOnAfterSustainedScoreAndOffAfterSustainedSilence() {
        var now = 101.0
        val speakingTags = mutableMapOf(1 to 0.9f)
        val scorer = FakeScorer(speakingTags)
        val detector = LightAsdDetector(scorer, timelineWithAudio(100.0, 30), direct, clock = { now })

        // Speech: feed a few seconds and poll like the UI does.
        var turnedOn = false
        while (now < 104.0) {
            detector.update(listOf(FaceSample(10, crop(1), null)), now)
            if (detector.isSpeakingNow(10)) turnedOn = true
            now += 0.1
        }
        assertTrue(turnedOn, "should turn on after the score stays above 0.55 for 0.25 s")

        // Silence: same face now scores low.
        speakingTags[1] = 0.05f
        var turnedOff = false
        while (now < 110.0) {
            detector.update(listOf(FaceSample(10, crop(1), null)), now)
            if (!detector.isSpeakingNow(10)) turnedOff = true
            now += 0.1
        }
        assertTrue(turnedOff, "should turn off after the score stays below 0.35 for 0.5 s")
    }

    // ---- pixel-difference fallback ----------------------------------------------------------

    private fun face(mouthShade: Int, upperShade: Int = 100): GrayImage {
        val w = 40
        val h = 40
        val data = ByteArray(w * h) { i -> (if (i / w >= (h * 0.6).toInt()) mouthShade else upperShade).toByte() }
        return GrayImage(w, h, data)
    }

    @Test
    fun pixelDiff_pointsAtTheFaceWhoseMouthMoves() {
        val detector = ActiveSpeakerDetector(lightAsd = null)
        var t = 10.0
        for (frame in 0 until 20) {
            detector.update(
                listOf(
                    FaceSample(1, null, face(mouthShade = if (frame % 2 == 0) 40 else 200)), // mouth moves, forehead still
                    FaceSample(2, null, face(mouthShade = 90))                                // static
                ),
                t
            )
            t += 0.1
        }
        assertEquals(1, detector.getActiveSpeaker(10.0, 12.0))
        assertEquals(1, detector.getBestGuess(10.0, 12.0))
    }

    @Test
    fun pixelDiff_rejectsWholeFaceMotionAsOcclusion() {
        val detector = ActiveSpeakerDetector(lightAsd = null)
        var t = 10.0
        for (frame in 0 until 20) {
            val shade = if (frame % 2 == 0) 40 else 200
            // The whole face (forehead too) changes: a hand or the camera moving, not speech.
            detector.update(listOf(FaceSample(1, null, face(mouthShade = shade, upperShade = shade))), t)
            t += 0.1
        }
        assertNull(detector.getActiveSpeaker(10.0, 12.0))
    }

    @Test
    fun pixelDiff_noFacesMeansNoSpeaker() {
        val detector = ActiveSpeakerDetector(lightAsd = null)
        assertNull(detector.getActiveSpeaker(0.0, 100.0))
        assertNull(detector.getBestGuess(0.0, 100.0))
    }

    @Test
    fun combinedDetectorPrefersLightAsdWhenItHasScores() {
        val scorer = FakeScorer(mapOf(1 to 0.9f, 2 to 0.1f))
        val light = LightAsdDetector(scorer, timelineWithAudio(100.0, 5), direct, clock = { 102.0 })
        val detector = ActiveSpeakerDetector(light)

        var t = 101.0
        while (t <= 102.4) {
            // Pixel-diff alone would pick track 20 (its mouth moves); Light-ASD says track 10 speaks.
            detector.update(
                listOf(
                    FaceSample(10, crop(1), face(mouthShade = 90)),
                    FaceSample(20, crop(2), face(mouthShade = if ((t * 10).toInt() % 2 == 0) 40 else 200))
                ),
                t
            )
            t += 0.1
        }
        assertEquals(10, detector.getActiveSpeaker(100.0, 103.0))
    }
}
