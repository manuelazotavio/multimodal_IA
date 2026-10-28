package com.avtracker.mobile.app

import android.graphics.Bitmap
import com.avtracker.mobile.fusion.SpeakerNaming
import com.avtracker.mobile.fusion.TranscriptEntry
import com.avtracker.mobile.pipeline.FrameResult
import com.avtracker.mobile.pipeline.HeadTrackerPipeline
import com.avtracker.mobile.session.SavedSession

/** A camera session can run independently of the optional local audio pipeline. */
interface TrackingSession : AutoCloseable {
    val facePipeline: HeadTrackerPipeline
    val localAudioEnabled: Boolean get() = true
    val transcript: List<TranscriptEntry> get() = emptyList()
    fun onFrame(frame: Bitmap, result: FrameResult, timestampSec: Double): FrameOverlay
    fun startAudio() { error("This session delegates audio to VocalID") }
    fun stopAudio() {}
    fun pendingSpeakers(): List<SpeakerNaming.PendingSpeaker> = emptyList()
    fun previewAudio(pending: SpeakerNaming.PendingSpeaker): FloatArray = error("Audio belongs to VocalID")
    fun enrollSpeaker(pending: SpeakerNaming.PendingSpeaker, name: String) { error("Voices belong to VocalID") }
    fun saveSession(): SavedSession
}
