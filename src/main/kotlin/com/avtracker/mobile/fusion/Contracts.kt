package com.avtracker.mobile.fusion

import com.avtracker.mobile.diarization.SpeakerTurn
import com.avtracker.mobile.whisper.TranscriptionResult

/** Speech turns inside a chunk of audio (implemented by Diarizer). */
fun interface TurnDetector {
    fun turns(audio: FloatArray): List<SpeakerTurn>
}

/** ECAPA voice embeddings (implemented by VoiceEmbedder). */
interface VoiceEncoder {
    /** Raw embedding, or null if the audio is too short. */
    fun embed(audio: FloatArray): FloatArray?

    /** Unit-norm embedding, or null if shorter than 0.4 s. */
    fun embedNormalized(audio: FloatArray): FloatArray?
}

/** Speech-to-text (implemented by WhisperTranscriber). */
fun interface SpeechRecognizer {
    fun transcribe(audio: FloatArray, language: String): TranscriptionResult?
}

/** Which face was speaking in a time window (implemented by ActiveSpeakerDetector). */
interface SpeakerActivity {
    fun getActiveSpeaker(timeStart: Double, timeEnd: Double): Int?
    fun getBestGuess(timeStart: Double, timeEnd: Double): Int?
}
