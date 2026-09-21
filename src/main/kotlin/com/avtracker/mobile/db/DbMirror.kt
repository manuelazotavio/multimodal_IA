package com.avtracker.mobile.db

import android.util.Log

/**
 * The places where av-tracker mirrors what it learns into SQLite: a face auto-enrolled by the tracker, and a voice saved
 * for a face-voice binding or from live audio. Every call is guarded like the Python's `try/except Exception`: a
 * database problem is logged and never interrupts tracking.
 */
class DbMirror(private val db: TrackerDb) {

    fun faceEnrolled(name: String, embedding: FloatArray, sourceFile: String) = guarded("face embedding of $name") {
        val speakerId = db.addSpeaker(name)
        db.saveFaceEmbedding(speakerId, embedding, sourceFile)
    }

    fun voiceSaved(name: String, embedding: FloatArray, sourceFile: String) = guarded("voice embedding of $name") {
        val speakerId = db.addSpeaker(name)
        db.saveVoiceEmbedding(speakerId, embedding, sourceFile)
    }

    private fun guarded(what: String, block: () -> Unit) {
        try {
            block()
        } catch (t: Throwable) {
            Log.w(TAG, "DB sync failed for $what", t)
        }
    }

    companion object {
        private const val TAG = "DbMirror"
    }
}
