package com.avtracker.mobile.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * Microphone capture (16 kHz mono) delivering float chunks in [-1, 1] with a start time on the
 * [AudioUtils.nowSec] clock. Requires the RECORD_AUDIO permission to be granted before [start].
 *
 * The timeline is anchored once at the first read and then advanced by sample count, so chunk
 * timestamps do not jitter with scheduling delays.
 */
class AudioCapture(private val onChunk: (chunk: FloatArray, startSec: Double) -> Unit) {

    @Volatile private var running = false
    private var thread: Thread? = null

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val sampleRate = AudioUtils.SAMPLE_RATE
        val minBytes = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferBytes = maxOf(minBytes, sampleRate * 2) // at least 1 s of PCM16
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC, sampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferBytes
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            throw IllegalStateException("AudioRecord failed to initialize (is RECORD_AUDIO granted?)")
        }

        running = true
        thread = Thread({ readLoop(record) }, "audio-capture").also { it.start() }
    }

    fun stop() {
        running = false
        thread?.join(1_000)
        thread = null
    }

    private fun readLoop(record: AudioRecord) {
        val chunkSamples = AudioUtils.SAMPLE_RATE / 10 // 100 ms, like the Python file-feeding path
        val pcm = ShortArray(chunkSamples)
        var anchorSec = 0.0
        var samplesRead = 0L

        try {
            record.startRecording()
            while (running) {
                val n = record.read(pcm, 0, chunkSamples)
                if (n <= 0) {
                    Log.w(TAG, "AudioRecord.read returned $n")
                    continue
                }
                if (samplesRead == 0L) anchorSec = AudioUtils.nowSec() - n.toDouble() / AudioUtils.SAMPLE_RATE
                val startSec = anchorSec + samplesRead.toDouble() / AudioUtils.SAMPLE_RATE
                val chunk = FloatArray(n) { pcm[it] / 32768f }
                samplesRead += n
                onChunk(chunk, startSec)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Audio capture failed", t)
        } finally {
            runCatching { record.stop() }
            record.release()
        }
    }

    companion object {
        private const val TAG = "AudioCapture"
    }
}
