package com.avtracker.mobile.ui

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/** Plays a mono float clip through the speaker (the post-session "who is this?" preview). */
class PcmPlayer(private val sampleRate: Int = 16000) {
    private var track: AudioTrack? = null

    @Synchronized
    fun play(samples: FloatArray) {
        stop()
        if (samples.isEmpty()) return
        val created = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(samples.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        created.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        created.play()
        track = created
    }

    @Synchronized
    fun stop() {
        track?.let {
            runCatching { it.stop() }
            it.release()
        }
        track = null
    }
}
