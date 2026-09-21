package com.avtracker.mobile.session

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Streams float audio in [-1, 1] to a mono 16-bit PCM WAV file, converting like the Python `save_session`
 * (`np.clip(x * 32767, -32768, 32767).astype(np.int16)`). The Python keeps every chunk in memory and writes the file at
 * the end; on a phone the samples go to disk as they arrive and the header sizes are patched on [close].
 */
class WavWriter(val file: File, private val sampleRate: Int = 16000) : Closeable {
    private val out = RandomAccessFile(file, "rw")
    private var dataBytes = 0L
    private var closed = false

    init {
        out.setLength(0)
        out.write(header(0))
    }

    val samples: Long @Synchronized get() = dataBytes / 2

    @Synchronized
    fun write(chunk: FloatArray) {
        if (closed || chunk.isEmpty()) return
        val buf = ByteBuffer.allocate(chunk.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in chunk) buf.putShort(toPcm16(v))
        out.write(buf.array())
        dataBytes += buf.capacity()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        out.seek(0)
        out.write(header(dataBytes))
        out.close()
    }

    private fun header(dataLength: Long): ByteArray =
        ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt((36 + dataLength).toInt())
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16)                      // fmt chunk size
            putShort(1)                     // PCM
            putShort(1)                     // mono
            putInt(sampleRate)
            putInt(sampleRate * 2)          // byte rate
            putShort(2)                     // block align
            putShort(16)                    // bits per sample
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataLength.toInt())
        }.array()

    companion object {
        /** numpy's clip then astype(int16): truncation toward zero. */
        fun toPcm16(v: Float): Short = (v * 32767f).coerceIn(-32768f, 32767f).toInt().toShort()
    }
}
