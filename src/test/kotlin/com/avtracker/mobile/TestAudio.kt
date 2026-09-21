package com.avtracker.mobile

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TestAudio {
    /** Reads a 16-bit PCM mono WAV from src/test/resources by walking the RIFF chunks (headers are not always 44 bytes). */
    fun readWav(name: String): FloatArray {
        val bytes = File("src/test/resources/$name").readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12 // after "RIFF" size "WAVE"
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = buffer.getInt(pos + 4)
            if (id == "data") {
                val samples = minOf(size, bytes.size - pos - 8) / 2
                return FloatArray(samples) { buffer.getShort(pos + 8 + it * 2) / 32768f }
            }
            pos += 8 + size + (size and 1)
        }
        error("no data chunk in $name")
    }
}
