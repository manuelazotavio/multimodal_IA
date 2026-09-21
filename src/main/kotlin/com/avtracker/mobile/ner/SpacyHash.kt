package com.avtracker.mobile.ner

/** The two hash functions the spaCy/thinc NER pipeline depends on, reproduced bit for bit. */
object SpacyHash {
    private const val M = -0x395b586ca42e166bL // 0xc6a4a7935bd1e995
    private const val R = 47

    /**
     * spaCy `hash_string`: MurmurHash64A of the UTF-8 bytes with seed 1. Lexeme attributes (NORM, PREFIX, SUFFIX, SHAPE)
     * reach the embedding tables as these 64-bit values.
     */
    fun hashString(s: String): Long {
        val data = s.toByteArray(Charsets.UTF_8)
        val len = data.size
        var h = 1L xor (len.toLong() * M)

        val blocks = len / 8
        for (i in 0 until blocks) {
            var k = 0L
            for (b in 0 until 8) k = k or ((data[i * 8 + b].toLong() and 0xFF) shl (8 * b))
            k *= M
            k = k xor (k ushr R)
            k *= M
            h = h xor k
            h *= M
        }

        val tail = blocks * 8
        val rest = len and 7
        if (rest > 0) {
            for (b in rest - 1 downTo 0) h = h xor ((data[tail + b].toLong() and 0xFF) shl (8 * b))
            h *= M
        }

        h = h xor (h ushr R)
        h *= M
        h = h xor (h ushr R)
        return h
    }

    private const val C1 = -0x783c846eeebdac2bL // 0x87c37b91114253d5
    private const val C2 = 0x4cf5ad432745937fL
    private const val F1 = -0x00ae502812aa7333L // 0xff51afd7ed558ccd
    private const val F2 = -0x3b314601e57a13adL // 0xc4ceb9fe1a85ec53

    /** thinc `MurmurHash3_x86_128_uint64`: four unsigned 32-bit bucket keys for a 64-bit id (returned as Longs). */
    fun hashIds(value: Long, seed: Int): LongArray {
        val s = seed.toLong() and 0xFFFFFFFFL
        var h1 = value
        h1 *= C1
        h1 = (h1 shl 31) or (h1 ushr 33)
        h1 *= C2
        h1 = h1 xor s
        h1 = h1 xor 8L
        var h2 = s xor 8L
        h1 += h2
        h2 += h1
        h1 = fmix(h1)
        h2 = fmix(h2)
        h1 += h2
        h2 += h1
        return longArrayOf(h1 and 0xFFFFFFFFL, h1 ushr 32, h2 and 0xFFFFFFFFL, h2 ushr 32)
    }

    private fun fmix(x: Long): Long {
        var h = x
        h = h xor (h ushr 33)
        h *= F1
        h = h xor (h ushr 33)
        h *= F2
        h = h xor (h ushr 33)
        return h
    }
}
