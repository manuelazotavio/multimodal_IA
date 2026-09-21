package com.avtracker.mobile.fusion

object TextSimilarity {
    /**
     * Python's `difflib.SequenceMatcher(None, a, b).ratio()`: 2*M/T where M counts the characters of the
     * longest common block, then recursively of the blocks left and right of it (Ratcliff/Obershelp).
     */
    fun ratio(a: String, b: String): Double {
        if (a.isEmpty() && b.isEmpty()) return 1.0
        return 2.0 * matchingChars(a, 0, a.length, b, 0, b.length) / (a.length + b.length)
    }

    private fun matchingChars(a: String, aLo: Int, aHi: Int, b: String, bLo: Int, bHi: Int): Int {
        if (aLo >= aHi || bLo >= bHi) return 0

        // Longest common substring; ties go to the earliest position in a, then in b, like difflib.
        var bestSize = 0
        var bestI = aLo
        var bestJ = bLo
        var previous = IntArray(bHi - bLo + 1)
        for (i in aLo until aHi) {
            val current = IntArray(bHi - bLo + 1)
            for (j in bLo until bHi) {
                if (a[i] == b[j]) {
                    val length = previous[j - bLo] + 1
                    current[j - bLo + 1] = length
                    if (length > bestSize) {
                        bestSize = length
                        bestI = i - length + 1
                        bestJ = j - length + 1
                    }
                }
            }
            previous = current
        }
        if (bestSize == 0) return 0

        return bestSize +
            matchingChars(a, aLo, bestI, b, bLo, bestJ) +
            matchingChars(a, bestI + bestSize, aHi, b, bestJ + bestSize, bHi)
    }
}
