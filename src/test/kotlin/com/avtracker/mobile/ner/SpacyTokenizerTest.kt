package com.avtracker.mobile.ner

import org.junit.Assert.assertEquals
import org.junit.Test

class SpacyTokenizerTest {
    @Test fun pythonAstralRangesSplitEmojiFromAccentedNames() {
        val emoji = "[\\U0001F600-\\U0001F64F]"
        val tokenizer = SpacyTokenizer(TokenizerData(
            prefix = "^$emoji", suffix = "$emoji$", infix = "(?!)", rules = emptyList()
        ))
        assertEquals(listOf("😀", "João", "🙂"), tokenizer.tokenize("😀João🙂").map { it.text })
    }
}
