package com.avtracker.mobile.fusion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PyTextTest {
    @Test fun accentedNamesRemainWholeWords() {
        assertEquals(listOf("João", "Vinícius"),
            PyText.regex("\\b\\w+\\b").findAll("João, Vinícius").map { it.value }.toList())
    }

    @Test fun namesMatchWithoutCaseAndKeepAccents() {
        assertTrue(PyText.regex("\\bjoão\\b", ignoreCase = true).containsMatchIn("JOÃO fala"))
    }

    @Test fun homonymSuffixAcceptsUnicodeWhitespaceAndDigits() {
        assertEquals("Vinícius", PyText.stripHomonymSuffix("Vinícius\u00a0２"))
    }
}
