package com.avtracker.mobile.gender

import java.text.Normalizer

/** Port of the first-name gender data and helpers at the top of realtime_transcriber.py. */
object NameGender {
    const val MALE = "male"
    const val FEMALE = "female"

    // Portuguese male first names -- data set for gender detection (kept as in the Python source).
    private val NAMES_MALE = setOf(
        "lucas", "mateus", "pedro", "joão", "gabriel", "rafael", "bruno", "carlos", "andré",
        "fernando", "ricardo", "rodrigo", "marcelo", "paulo", "thiago", "vitor", "daniel",
        "augusto", "leonardo", "eduardo", "henrique", "diego", "felipe", "guilherme",
        "kauan", "marcos", "jorge", "cesar", "antonio", "luis", "francisco", "manuel",
        "xavier", "mario", "sergio", "alberto", "manoel", "josé", "jose", "joaquim",
        "arthur", "miguel", "enzo", "bernardo", "heitor", "davi", "murilo", "caio",
        "christopher", "gustavo", "william", "nicolas", "victor", "rafael", "ryan",
        "nicolas", "renan", "leandro", "alex", "alexandre", "anderson", "alan"
    )

    // Portuguese female first names -- data set for gender detection (kept as in the Python source).
    private val NAMES_FEMALE = setOf(
        "maria", "ana", "julia", "beatriz", "leticia", "amanda", "carolina", "fernanda",
        "mariana", "juliana", "patricia", "camila", "bruna", "aline", "bianca", "renata",
        "manuela", "barbara", "iris", "ines", "isis", "isadora", "ivana", "isabel", "joana",
        "josefa", "joséfina", "josiane", "jovita", "joyceana", "judite", "julieta",
        "jussara", "justina", "justine", "juvita", "kaila", "karina", "karla", "carla",
        "cassandra", "cecilia", "célia", "celina", "celia", "clara", "clarice", "claudia",
        "cleide", "cleonice", "conceição", "consuelo", "constanza", "corina",
        "cornelia", "coromila", "corsina", "cosma", "covadonga", "rosa", "rosana", "roselia",
        "vitoria", "vanessa", "valeria", "vanira", "veronica", "verena", "vera", "vidya",
        "viviana", "violeta", "vilma", "virgilia", "virginia", "antonia", "anatolia"
    )

    /** Masculine -> feminine form of Portuguese names. */
    private val MALE_TO_FEMALE = mapOf(
        "manoel" to "manuela", "manuel" to "manuela", "gabriel" to "gabriela", "ricardo" to "ricarda",
        "luis" to "luisa", "francisco" to "francisca", "antonio" to "antonia", "rio" to "ria",
        "mario" to "maria", "carlos" to "carla", "paulo" to "paula", "julio" to "julia",
        "sergio" to "serbia", "pedro" to "petra"
    )

    /** Port of _normalize_name_token: lower-case, strip accents, so Joao == João. */
    fun normalize(name: String?): String {
        if (name.isNullOrEmpty()) return ""
        val decomposed = Normalizer.normalize(name.trim().lowercase(), Normalizer.Form.NFKD)
        return decomposed.filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
    }

    private val MALE_NORMALIZED = NAMES_MALE.map(::normalize).toSet()
    private val FEMALE_NORMALIZED = NAMES_FEMALE.map(::normalize).toSet()

    fun isFemaleName(normalizedToken: String) = normalizedToken in FEMALE_NORMALIZED
    fun isMaleName(normalizedToken: String) = normalizedToken in MALE_NORMALIZED

    /** Port of _gender_of: "male"/"female" from the first name, or null if unknown. */
    fun genderOf(name: String?): String? {
        val first = name?.trim()?.split(Regex("\\s+"))?.firstOrNull().orEmpty()
        val base = normalize(first)
        if (base in FEMALE_NORMALIZED) return FEMALE
        if (base in MALE_NORMALIZED) return MALE
        return null
    }

    /**
     * Port of _correct_name_for_gender: fixes the gender suffix of the FIRST name only ("Manoel" -> "Manuela" for a
     * female voice); surnames are left alone.
     */
    fun correctForGender(name: String, gender: String?): String {
        if (gender == null) return name
        val parts = name.split(Regex("\\s+")).filter { it.isNotEmpty() }.toMutableList()
        val first = if (parts.isNotEmpty()) normalize(parts[0]) else ""
        if (gender == FEMALE) {
            if (first in MALE_TO_FEMALE) {
                parts[0] = capitalize(MALE_TO_FEMALE.getValue(first))
            } else if (first.endsWith("o") && first !in FEMALE_NORMALIZED) {
                parts[0] = capitalize(first.dropLast(1) + "a")
            }
        }
        return parts.joinToString(" ")
    }

    /** Python str.capitalize(): first character upper, the rest lower. */
    private fun capitalize(s: String) = if (s.isEmpty()) s else s.substring(0, 1).uppercase() + s.substring(1).lowercase()
}
