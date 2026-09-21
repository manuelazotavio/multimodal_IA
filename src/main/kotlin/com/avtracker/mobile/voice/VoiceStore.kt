package com.avtracker.mobile.voice

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.text.Normalizer

/** One stored voice embedding: the Python `data/embeddings/<name>_<yyyymmdd>_<hhmmss>[_auto|_fvbind].npy` file. */
@Serializable
data class StoredVoice(
    /** File base name without extension, e.g. "Gustavo_20260306_080555_auto". Naming rules are the Python ones. */
    val name: String,
    val embedding: List<Float>
)

/** The voice embedding directory (Python `emb_dir`). */
interface VoiceStore {
    /** Every stored embedding, ordered by name (a stable stand-in for os.listdir). */
    fun loadAll(): List<StoredVoice>

    fun baseNames(): List<String> = loadAll().map { it.name }

    fun save(name: String, embedding: FloatArray)

    /** Renames a stored file; false if [newName] already exists. */
    fun rename(oldName: String, newName: String): Boolean
}

class InMemoryVoiceStore(initial: List<StoredVoice> = emptyList()) : VoiceStore {
    private val voices = LinkedHashMap<String, FloatArray>()

    init {
        initial.forEach { voices[it.name] = it.embedding.toFloatArray() }
    }

    @Synchronized override fun loadAll() = voices.entries.sortedBy { it.key }.map { StoredVoice(it.key, it.value.toList()) }

    @Synchronized override fun save(name: String, embedding: FloatArray) { voices[name] = embedding.copyOf() }

    @Synchronized
    override fun rename(oldName: String, newName: String): Boolean {
        if (newName in voices) return false
        voices[newName] = voices.remove(oldName) ?: return false
        return true
    }
}

/**
 * Embeddings as JSON files in a directory. The file name is a filesystem-safe slug; the real (possibly accented)
 * base name lives inside the file, so "João_2026..." survives any filesystem.
 */
class DirectoryVoiceStore(private val dir: File) : VoiceStore {
    private val json = Json { ignoreUnknownKeys = true }

    init { dir.mkdirs() }

    private fun fileFor(name: String) = File(dir, "${slug(name)}.json")

    @Synchronized
    override fun loadAll(): List<StoredVoice> {
        val voices = mutableListOf<StoredVoice>()
        for (file in dir.listFiles { f -> f.extension == "json" }.orEmpty()) {
            runCatching { voices += json.decodeFromString(StoredVoice.serializer(), file.readText()) }
                .onFailure { Log.e(TAG, "Unreadable voice file ${file.name}", it) }
        }
        return voices.sortedBy { it.name }
    }

    @Synchronized
    override fun save(name: String, embedding: FloatArray) {
        fileFor(name).writeText(json.encodeToString(StoredVoice.serializer(), StoredVoice(name, embedding.toList())))
    }

    @Synchronized
    override fun rename(oldName: String, newName: String): Boolean {
        val old = fileFor(oldName)
        if (!old.exists() || fileFor(newName).exists()) return false
        val voice = json.decodeFromString(StoredVoice.serializer(), old.readText())
        save(newName, voice.embedding.toFloatArray())
        old.delete()
        return true
    }

    companion object {
        private const val TAG = "VoiceStore"

        fun slug(name: String): String {
            val ascii = Normalizer.normalize(name, Normalizer.Form.NFKD).filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
            return ascii.lowercase().replace(Regex("[^a-z0-9]+"), "_").trim('_').ifEmpty { "voice" }
        }

        /**
         * The app's voice directory. On the first run it is seeded from assets/[assetDir] (the embeddings exported
         * from av-tracker's data/embeddings); afterwards it is the source of truth, so voices learned in a session
         * persist.
         */
        fun open(context: Context, assetDir: String = "voice_embeddings"): DirectoryVoiceStore {
            val dir = File(context.filesDir, "voice_embeddings")
            val marker = File(dir, ".seeded")
            val store = DirectoryVoiceStore(dir)
            if (!marker.exists()) {
                val seed = Json { ignoreUnknownKeys = true }
                for (file in context.assets.list(assetDir).orEmpty().filter { it.endsWith(".json") }) {
                    runCatching {
                        val text = context.assets.open("$assetDir/$file").bufferedReader().use { it.readText() }
                        val voice = seed.decodeFromString(StoredVoice.serializer(), text)
                        store.save(voice.name, voice.embedding.toFloatArray())
                    }.onFailure { Log.e(TAG, "Failed to seed $file", it) }
                }
                marker.writeText("seeded")
            }
            return store
        }
    }
}
