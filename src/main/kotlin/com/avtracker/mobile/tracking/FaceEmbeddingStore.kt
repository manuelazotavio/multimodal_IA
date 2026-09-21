package com.avtracker.mobile.tracking

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The face embedding directory (Python `data/face_embeddings`): one `.npy` file per enrolled face, named
 * `<name>[_<yyyymmdd>_<hhmmss>][_auto].npy`. [PersonIdTracker] loads, renames and prunes these files exactly as the
 * Python does, so the store only needs plain file operations.
 */
interface FaceEmbeddingStore {
    /** Names of the `.npy` files, sorted (a stable stand-in for os.listdir). */
    fun list(): List<String>

    fun read(file: String): FloatArray?

    fun write(file: String, embedding: FloatArray)

    fun delete(file: String)

    /** os.rename: the target is overwritten. */
    fun rename(oldFile: String, newFile: String)

    fun exists(file: String): Boolean
}

class InMemoryFaceStore(initial: Map<String, FloatArray> = emptyMap()) : FaceEmbeddingStore {
    private val files = LinkedHashMap<String, FloatArray>()

    init { initial.forEach { (k, v) -> files[k] = v.copyOf() } }

    @Synchronized override fun list() = files.keys.filter { it.endsWith(".npy") }.sorted()
    @Synchronized override fun read(file: String) = files[file]?.copyOf()
    @Synchronized override fun write(file: String, embedding: FloatArray) { files[file] = embedding.copyOf() }
    @Synchronized override fun delete(file: String) { files.remove(file) }
    @Synchronized override fun rename(oldFile: String, newFile: String) { files.remove(oldFile)?.let { files[newFile] = it } }
    @Synchronized override fun exists(file: String) = file in files
}

/** Real `.npy` files in a directory, so the folder is interchangeable with av-tracker's `data/face_embeddings`. */
class DirectoryFaceStore(private val dir: File) : FaceEmbeddingStore {
    init { dir.mkdirs() }

    @Synchronized override fun list() = dir.list { _, n -> n.endsWith(".npy") }.orEmpty().sorted()

    @Synchronized
    override fun read(file: String): FloatArray? = runCatching { Npy.read(File(dir, file).readBytes()) }
        .onFailure { Log.e(TAG, "Unreadable face embedding $file", it) }
        .getOrNull()

    @Synchronized override fun write(file: String, embedding: FloatArray) { File(dir, file).writeBytes(Npy.write(embedding)) }

    @Synchronized override fun delete(file: String) { File(dir, file).delete() }

    @Synchronized
    override fun rename(oldFile: String, newFile: String) {
        val target = File(dir, newFile)
        if (target.exists()) target.delete()
        File(dir, oldFile).renameTo(target)
    }

    @Synchronized override fun exists(file: String) = File(dir, file).exists()

    @Serializable
    private class SeedFile(val file: String, val embedding: List<Float>)

    companion object {
        private const val TAG = "FaceStore"

        /**
         * The app's face directory. On the first run it is seeded from assets/[assetDir] (av-tracker's
         * data/face_embeddings exported by scripts/export_models.py); afterwards it is the source of truth, so faces
         * enrolled in a session persist.
         */
        fun open(context: Context, assetDir: String = "face_embeddings"): DirectoryFaceStore {
            val dir = File(context.filesDir, "face_embeddings")
            val marker = File(dir, ".seeded")
            val store = DirectoryFaceStore(dir)
            if (!marker.exists()) {
                val json = Json { ignoreUnknownKeys = true }
                for (asset in context.assets.list(assetDir).orEmpty().filter { it.endsWith(".json") }) {
                    runCatching {
                        val text = context.assets.open("$assetDir/$asset").bufferedReader().use { it.readText() }
                        val seed = json.decodeFromString(SeedFile.serializer(), text)
                        store.write(seed.file, seed.embedding.toFloatArray())
                    }.onFailure { Log.e(TAG, "Failed to seed $asset", it) }
                }
                marker.writeText("seeded")
            }
            return store
        }
    }
}

/** Minimal NumPy `.npy` (format 1.0) reader/writer for one-dimensional float32/float64 arrays. */
object Npy {
    private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    fun write(values: FloatArray): ByteArray {
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (${values.size},), }"
        // magic (6) + version (2) + header length (2) + header + '\n' is a multiple of 64
        val padding = (64 - (10 + header.length + 1) % 64) % 64
        header += " ".repeat(padding) + "\n"

        val out = ByteBuffer.allocate(10 + header.length + values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        out.put(MAGIC).put(1).put(0).putShort(header.length.toShort()).put(header.toByteArray(Charsets.US_ASCII))
        for (v in values) out.putFloat(v)
        return out.array()
    }

    fun read(bytes: ByteArray): FloatArray {
        require(bytes.size > 10 && bytes.copyOfRange(0, 6).contentEquals(MAGIC)) { "not an .npy file" }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val major = bytes[6].toInt()
        val headerLen: Int
        val dataStart: Int
        if (major == 1) {
            headerLen = buf.getShort(8).toInt() and 0xFFFF
            dataStart = 10 + headerLen
        } else {
            headerLen = buf.getInt(8)
            dataStart = 12 + headerLen
        }
        val header = String(bytes, dataStart - headerLen, headerLen, Charsets.US_ASCII)
        require("'fortran_order': False" in header) { "unsupported .npy layout: $header" }
        buf.position(dataStart)
        return when {
            "'<f4'" in header -> FloatArray((bytes.size - dataStart) / 4) { buf.getFloat() }
            "'<f8'" in header -> FloatArray((bytes.size - dataStart) / 8) { buf.getDouble().toFloat() }
            else -> error("unsupported .npy dtype: $header")
        }
    }
}
