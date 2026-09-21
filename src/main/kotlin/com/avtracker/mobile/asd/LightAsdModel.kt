package com.avtracker.mobile.asd

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.voice.AssetFiles
import java.io.File
import java.nio.FloatBuffer

/** Scores 25 face crops + the matching audio with Light-ASD. An interface so the detector logic can be tested without the model. */
interface LightAsdScorer {
    /** [crops]: 25 grayscale 112x112 crops (row-major bytes); [audio]: ~1 s of 16 kHz audio. Returns 25 speaking probabilities. */
    fun score(crops: List<ByteArray>, audio: FloatArray): FloatArray?
}

/**
 * Light-ASD (Liao et al., CVPR 2023) exported by scripts/export_models.py. The graph contains the
 * librosa-compatible MFCC front-end, so it takes raw audio and raw 0-255 pixels.
 */
class LightAsdModel(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : LightAsdScorer, AutoCloseable {

    override fun score(crops: List<ByteArray>, audio: FloatArray): FloatArray? {
        if (crops.size != FRAMES || audio.size < MIN_AUDIO_SAMPLES) return null

        val video = FloatArray(FRAMES * CROP * CROP)
        crops.forEachIndexed { f, crop ->
            val base = f * CROP * CROP
            for (i in 0 until CROP * CROP) video[base + i] = (crop[i].toInt() and 0xFF).toFloat()
        }

        val videoTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(video), longArrayOf(1, FRAMES.toLong(), CROP.toLong(), CROP.toLong()))
        val audioTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), longArrayOf(1, audio.size.toLong()))
        return videoTensor.use {
            audioTensor.use {
                session.run(mapOf("video" to videoTensor, "audio" to audioTensor)).use { result ->
                    (result[0].value as FloatArray).copyOf()
                }
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val FRAMES = 25
        const val CROP = 112
        private const val MIN_AUDIO_SAMPLES = 800 // 50 ms, same guard as LightASDDetector._run_inference

        fun fromFile(file: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): LightAsdModel =
            LightAsdModel(env, env.createSession(file.absolutePath, OrtSession.SessionOptions()))

        fun fromAssets(context: Context, assetPath: String): LightAsdModel =
            fromFile(AssetFiles.materialize(context, assetPath))
    }
}
