package com.avtracker.mobile.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.avtracker.mobile.audio.AudioUtils
import com.avtracker.mobile.fusion.VoiceEncoder
import java.io.File
import java.nio.FloatBuffer

/**
 * speechbrain/spkrec-ecapa-voxceleb speaker embedder (192-d).
 *
 * The ONNX graph (scripts/export_models.py) takes the raw 16 kHz waveform [1, T] and includes the
 * Fbank front-end and per-utterance mean normalisation, so its output equals
 * EncoderClassifier.encode_batch: raw, NOT L2-normalised.
 */
class VoiceEmbedder(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : VoiceEncoder, AutoCloseable {
    private val inputName: String = session.inputInfo.keys.first()

    /** Raw embedding of [audio], or null if it is too short to produce any feature frame. */
    override fun embed(audio: FloatArray): FloatArray? {
        if (audio.size < MIN_SAMPLES) return null
        val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), longArrayOf(1, audio.size.toLong()))
        return input.use {
            session.run(mapOf(inputName to it)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<FloatArray>)[0].copyOf()
            }
        }
    }

    /** Port of RealtimeTranscriber._voice_embedding: unit-norm embedding, null if shorter than 0.4 s. */
    override fun embedNormalized(audio: FloatArray): FloatArray? {
        if (audio.size < AudioUtils.SAMPLE_RATE * 2 / 5) return null
        return embed(audio)?.let { AudioUtils.l2Normalize(it) }
    }

    override fun close() = session.close()

    companion object {
        const val DIMENSION = 192
        private const val MIN_SAMPLES = 1_600 // 0.1 s

        fun fromFile(file: File, env: OrtEnvironment = OrtEnvironment.getEnvironment()): VoiceEmbedder =
            VoiceEmbedder(env, env.createSession(file.absolutePath, OrtSession.SessionOptions()))

        /** Copies the asset to app storage first: avoids holding an 84 MB byte array next to ORT's own copy. */
        fun fromAssets(context: Context, assetPath: String): VoiceEmbedder =
            fromFile(AssetFiles.materialize(context, assetPath))
    }
}
