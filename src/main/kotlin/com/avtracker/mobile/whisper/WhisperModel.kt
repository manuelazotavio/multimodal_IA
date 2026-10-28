package com.avtracker.mobile.whisper

/** Independently packaged multilingual models, from fastest to most accurate in the local evaluation. */
enum class WhisperModel(val id: String, val assetDirectory: String) {
    BASE("base", "whisper"),
    SMALL("small", "whisper-small");

    fun decodingOptions(): WhisperOptions = when (this) {
        // Diarization already supplies short clips. Concatenating VAD islands can delete
        // real words in reverberant microphone audio; use VAD only to reject no-speech clips.
        BASE -> WhisperOptions(vadMode = WhisperVadMode.SPEECH_GATE, beamSize = 1,
            bestOf = 1, temperatures = listOf(0.0), usePrompt = false,
            rejectUnreliable = true, maxTokensPerAudioSecond = 12.0)
        // A generic initial prompt added decoder work without improving the local five-clip benchmark.
        // Keep Portuguese selected through the language token and retain beam 3 for decoding.
        SMALL -> WhisperOptions(beamSize = 3, bestOf = 1, temperatures = listOf(0.0),
            vadMode = WhisperVadMode.SPEECH_GATE, usePrompt = false,
            rejectUnreliable = true, maxTokensPerAudioSecond = 12.0)
    }
}
