"""Referencia do pipeline de transcricao: roda o faster-whisper ORIGINAL (o mesmo que o av-tracker usa) sobre alguns clipes
e grava o que ele produz, para o teste Kotlin (WhisperPipelineTest) exigir o mesmo.

  * VAD: faster_whisper.vad.get_speech_timestamps com os parametros do av-tracker (min_silence 500 ms, pad 300 ms);
  * transcricao: beam 5, prompt inicial, fallback de temperatura, limiares de compressao/logprob/no_speech.

O modelo e o whisper-base convertido para CTranslate2 em float32 (o app usa o mesmo whisper-base em ONNX; o av-tracker usa o
medium, que nao cabe no celular). O decodificador do app nao gera timestamps, entao a referencia roda com
`without_timestamps=True`; o restante e igual ao av-tracker.

Uso:  python scripts/parity/whisper_reference.py --ct2-model <pasta ct2 do whisper-base> --tts-dir src/test/resources
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 16000


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, audio):
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def silence(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def resample(x, factor):
    """Muda a velocidade (e o tom) por interpolacao linear: factor > 1 acelera."""
    n = int(len(x) / factor)
    return np.interp(np.arange(n) * factor, np.arange(len(x)), x).astype(np.float32)


def lowpass(x, k):
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same").astype(np.float32)


def make_clips(tts_dir):
    en = read_wav(tts_dir / "tts_en.wav")
    pt = read_wav(tts_dir / "tts_pt.wav")
    rng = np.random.RandomState(7)
    noise = lambda n, a: (rng.randn(n) * a).astype(np.float32)
    return [
        ("en_plain", "en", en),
        ("pt_plain", "pt", pt),
        ("en_padded", "en", np.concatenate([silence(2.0), en, silence(1.5)])),
        ("pt_two_utterances", "pt", np.concatenate([pt[: len(pt) // 2], silence(1.2), pt[len(pt) // 2:]])),
        ("en_noisy", "en", en + noise(len(en), 0.01)),
        ("pt_quiet", "pt", pt * 0.05),
        ("en_forced_pt", "pt", en),  # idioma errado: o logprob cai, exercita o fallback
        ("silence", "en", silence(3.0)),
        ("noise_only", "pt", noise(SR * 3, 0.02)),
        ("en_cut", "en", en[: int(len(en) * 0.6)]),
        ("pt_short", "pt", pt[int(len(pt) * 0.1): int(len(pt) * 0.1) + int(1.4 * SR)]),
        ("en_repeated", "en", np.concatenate([en, silence(0.6), en, silence(0.6), en])),
        # audio dificil: o beam search e o fallback so aparecem quando o modelo hesita
        ("en_noise_snr5", "en", en + noise(len(en), 0.06)),
        ("pt_noise_snr5", "pt", pt + noise(len(pt), 0.06)),
        ("en_fast", "en", resample(en, 1.5)),
        ("pt_slow", "pt", resample(pt, 0.7)),
        ("en_lowpass", "en", lowpass(en, 9)),
        ("pt_forced_en", "en", pt),
        ("pt_cut_mid", "pt", pt[: int(len(pt) * 0.45)]),
        ("en_noise_fast", "en", resample(en, 1.3) + noise(int(len(en) / 1.3), 0.04)),
        ("pt_noise_quiet", "pt", pt * 0.3 + noise(len(pt), 0.03)),
        ("en_tail_cut", "en", en[int(len(en) * 0.35):]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2-model", type=Path, required=True)
    ap.add_argument("--tts-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("src/test/resources/whisper_parity"))
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    args.out.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(str(args.ct2_model), device="cpu", compute_type="float32")
    vad_params = {"min_silence_duration_ms": 500, "speech_pad_ms": 300}

    results = []
    for name, lang, audio in make_clips(args.tts_dir):
        audio = audio.astype(np.float32)
        write_wav(args.out / f"{name}.wav", audio)
        # o app le o WAV de 16 bits: a referencia usa exatamente essas amostras
        audio = read_wav(args.out / f"{name}.wav")

        speech = get_speech_timestamps(audio, VadOptions(**vad_params))
        prompt = "Transcrição de conversa em português brasileiro." if lang == "pt" else "Transcript of a conversation."
        segs, info = model.transcribe(
            audio, language=lang, beam_size=5, vad_filter=True, vad_parameters=vad_params,
            condition_on_previous_text=False, without_timestamps=True, compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0, no_speech_threshold=0.6, initial_prompt=prompt,
        )
        segments = [{
            "text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "temperature": s.temperature, "compression_ratio": s.compression_ratio, "tokens": list(s.tokens),
        } for s in segs]
        results.append({
            "name": name, "language": lang, "prompt": prompt,
            "vad": [[int(c["start"]), int(c["end"])] for c in speech],
            "duration_after_vad": info.duration_after_vad,
            "segments": segments,
        })
        print(f"{name:20s} vad={len(speech)} chunks segs={[s['text'] for s in segments]}")

    (args.out / "expected.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
