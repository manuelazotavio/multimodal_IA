"""Referencia da diarizacao: roda o pyannote/speaker-diarization-3.1 ORIGINAL e grava o que ele produz.
O teste Kotlin (PyannoteDiarizerTest) exige o mesmo.

  1. clustering: AgglomerativeClustering do pyannote (centroid + fcluster + tamanho minimo) sobre embeddings sinteticos;
  2. pipeline: audios (TTS misturados, ruido, silencio, recortes) -> turnos (inicio, fim, falante) e os artefatos internos
     (contagem de falantes, clusters de cada (janela, falante)).

Os parametros sao os do av-tracker: clustering.threshold = 0.6 (o instantiate() do av-tracker falha em segmentation.threshold
DEPOIS de aplicar o do clustering, e o except o engole: o efeito real e so esse), min_cluster_size 12 e min_duration_off 0.

Uso:  python scripts/parity/diarization_reference.py --tts-dir src/test/resources
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SR = 16000


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, audio):
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------------------------------------------
# 1. clustering
# ----------------------------------------------------------------------------------------------------------------

def clustering_cases(n_cases=120):
    from pyannote.audio.pipelines.clustering import AgglomerativeClustering
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    rng = np.random.RandomState(3)
    cases = []
    for c in range(n_cases):
        dim = 8
        n_chunks = int(rng.randint(1, 16))
        k = int(rng.randint(1, 5))
        centers = rng.randn(k, dim)
        noise = float(rng.choice([0.05, 0.2, 0.5, 1.0]))
        emb = np.zeros((n_chunks, 3, dim), dtype=np.float32)
        owner = rng.randint(0, k, size=(n_chunks, 3))
        for i in range(n_chunks):
            for s in range(3):
                emb[i, s] = (centers[owner[i, s]] + noise * rng.randn(dim)).astype(np.float32)
        active = rng.rand(n_chunks, 3) < 0.7
        if not active.any():
            active[0, 0] = True
        seg = np.zeros((n_chunks, 4, 3), dtype=np.float32)
        seg[:, 0, :] = active  # atividade so no primeiro quadro: o clustering so olha se ha algum quadro ativo
        # como o pipeline chama: (num, min, max) saem de set_num_speakers
        mode = int(rng.randint(0, 4))
        if mode == 0: num, mn, mx = None, 1, float("inf")
        elif mode == 1:
            f = int(rng.randint(1, 4)); num, mn, mx = f, f, f
        elif mode == 2: num, mn, mx = None, int(rng.randint(1, 3)), float("inf")
        else: num, mn, mx = None, 1, int(rng.randint(1, 4))

        clus = AgglomerativeClustering(metric="cosine")
        clus.instantiate({"method": "centroid", "min_cluster_size": 12, "threshold": 0.6})
        window = SlidingWindow(start=0.0, duration=10.0, step=1.0)
        hard, _, _ = clus(embeddings=emb.copy(), segmentations=SlidingWindowFeature(seg, window),
                          num_clusters=num, min_clusters=mn, max_clusters=mx)
        cases.append({
            "embeddings": [[[float(f"{float(v):.9g}") for v in e] for e in row] for row in emb],
            "active": active.tolist(),
            "num": num, "min": mn, "max": None if mx == float("inf") else mx,
            "hard": [[int(h) if active[i, s] else -2 for s, h in enumerate(row)] for i, row in enumerate(hard)],
        })
    return cases


# ----------------------------------------------------------------------------------------------------------------
# 2. pipeline
# ----------------------------------------------------------------------------------------------------------------

def audio_cases(tts_dir):
    en = read_wav(tts_dir / "tts_en.wav")
    pt = read_wav(tts_dir / "tts_pt.wav")
    rng = np.random.RandomState(5)
    z = lambda s: np.zeros(int(s * SR), dtype=np.float32)
    noise = lambda n, a: (rng.randn(n) * a).astype(np.float32)
    pit = lambda x, f: np.interp(np.arange(int(len(x) / f)) * f, np.arange(len(x)), x).astype(np.float32)
    return [
        ("en_pt_alternating", np.concatenate([z(0.5), en, z(1.0), pt, z(0.5)]), None),
        ("en_pt_one_speaker_forced", np.concatenate([en, z(0.5), pt]), 1),
        ("en_pt_two_speakers_forced", np.concatenate([en, z(0.5), pt]), 2),
        ("en_pitched", np.concatenate([en, z(0.8), pit(en, 1.25), z(0.8), en]), None),
        ("three_voices", np.concatenate([en, z(0.6), pit(en, 1.3), z(0.6), pt, z(0.6), pit(pt, 0.8)]), None),
        ("overlap", np.concatenate([en + 0.8 * np.pad(pt, (0, max(0, len(en) - len(pt))))[: len(en)], z(1.0), pt]), None),
        ("short_8s", np.concatenate([z(0.3), en[: int(5 * SR)], z(0.2)]), None),
        ("long_15s", np.concatenate([en, z(0.5), pt, z(0.5), en[: int(2 * SR)]]), None),
        ("noisy", np.concatenate([en, z(1.0), pt]) + noise(len(en) + len(pt) + SR, 0.02), None),
        ("silence", z(6.0), None),
        ("noise_only", noise(6 * SR, 0.03), None),
        ("single_short_utterance", np.concatenate([z(2.0), pt[: int(2 * SR)], z(3.0)]), None),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "src/test/resources/diarization")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = {"clustering": clustering_cases()}
    print("clustering:", len(result["clustering"]), "casos")

    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook  # noqa: F401  (so para garantir a importacao)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    try:  # exatamente o que o av-tracker faz
        pipeline.instantiate({"clustering": {"threshold": 0.6}, "segmentation": {"threshold": 0.4}})
    except Exception:
        pass
    print("parametros efetivos:", pipeline.parameters(instantiated=True))

    cases = []
    for name, audio, num_speakers in audio_cases(args.tts_dir):
        audio = audio.astype(np.float32)
        write_wav(args.out / f"{name}.wav", audio)
        audio = read_wav(args.out / f"{name}.wav")  # o app le o WAV de 16 bits

        params = {}
        if num_speakers:
            params = {"min_speakers": num_speakers, "max_speakers": num_speakers}
        artefacts = {}

        def hook(step, artefact, file=None, total=None, completed=None):
            if step in ("speaker_counting", "discrete_diarization") and artefact is not None and hasattr(artefact, "data"):
                artefacts[step] = artefact.data.tolist()

        diarization = pipeline({"waveform": torch.from_numpy(audio)[None], "sample_rate": SR}, hook=hook, **params)
        turns = [[float(t.start), float(t.end), str(spk)] for t, _, spk in diarization.itertracks(yield_label=True)]
        cases.append({"name": name, "num_speakers": num_speakers, "turns": turns,
                      "count": artefacts.get("speaker_counting"), "discrete": artefacts.get("discrete_diarization")})
        print(f"{name:28s} {len(turns)} turnos, falantes={sorted(set(t[2] for t in turns))}")
    result["pipeline"] = cases

    (args.out / "expected.json").write_text(json.dumps(result), encoding="utf-8")
    print("gravado:", args.out / "expected.json")


if __name__ == "__main__":
    main()
