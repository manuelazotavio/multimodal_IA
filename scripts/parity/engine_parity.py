"""Teste diferencial do motor de fusao: roda o RealtimeTranscriber ORIGINAL do av-tracker (sem carregar Whisper,
pyannote nem ECAPA) com stubs deterministicos, e grava o estado esperado apos cada passo. O teste Kotlin
(PythonParityTest) reproduz os mesmos cenarios e exige o mesmo estado.

Stubs (a unica coisa trocada): o classificador ECAPA (a "voz" e um codigo = tamanho da primeira sequencia de
amostras de mesmo sinal, invariante a normalizacao de volume), a deteccao de genero por pitch, o Whisper, a
diarizacao, o SepFormer, a ASD e o face tracker. Tudo o mais e o codigo real: MultiSpeakerVerifier, decisoes,
nomes, embeddings salvos em disco, FN/FP, metricas.

Uso:  python scripts/parity/engine_parity.py --av-tracker C:/Users/manu/av-tracker [--ner]
"""
import argparse
import datetime as real_datetime_module
import json
import os
import random
import re
import sys
import tempfile
import threading
import time as real_time
import types
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

SR = 16000
DIM = 16


# ----------------------------------------------------------------------------------------------------------------
# geracao de cenarios
# ----------------------------------------------------------------------------------------------------------------

MALE = ["João", "Pedro", "Arthur", "Heitor", "Gustavo", "Marcos"]
FEMALE = ["Maria", "Manuela", "Isabel", "Luísa", "Carla", "Ana"]
NEUTRAL = ["Kauan", "Sasha", "Robin"]

TEXTS_PLAIN = [
    "I think we should start the meeting now.", "That sounds like a reasonable plan to me.",
    "We need to finish the report before Friday.", "Let me share my screen for a second.",
    "Eu acho que a gente deveria começar agora.", "Podemos revisar o orçamento depois do almoço.",
    "The numbers look better than last quarter.", "Não tenho certeza se isso vai funcionar.",
]
TEXTS_HALLUC = ["Thank you. Thank you. Thank you.", "Subtitles by the Amara.org community", "Música.", "ok",
                "Transcript of a conversation.", "Não se esqueça de se inscrever no canal"]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def make_scenario(seed, with_ner):
    rng = random.Random(seed)
    nrng = np.random.RandomState(seed)
    people = rng.sample(MALE + FEMALE + NEUTRAL, rng.randint(3, 5))
    directions = {p: unit(nrng.randn(DIM)) for p in people}

    seed_files, voices = [], {}
    code = 40
    for p in people:
        for k in range(rng.randint(1, 3)):
            stamp = f"2026030{k + 1}_10{k}0{k}0"
            suffix = rng.choice(["", "_auto", "_fvbind"])
            vec = (directions[p] + 0.05 * nrng.randn(DIM)) * rng.choice([1.0, 30.0])
            seed_files.append({"name": f"{p}_{stamp}{suffix}", "vec": vec.tolist()})
    # arquivos genericos que existem so para o ciclo de renomeacao
    seed_files.append({"name": "Person_1_20260301_090000_auto", "vec": (nrng.randn(DIM)).tolist()})

    def gender_for(name):
        if name in MALE: g = "male"
        elif name in FEMALE: g = "female"
        else: g = None
        r = rng.random()
        if g and r < 0.12: g = "female" if g == "male" else "male"   # pitch contradiz o nome
        elif r > 0.93: g = None
        return g

    for p in people:      # vozes que soam claramente como a pessoa
        voices[str(code)] = {"vec": (directions[p] + 0.15 * nrng.randn(DIM)).tolist(), "gender": gender_for(p), "owner": p}
        code += 10
    for p in people[:2]:  # vozes "meio parecidas": faixa 0.6-0.8 de similaridade
        mix = unit(0.75 * directions[p] + 0.66 * unit(nrng.randn(DIM)))
        voices[str(code)] = {"vec": mix.tolist(), "gender": gender_for(p), "owner": None}
        code += 10
    for _ in range(3):    # desconhecidos
        voices[str(code)] = {"vec": unit(nrng.randn(DIM)).tolist(), "gender": rng.choice(["male", "female", None]), "owner": None}
        code += 10

    codes = list(voices)
    steps, t = [], 100.0
    intro_pool = ["My name is {n} and I am new here.", "Eu sou o {n}, prazer.", "Meu nome é {n}.", "Hi, I'm {n}.",
                  "Aqui é a {n}."]
    vocative_pool = ["{n}, what do you think about this?", "Pode falar, {n}.", "Thanks, {n}. That was helpful.",
                     "Fala, {n}.", "Vamos ouvir a parte do {n} agora."]

    def new_faces_step():
        tracks = rng.sample(range(1, 9), rng.randint(0, 3))
        active, names, set_names, bind, known_faces = {}, {}, {}, {}, []
        for i, tr in enumerate(tracks):
            pid = f"spk_9{seed % 10}{tr}"
            if rng.random() < 0.55:
                nm = rng.choice(people)
                names[str(tr)] = nm; set_names[pid] = nm; bind[nm] = pid; known_faces.append(nm)
            else:
                nm = f"Person_{tr}"
                names[str(tr)] = nm; set_names[pid] = nm
            active[str(tr)] = pid
        return {"type": "faces", "active_faces": active, "face_names": names, "set_names": set_names, "bind": bind,
                "known_faces": known_faces}

    steps.append(new_faces_step())
    n_steps = rng.randint(28, 40)
    for _ in range(n_steps):
        t += rng.uniform(2.0, 9.0)
        r = rng.random()
        if r < 0.18:
            steps.append(new_faces_step())
            continue
        active_tracks = [int(k) for k in steps_last_faces(steps)]
        pick = lambda: (rng.choice(active_tracks) if active_tracks and rng.random() < 0.6 else None)
        if r < 0.30:  # passo de chunk: 1-3 falas de vozes diferentes/iguais
            pieces = []
            for _ in range(rng.randint(1, 3)):
                pieces.append({"voice": rng.choice(codes), "seconds": round(rng.uniform(1.0, 4.0), 2), "slot": rng.randint(0, 2)})
            steps.append({"type": "chunk", "t": t, "pieces": pieces, "zone": rng.choice([0, 0, 2 * SR]),
                          "texts": [rng.choice(TEXTS_PLAIN + TEXTS_HALLUC[:2]) for _ in pieces],
                          "asd": pick(), "guess": pick()})
            continue
        voice = rng.choice(codes)
        rr = rng.random()
        person = rng.choice(people)
        if rr < 0.10: text = rng.choice(TEXTS_HALLUC)
        elif rr < 0.30: text = rng.choice(intro_pool).format(n=rng.choice(people + ["Laura", "David", "Joe"]))
        elif rr < 0.55: text = rng.choice(vocative_pool).format(n=person)
        else: text = rng.choice(TEXTS_PLAIN)
        steps.append({"type": "segment", "t": t, "voice": voice, "seconds": round(rng.uniform(2.0, 6.0), 2), "text": text,
                      "no_speech": rng.choice([0.01, 0.02, 0.05, 0.2, 0.7]), "logprob": round(rng.uniform(-1.6, -0.1), 2),
                      "asd": pick(), "guess": pick()})

    return {"seed": seed, "ner": with_ner,
            "config": {"language": rng.choice(["en", "pt"]), "num_speakers": rng.choice([None, None, 2, 3, 4])},
            "seed_files": seed_files, "voices": voices, "steps": steps}


def steps_last_faces(steps):
    for s in reversed(steps):
        if s["type"] == "faces":
            return list(s["active_faces"].keys())
    return []


# ----------------------------------------------------------------------------------------------------------------
# execucao com o codigo Python original
# ----------------------------------------------------------------------------------------------------------------

class FakeClock:
    epoch = 1_800_000_000.0


class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)

    def join(self, timeout=None):
        pass


def voice_code(audio):
    """Identidade da voz = comprimento mais comum das sequencias de mesmo sinal (onda quadrada de periodo 2*code).

    Sobrevive a cortes do audio e a normalizacao de volume; as sequencias das pontas (parciais) sao ignoradas.
    """
    a = np.asarray(audio).flatten()
    runs, n = [], 1
    for i in range(1, len(a)):
        if (a[i] < 0) == (a[i - 1] < 0):
            n += 1
        else:
            runs.append(n)
            n = 1
    runs.append(n)
    interior = runs[1:-1] or runs
    counts = {}
    for r in interior:
        counts[r] = counts.get(r, 0) + 1
    best = max(counts.values())
    return min(r for r, c in counts.items() if c == best)


def make_audio(code, seconds):
    """Onda quadrada de amplitude 0.2 cujas sequencias de mesmo sinal tem `code` amostras."""
    n = int(seconds * SR)
    idx = np.arange(n)
    return np.where((idx // code) % 2 == 0, 0.2, -0.2).astype(np.float32)


def run_python(scenario, av_tracker, ner_model):
    os.chdir(av_tracker)
    sys.path.insert(0, str(av_tracker))
    import logging
    logging.disable(logging.CRITICAL)

    # O codigo original sincroniza cada voz salva com o SQLite do projeto (`from src.database import get_db`).
    # Uma versao anterior deste script nao bloqueava isso e gravou dados falsos em av_tracker.db. Com None em
    # sys.modules o import levanta ImportError, que o original ja captura ("DB sync failed") e ignora.
    sys.modules["src.database"] = None

    real_dt = real_datetime_module.datetime

    class FakeDT(real_dt):
        @classmethod
        def now(cls, tz=None):
            return real_dt.fromtimestamp(FakeClock.epoch)

    real_datetime_module.datetime = FakeDT
    real_time.time = lambda: FakeClock.epoch
    threading.Thread = SyncThread

    rt = __import__("src.realtime_transcriber", fromlist=["x"])
    msv = __import__("src.multi_speaker_verifier", fromlist=["x"])
    rt.datetime = FakeDT
    rt.torch = torch

    voices = scenario["voices"]

    def embedding_of(audio):
        """Embedding da voz + ruido deterministico por segmento (como ECAPA real, que nunca repete o vetor exato).

        Sem o ruido, vozes iguais dao vetores identicos e os empates entre sessoes dependem da ordem de soma do
        BLAS, o que nao existe com embeddings reais. O ruido e uma LCG identica no teste Kotlin.
        """
        code = voice_code(audio)
        state = (len(audio) * 7919 + code) & 0xFFFFFFFF
        noise = []
        for _ in range(DIM):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            noise.append(((state >> 8) / float(1 << 24) - 0.5) * 2.0)
        return (np.array(voices[str(code)]["vec"], dtype=np.float64) + 0.02 * np.array(noise)).astype(np.float32)

    class FakeClassifier:
        def encode_batch(self, signal):
            return torch.from_numpy(embedding_of(signal.detach().cpu().numpy()[0])).view(1, 1, -1)

    tmp = tempfile.mkdtemp(prefix="parity_emb_")
    for f in scenario["seed_files"]:
        np.save(os.path.join(tmp, f["name"] + ".npy"), np.array(f["vec"], dtype=np.float32))

    verifier = msv.MultiSpeakerVerifier.__new__(msv.MultiSpeakerVerifier)
    verifier.device = "cpu"; verifier.threshold = 0.80; verifier.embedding_directory = tmp
    verifier.classifier = FakeClassifier(); verifier.embeddings = {}; verifier._raw_embeddings = {}; verifier._auto_enroll_last = {}
    verifier.load_embeddings(tmp)

    counter = [0]

    def make_pid():
        counter[0] += 1
        return f"spk_{counter[0]:03d}"

    shared = {"active_faces": {}, "person_names": {}, "embedding_to_person": {}, "person_id_factory": make_pid}
    for name in verifier.embeddings.keys():
        pid = make_pid(); shared["embedding_to_person"][name] = pid; shared["person_names"][pid] = name

    class Track:
        def __init__(self):
            self._track_to_name, self.known_embeddings, self.renames = {}, {}, []

        def rename_person(self, old, new, only_track_id=None):
            self.renames.append([old, new, only_track_id])

    face = Track(); shared["face_tracker"] = face

    class Asd:
        active = None; guess = None
        def get_active_speaker(self, a, b): return Asd.active
        def get_best_guess(self, a, b): return Asd.guess

    shared["asd"] = Asd()

    t = rt.RealtimeTranscriber.__new__(rt.RealtimeTranscriber)
    cfg = scenario["config"]
    t.verifier = verifier; t.device = "cpu"; t.language = cfg["language"]; t.emb_dir = tmp; t.use_ai_analysis = False
    t.diarization_clustering_threshold = 0.6; t.verifier_confidence_min = 0.70; t.chunk_duration = 2.0; t.sample_rate = 16000
    t.shared_state = shared; t.num_speakers = cfg["num_speakers"]; t.audio_device = None; t.running = True
    t.processing_buffer = []; t.speaker_history = {}; t.full_transcript = []; t.all_audio_chunks = []
    t.unknown_speakers_audio = defaultdict(list)
    t.speaker_names = shared["person_names"]; t._emb_to_pid = shared["embedding_to_person"]
    t.identified_speakers = set(); t._voice_emb_saved = set(); t._fv_bind_last = {}; t._session_to_face = {}
    t._session_gender = {}; t._session_unknown_embs = {}; t._session_unknown_counter = 0
    t.classifier = FakeClassifier(); t._perf = defaultdict(list); t._perf_chunk_count = 0; t._perf_summary_interval = 10 ** 9
    t._fn_events = []; t._pending_addressee = None; t._context_names = set()
    t._addressee_votes = defaultdict(lambda: defaultdict(int)); t._llm_analysis_interval = 10; t._llm_last_analysis = 0
    t._llm_client = None; t._segment_metrics = []; t._session_start = FakeDT.now(); t._whisper_size = "medium"
    t.llm = None
    t.nlp = None
    if ner_model:
        import spacy
        t.nlp = spacy.load(ner_model)

    pending_texts = deque(); current = {"ns": 0.01, "lp": -0.2}

    class Seg:
        def __init__(self, text, ns, lp): self.text, self.no_speech_prob, self.avg_logprob = text, ns, lp

    class Whisper:
        def transcribe(self, audio, **kw):
            text = pending_texts.popleft() if pending_texts else ""
            return iter([Seg(text, current["ns"], current["lp"])]), None

    t.whisper = Whisper()
    t._detect_gender_from_audio = lambda audio: voices[str(voice_code(audio))]["gender"]
    t._separate_with_sepformer = lambda audio: [np.zeros(10, dtype=np.float32)]
    t._match_streams_to_speakers = lambda streams, mapping: {}

    class Turn:
        def __init__(self, s, e): self.start, self.end = s, e

    class Diar:
        def __init__(self, turns): self.turns = turns
        def itertracks(self, yield_label=True): return [(Turn(s, e), None, lab) for s, e, lab in self.turns]

    def install_pipeline(turns):
        t.pipeline = lambda inp, **kw: Diar(turns)

    def snapshot(new_entries, new_metrics):
        return {
            "appended": [{"speaker": e["speaker"], "name": e["verified_name"], "text": e["text"]} for e in new_entries],
            "decisions": [m["decision"] for m in new_metrics],
            "fn": [m["fn_reasons"] for m in new_metrics], "fp": [[m["fp_risk"], m["fp_reasons"]] for m in new_metrics],
            "names": dict(sorted(t.speaker_names.items())), "emb_to_pid": dict(sorted(t._emb_to_pid.items())),
            "identified": sorted(t.identified_speakers), "session_to_face": dict(sorted(t._session_to_face.items())),
            "context": sorted(t._context_names), "verifier": sorted(verifier.embeddings.keys()),
            "files": sorted(f[:-4] for f in os.listdir(tmp) if f.endswith(".npy")), "renames": [list(r) for r in face.renames],
            "session_ids": sorted(t._session_unknown_embs.keys()),
        }

    results = []
    for step in scenario["steps"]:
        n_entries, n_metrics = len(t.full_transcript), len(t._segment_metrics)
        if step["type"] == "faces":
            shared["active_faces"] = {int(k): v for k, v in step["active_faces"].items()}
            for pid, nm in step["set_names"].items(): t.speaker_names[pid] = nm
            for nm, pid in step["bind"].items(): t._emb_to_pid[nm] = pid
            face._track_to_name = {int(k): v for k, v in step["face_names"].items()}
            face.known_embeddings = {n: 1 for n in step["known_faces"]}
        else:
            FakeClock.epoch = 1_800_000_000.0 + step["t"]
            Asd.active, Asd.guess = step.get("asd"), step.get("guess")
            if step["type"] == "segment":
                current.update(ns=step["no_speech"], lp=step["logprob"])
                pending_texts.append(step["text"])
                audio = make_audio(int(step["voice"]), step["seconds"])
                g = t._detect_gender_from_audio(audio)
                sid = t._get_or_create_session_speaker_id(audio, audio_gender=g)
                fpid = t._session_to_face.get(sid)
                if fpid and fpid in t.speaker_names: sid = fpid
                t._process_segment(audio, session_hint=sid, wall_time_start=step["t"], wall_time_end=step["t"] + step["seconds"])
            else:  # chunk
                current.update(ns=0.01, lp=-0.2)
                for tx in step["texts"]: pending_texts.append(tx)
                pieces, parts, turns, pos = step["pieces"], [], [], 0.0
                for p in pieces:
                    parts.append(make_audio(int(p["voice"]), p["seconds"]))
                    turns.append((pos, pos + p["seconds"], f"SPEAKER_{p['slot']:02d}"))
                    pos += p["seconds"]
                audio = np.concatenate(parts)
                install_pipeline(turns)
                t._process_chunk(audio, chunk_wall_start=step["t"], new_zone_start_samples=int(step["zone"]))
            pending_texts.clear()
        results.append(snapshot(t.full_transcript[n_entries:], t._segment_metrics[n_metrics:]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-tracker", type=Path, required=True)
    ap.add_argument("--ner", action="store_true", help="usa o spaCy real (en_core_web_sm / pt_core_news_sm) no lado Python")
    ap.add_argument("--scenarios", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "src/test/resources/parity/engine_scenarios.json")
    a = ap.parse_args()

    out = []
    for seed in range(a.scenarios):
        sc = make_scenario(seed, a.ner)
        ner_model = None
        if a.ner:
            ner_model = "pt_core_news_sm" if sc["config"]["language"] == "pt" else "en_core_web_sm"
        sc["expected"] = run_python(sc, a.av_tracker, ner_model)
        out.append(sc)
        print(f"cenario {seed}: {len(sc['steps'])} passos, decisoes = {sum(len(e['decisions']) for e in sc['expected'])}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("gravado:", a.out)


if __name__ == "__main__":
    main()
