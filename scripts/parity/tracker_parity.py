"""Teste diferencial do rastreador facial: roda o PersonIDTracker e o ByteTrack ORIGINAIS do av-tracker (boxmot) com
embeddings deterministicos e grava o estado esperado apos cada frame. O teste Kotlin (TrackerParityTest) reproduz os
mesmos cenarios e exige o mesmo estado.

O que e trocado: so a rede EdgeFace (o `_extract_face_embedding` devolve o embedding do cenario para o rosto cujo
"rotulo" foi pintado no centro do recorte) e o relogio (`time.time`/`datetime.now` seguem o tempo do cenario). Todo o
resto e o codigo real: ByteTrack, votos de confirmacao, reservas de nome, persistencia, auto-cadastro, fusao,
limite de identidades, renomeacao, consolidacao e os arquivos .npy em disco.

Seguranca: o modulo src.database e bloqueado (o PersonIDTracker tenta espelhar cadastros no SQLite do av-tracker;
isso nunca pode tocar o banco real).

Uso:  python scripts/parity/tracker_parity.py --av-tracker C:/Users/manu/av-tracker --scenarios 10 --out <json>
"""
import argparse
import datetime as real_datetime
import json
import os
import random
import sys
import tempfile
import time as real_time
import types
from pathlib import Path

import numpy as np

W, H = 640, 480
D = 16
DT = 1.0 / 15


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def q32(v):
    """9 digitos significativos identificam um float32: o JSON leva o mesmo valor para Python e Kotlin."""
    return [float(f"{float(x):.9g}") for x in np.asarray(v, dtype=np.float64)]


def f32(v):
    return np.asarray(v, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------------
# geracao de cenarios
# ----------------------------------------------------------------------------------------------------------------

def make_scenario(seed):
    rng = random.Random(seed)
    nrng = np.random.RandomState(seed)

    sigma = rng.choice([0.03, 0.05, 0.08, 0.12])
    n_known = rng.choice([0, 1, 1, 2, 3])  # 0: ninguem cadastrado, todos entram por auto-cadastro
    n_people = n_known + rng.randint(1, 4)
    bases = [unit(nrng.randn(D)) for _ in range(n_people)]
    if n_known > 0 and n_people > n_known and rng.random() < 0.5:  # um desconhecido parecido com um conhecido
        bases[n_known] = unit(bases[rng.randrange(n_known)] * 0.75 + unit(nrng.randn(D)) * 0.66)

    known_names = ["Alice", "Bob", "Mary_Ann"][:n_known]
    # quanto o rosto ao vivo se afasta do cadastrado (iluminacao, idade): 0 = igual, 0.9 = quase no limiar de 0.65
    drift = [rng.choice([0.0, 0.0, 0.5, 0.9]) for _ in range(n_known)]
    files = []
    for k, name in enumerate(known_names):
        for j in range(rng.randint(1, 3)):
            files.append({"file": f"{name}_2026030{j + 1}_1{k}00{j}0_auto.npy",
                          "vec": q32(unit(bases[k] + 0.05 * nrng.randn(D)))})
    many_files = n_known > 0 and rng.random() < 0.3  # passa de 8 arquivos: a renomeacao poda os mais antigos
    if many_files:
        for j in range(8):
            files.append({"file": f"Alice_202602{10 + j:02d}_08{j}000_auto.npy", "vec": q32(unit(bases[0] + 0.05 * nrng.randn(D)))})
    if n_known > 0 and rng.random() < 0.5:
        files.append({"file": "Alice.npy", "vec": q32(unit(bases[0] + 0.05 * nrng.randn(D)))})
    if n_known > 0 and rng.random() < 0.7:  # Person_N parecido com um conhecido: deve ser consolidado
        files.append({"file": "Person_1_auto.npy", "vec": q32(unit(bases[0] + 0.1 * nrng.randn(D)))})
    if rng.random() < 0.5:
        files.append({"file": "Person_2_auto.npy", "vec": q32(unit(nrng.randn(D)))})
    if rng.random() < 0.3:
        files.append({"file": "Unknown_1_auto.npy", "vec": q32(unit(nrng.randn(D)))})
    if rng.random() < 0.3:  # nome novo parecido com um Person_N do disco
        files.append({"file": "Zelia_20260301_090000_auto.npy", "vec": q32(unit(nrng.randn(D)))})

    long_absence = rng.random() < 0.3
    n_frames = rng.randint(260, 330) if long_absence else rng.randint(200, 260)

    people = []
    for i in range(n_people):
        start = rng.randint(0, 50) if (i < 2 or i >= n_known) else rng.randint(20, n_frames // 2)
        people.append({
            "start": start, "end": n_frames - rng.randint(0, 30),
            "cx": rng.uniform(120, W - 120), "cy": rng.uniform(110, H - 110),
            "vx": rng.uniform(-1.6, 1.6), "vy": rng.uniform(-0.8, 0.8),
            "ax": rng.uniform(0, 14), "ay": rng.uniform(0, 8), "period": rng.uniform(40, 120),
            "w": rng.uniform(55, 130), "aspect": rng.uniform(0.75, 1.0),
            "away": [],  # (inicio, fim) sem deteccao
            "turned": rng.random() < 0.5,  # as vezes vira o rosto (embedding ruim)
            "phase": 0,
        })
        for _ in range(rng.randint(1, 3)):
            a = rng.randint(30, n_frames - 30)
            people[i]["away"].append((a, a + rng.randint(3, 40)))
        if long_absence and i == 0:
            a = rng.randint(60, 100)
            people[i]["away"].append((a, a + rng.randint(150, 190)))

    # duas levas: os primeiros saem e outros chegam depois (o limite de identidades forca a fusao com quem saiu)
    if n_people >= 4 and rng.random() < 0.5:
        cut = n_frames // 3 + rng.randint(-10, 10)
        for i in range(2):
            people[i]["end"] = cut
            people[i]["away"] = [(x, y) for x, y in people[i]["away"] if y < cut]
        for i in range(2, n_people):
            people[i]["start"] = max(people[i]["start"], cut + 40 + rng.randint(0, 20))

    # passagem de bastao: b assume exatamente a trilha de a (mesma posicao) quando a sai; o ByteTrack mantem o id
    if n_people >= 2 and rng.random() < 0.7:
        a, b = rng.sample(range(n_people), 2)
        split = rng.randint(70, max(71, n_frames - 70))
        people[a]["start"] = min(people[a]["start"], split - 30)
        people[a]["end"] = split
        people[a]["away"] = [(x, y) for x, y in people[a]["away"] if y < split]
        for k in ("cx", "cy", "vx", "vy", "ax", "ay", "period", "w", "aspect"):
            people[b][k] = people[a][k]
        people[b]["phase"] = split - people[a]["start"]
        people[b]["start"] = split
        people[b]["end"] = n_frames
        people[b]["away"] = [(x, y) for x, y in people[b]["away"] if x > split + 10]

    live_bases = [unit(b + (drift[i] * 0.4 * unit(nrng.randn(D)) if i < n_known else 0)) for i, b in enumerate(bases)]

    frames = []
    t = 1000.0
    for f in range(n_frames):
        t += DT
        if rng.random() < 0.012:  # pausa: exercita as janelas de 2 s, 5 s e 30 s
            t += rng.choice([1.5, 2.5, 6.0, 12.0, 33.0])
        dets = []
        labels = []
        embs = {}
        for i, p in enumerate(people):
            if not (p["start"] <= f < p["end"]):
                continue
            if any(a <= f < b for a, b in p["away"]):
                continue
            ph = f - p["start"] + p["phase"]
            cx = p["cx"] + p["vx"] * ph + p["ax"] * np.sin(2 * np.pi * ph / p["period"])
            cy = p["cy"] + p["vy"] * ph + p["ay"] * np.cos(2 * np.pi * ph / p["period"])
            w = p["w"] * (1 + 0.08 * np.sin(ph / 17.0))
            h = w / p["aspect"]
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, x2 = (x1 + rng.gauss(0, 1.2), x2 + rng.gauss(0, 1.2))
            y1, y2 = (y1 + rng.gauss(0, 1.2), y2 + rng.gauss(0, 1.2))
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
            if x2 - x1 < 14 or y2 - y1 < 14:
                continue
            conf = rng.uniform(0.62, 0.96)
            if rng.random() < 0.12:
                conf = rng.uniform(0.3, 0.44)  # segunda associacao do ByteTrack
            elif rng.random() < 0.02:
                conf = rng.uniform(0.44, 0.47)
            dets.append([x1, y1, x2, y2, conf, 0.0])
            labels.append(i + 1)
            s = 0.9 if (p["turned"] and rng.random() < 0.05) else sigma
            if rng.random() < 0.05:  # rosto ambiguo: mistura com outra pessoa
                wgt = rng.uniform(0.5, 0.85)
                embs[i + 1] = unit(wgt * live_bases[i] + (1 - wgt) * bases[rng.randrange(n_people)] + s * nrng.randn(D))
            else:
                embs[i + 1] = unit(live_bases[i] + s * nrng.randn(D))
        if rng.random() < 0.05:  # falso positivo (rotulo 0: fundo)
            x1 = rng.uniform(0, W - 60)
            y1 = rng.uniform(0, H - 60)
            dets.append([x1, y1, x1 + rng.uniform(20, 60), y1 + rng.uniform(20, 60), rng.uniform(0.3, 0.9), 0.0])
            labels.append(0)
        embs[0] = unit(nrng.randn(D))
        order = sorted(range(len(dets)), key=lambda k: -dets[k][4])  # YOLO devolve por confianca decrescente
        dets = [dets[k] for k in order]
        labels = [labels[k] for k in order]
        frames.append({
            "t": t,
            "dets": [[float(f"{v:.6f}") for v in d] for d in dets],
            "labels": labels,
            "embs": {str(k): q32(v) for k, v in embs.items()},
        })

    # renomeacoes pedidas de fora (a voz descobrindo o nome), com pelo menos 3 s entre elas
    spare = ["Zoé", "Yuri", "Xavi", "Wanda", "Vera", "Ulisses"]
    if many_files:
        spare.append("Alice")  # a voz confirma um nome que ja tem muitos arquivos
    rng.shuffle(spare)
    actions = []
    last_t = -100.0
    for f in range(45, n_frames, 5):
        if spare and rng.random() < 0.22 and frames[f]["t"] - last_t > 3.0:
            actions.append({"frame": f, "op": rng.choice(["generic_only_track", "generic_all", "real_only_track"]),
                            "new": spare.pop()})
            last_t = frames[f]["t"]

    return {
        "seed": seed, "max_identities": rng.choice([None, 2, 3, 3, 4]),
        "files": files, "frames": frames, "actions": actions,
    }


# ----------------------------------------------------------------------------------------------------------------
# execucao com o codigo original
# ----------------------------------------------------------------------------------------------------------------

class Clock:
    now = 0.0


class FakeTimeModule:
    @staticmethod
    def time():
        return Clock.now

    @staticmethod
    def perf_counter():
        return real_time.perf_counter()


class FakeDatetime:
    @staticmethod
    def now():
        return real_datetime.datetime.fromtimestamp(Clock.now, tz=real_datetime.timezone.utc).replace(tzinfo=None)


class RecorderDB:
    """Substitui src.database: grava as chamadas de espelhamento em vez de abrir o SQLite (o av_tracker.db real nunca e tocado)."""

    def __init__(self):
        self.calls = []
        self.ids = {}
        self.names = {}

    def add_speaker(self, name, gender=None):
        self.calls.append(["add_speaker", name])
        if name not in self.ids:
            self.ids[name] = len(self.ids) + 1
            self.names[self.ids[name]] = name
        return self.ids[name]

    def _save(self, kind, speaker_id, embedding, source_file):
        emb = np.asarray(embedding, dtype=np.float32)
        self.calls.append([kind, self.names[speaker_id], source_file, round(float(np.sum(emb)), 3), int(emb.size)])
        return len(self.calls)

    def save_voice_embedding(self, speaker_id, embedding, source_file=None):
        return self._save("voice", speaker_id, embedding, source_file)

    def save_face_embedding(self, speaker_id, embedding, source_file=None):
        return self._save("face", speaker_id, embedding, source_file)


def install_recorder_db():
    """Instala o stub como `src.database` e devolve o gravador."""
    import types as _types

    recorder = RecorderDB()
    module = _types.ModuleType("src.database")
    module.get_db = lambda db_path=None: recorder
    sys.modules["src.database"] = module
    return recorder


def load_original(av_tracker):
    sys.path.insert(0, str(av_tracker))
    recorder = install_recorder_db()  # src.database vira um gravador: NUNCA toca o banco real do av-tracker
    import src.personid_tracker as pt

    class DummyNet:
        def load_state_dict(self, *_a, **_k):
            return None

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

    pt._EdgeFaceXXS = DummyNet
    pt.torch.load = lambda *a, **k: {}
    pt.time = FakeTimeModule
    return pt, recorder


def event_view(e):
    out = {}
    for k, v in e.items():
        if k == "ts":
            continue
        out[k] = int(v) if k == "track_id" and v is not None else v
    return out


def run_original(pt, recorder, sc, workdir):
    from boxmot.trackers.bytetrack.basetrack import BaseTrack

    recorder.calls.clear()
    recorder.ids.clear()
    recorder.names.clear()

    emb_dir = Path(workdir) / f"faces_{sc['seed']}"
    emb_dir.mkdir()
    for f in sc["files"]:
        np.save(emb_dir / f["file"], f32(f["vec"]))

    BaseTrack.clear_count()
    tracker = pt.PersonIDTracker(device="cpu", max_identities=sc["max_identities"])
    current = {"emb": None}

    def fake_extract(self, face_img):
        label = int(face_img[face_img.shape[0] // 2, face_img.shape[1] // 2])
        return f32(current["emb"][str(label)])

    pt.PersonIDTracker._extract_face_embedding = fake_extract

    Clock.now = sc["frames"][0]["t"]
    tracker.load_known_embeddings(str(emb_dir))
    loaded = {n: [float(x) for x in v] for n, v in tracker.known_embeddings.items()}
    loaded_files = sorted(os.listdir(emb_dir))

    actions_by_frame = {}
    for a in sc["actions"]:
        actions_by_frame.setdefault(a["frame"], []).append(a)

    frames_out = []
    last_results = []
    for f, fr in enumerate(sc["frames"]):
        Clock.now = fr["t"]
        current["emb"] = fr["embs"]
        frame = np.zeros((H, W), dtype=np.int16)
        for d, label in zip(fr["dets"], fr["labels"]):
            frame[int(d[1]):int(d[3]), int(d[0]):int(d[2])] = label
        results = tracker.update(frame, fr["dets"])
        last_results = results

        done = []
        for a in actions_by_frame.get(f, []):
            target = None
            for r in results:
                is_generic = pt.re.match(r"^Person_\d+$", r["name"]) is not None
                is_real = r["name"] != "Unknown" and not is_generic
                if (a["op"].startswith("generic") and is_generic) or (a["op"] == "real_only_track" and is_real):
                    target = r
                    break
            if target is None:
                done.append({"op": a["op"], "skipped": True})
                continue
            only = int(target["track_id"]) if a["op"].endswith("only_track") else None
            fake = types.ModuleType("datetime")
            fake.datetime = FakeDatetime
            real = sys.modules["datetime"]
            sys.modules["datetime"] = fake
            try:
                tracker.rename_person(target["name"], a["new"], only_track_id=only)
            finally:
                sys.modules["datetime"] = real
            done.append({"op": a["op"], "old": target["name"], "new": a["new"], "only": only, "skipped": False})

        frames_out.append({
            "results": [[int(r["track_id"]), r["name"], float(r["confidence"])] + [float(v) for v in r["bbox"]]
                        for r in results],
            "track_to_name": [[int(k), v] for k, v in tracker._track_to_name.items()],
            "actions": done,
        })

    metrics = tracker.get_session_face_metrics()
    final_files = {}
    for name in sorted(os.listdir(emb_dir)):
        if name.endswith(".npy"):
            final_files[name] = [float(x) for x in np.load(emb_dir / name)]

    # ByteTrack sozinho, alimentado com as mesmas deteccoes (o PersonIDTracker nao o chama sem deteccoes)
    BaseTrack.clear_count()
    from boxmot import ByteTrack
    bt = ByteTrack(track_buffer=150)
    bt_out = []
    for fr in sc["frames"]:
        if not fr["dets"]:
            bt_out.append(None)
            continue
        out = bt.update(np.array(fr["dets"]), np.zeros((H, W), dtype=np.int16))
        bt_out.append([[float(v) for v in row] for row in out])

    return {
        "loaded": loaded, "loaded_files": loaded_files,
        "frames": frames_out,
        "final": {
            "known": {n: [float(x) for x in v] for n, v in tracker.known_embeddings.items()},
            "files": final_files,
            "events": [event_view(e) for e in tracker._face_events],
            "metrics": {k: metrics[k] for k in ("total_tracks", "tracks_identified", "tracks_generic", "face_fn_count",
                                                "enrollments", "known_embeddings_count", "total_frames")}
                       | {"stats": metrics["match_score_stats"], "samples": len(metrics["match_scores_sample"])},
        },
        "bytetrack": bt_out,
        "db": [list(c) for c in recorder.calls],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-tracker", type=Path, required=True)
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pt, recorder = load_original(args.av_tracker)
    scenarios = []
    with tempfile.TemporaryDirectory() as workdir:
        for seed in range(args.scenarios):
            sc = make_scenario(seed)
            expected = run_original(pt, recorder, sc, workdir)
            scenarios.append({"input": sc, "expected": expected})
            ident = sum(1 for fr in expected["frames"] for r in fr["results"] if r[1] not in ("Unknown",))
            print(f"cenario {seed}: {len(sc['frames'])} frames, {ident} rostos com nome, "
                  f"{len(expected['final']['events'])} eventos, acoes={sum(1 for fr in expected['frames'] for a in fr['actions'] if not a['skipped'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scenarios": scenarios}, separators=(",", ":")), encoding="utf-8")
    print("gravado:", args.out, f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
