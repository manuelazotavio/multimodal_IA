"""Referencia do banco SQLite: roda o TrackerDB ORIGINAL (src/database.py do av-tracker) sobre uma sequencia de operacoes
e grava o conteudo de todas as tabelas. O teste Kotlin (TrackerDbTest) repete as mesmas operacoes e exige o mesmo.

Seguranca: o banco e criado numa pasta temporaria (db_path explicito). O av_tracker.db real nunca e aberto.

Uso:  python scripts/parity/db_reference.py --av-tracker C:/Users/manu/av-tracker
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TS_COLUMNS = {"created_at", "validated_at"}
JSON_COLUMNS = {"config", "summary", "metrics_json", "details_json"}
TABLES = ["speakers", "voice_embeddings", "face_embeddings", "sessions", "segments", "validations", "validation_lines"]


def q32(vec):
    return [float(f"{float(x):.9g}") for x in vec]


def make_ops():
    rng = np.random.RandomState(11)
    vec = lambda: q32(rng.randn(8))
    seg = lambda i, **kw: {
        "timestamp": f"2026-01-01T10:10:{i:02d}.100", "decision": "VERIFIER_HIGH (Manuela=0.91)", "decision_type": "VERIFIER_HIGH",
        "raw_best_name": "Manuela" if i % 2 == 0 else None, "raw_conf": 0.9123 if i % 3 else 0.0, "chosen_name": "Manuela", "chosen_conf": 0.9,
        "final_name": "Manuela" if i % 4 else None, "final_speaker": f"spk_{i:03d}", "is_identified": bool(i % 2), "audio_gender": "female" if i % 3 == 0 else None,
        "n_candidates": 2, "is_fn": i == 2, "fp_risk": "high" if i == 1 else "none", "fp_reasons": [], "text_len": 20 + i, "whisper_ms": 812.5 + i,
        "verifier_ms": 41.0, "segment_ms": 990.5, "candidates": [["Manuela", 0.9123], ["Pedro", 0.5]], **kw,
    }
    return [
        {"op": "add_speaker", "name": "Manuela", "gender": "female"},
        {"op": "add_speaker", "name": "Manuela"},  # INSERT OR IGNORE: mesma linha
        {"op": "add_speaker", "name": "João Nascimento"},
        {"op": "add_speaker", "name": "Person_1"},
        {"op": "voice", "speaker": "Manuela", "vec": vec(), "source": "Manuela_20260101_101010.npy"},
        {"op": "voice", "speaker": "Manuela", "vec": vec(), "source": "Manuela_20260101_101011_auto.npy"},
        {"op": "voice", "speaker": "João Nascimento", "vec": vec(), "source": None},
        {"op": "face", "speaker": "Person_1", "vec": vec(), "source": "Person_1_auto.npy"},
        {"op": "face", "speaker": "Manuela", "vec": vec(), "source": "Manuela_20260101_auto.npy"},
        {"op": "face", "speaker": "Manuela", "vec": vec(), "source": "Manuela_20260102_auto.npy"},
        {"op": "deactivate_voice", "index": 1},
        {"op": "deactivate_face", "index": 2},
        {"op": "session", "session_id": "20260101_101010", "started_at": "2026-01-01T10:10:10.123456",
         "config": {"verifier_threshold": 0.8, "num_speakers": None, "whisper_size": "medium", "sample_rate": 16000},
         "summary": {"total_segments": 3, "fn_rate": 0.3333, "decision_counts": {"VERIFIER_HIGH": 2, "FACE_ONLY": 1}, "nome": "acentuação ✓"},
         "transcript": "[10:10:11] Manuela: Bom dia a todos.\n[10:10:15] spk_002: Olá, \"tudo\" bem?\n", "audio_file": "realtime_sessions/audio_20260101_101010.wav"},
        {"op": "segments", "session": "20260101_101010", "segments": [seg(0), seg(1), seg(2), {"final_speaker": "spk_009"}, {"final_name": "", "final_speaker": "spk_010", "text": "x"}]},
        {"op": "session", "session_id": "20260102_090000", "started_at": None, "config": None, "summary": {}, "transcript": None, "audio_file": None},
        {"op": "validation", "session": "20260101_101010", "accuracy": 0.8, "named_accuracy": 0.75, "total_lines": 5, "correct": 4, "incorrect": 1,
         "details": {"errors": [{"seg": 2, "expected": "Pedro"}]}, "lines": [[0, "Manuela", "Manuela"], [2, "spk_002", "Pedro"]]},
        {"op": "rename_speaker", "old": "Person_1", "new": "Kauan"},
        {"op": "session", "session_id": "20260101_101010", "started_at": "2026-01-01T10:10:10.123456", "config": {"a": 1}, "summary": {"total_segments": 9},
         "transcript": "replaced", "audio_file": None},  # INSERT OR REPLACE: apaga segmentos e validacoes em cascata
        {"op": "segments", "session": "20260101_101010", "segments": [seg(4)]},
        {"op": "delete_speaker", "name": "João Nascimento"},  # cascata nas vozes
        {"op": "delete_face", "index": 1},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-tracker", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.av_tracker))
    from src.database import TrackerDB

    ops = make_ops()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parity.db")
        db = TrackerDB(db_path=path)
        assert os.path.abspath(db.db_path).startswith(os.path.abspath(tmp)), "o banco de teste tem de ficar na pasta temporaria"

        speakers, sessions, voice_ids, face_ids = {}, {}, [], []
        for op in ops:
            kind = op["op"]
            if kind == "add_speaker":
                speakers[op["name"]] = db.add_speaker(op["name"], op.get("gender"))
            elif kind == "voice":
                voice_ids.append(db.save_voice_embedding(speakers[op["speaker"]], np.array(op["vec"], dtype=np.float32), op["source"]))
            elif kind == "face":
                face_ids.append(db.save_face_embedding(speakers[op["speaker"]], np.array(op["vec"], dtype=np.float32), op["source"]))
            elif kind == "deactivate_voice":
                db.deactivate_voice_embedding(voice_ids[op["index"]])
            elif kind == "deactivate_face":
                db.deactivate_face_embedding(face_ids[op["index"]])
            elif kind == "delete_face":
                db.delete_face_embedding(face_ids[op["index"]])
            elif kind == "session":
                sessions[op["session_id"]] = db.save_session(op["session_id"], op["started_at"], op["config"], op["summary"], op["transcript"], op["audio_file"])
            elif kind == "segments":
                db.save_segments(sessions[op["session"]], op["segments"])
            elif kind == "validation":
                db.save_validation(sessions[op["session"]], op["accuracy"], op["named_accuracy"], op["total_lines"], op["correct"], op["incorrect"],
                                   op["details"], [tuple(l) for l in op["lines"]])
            elif kind == "rename_speaker":
                db.rename_speaker(op["old"], op["new"])
            elif kind == "delete_speaker":
                db.delete_speaker(op["name"])

        conn = db._get_conn()
        tables = {}
        for t in TABLES:
            rows = []
            for r in conn.execute(f"SELECT * FROM {t} ORDER BY id"):
                d = dict(r)
                for k in list(d):
                    if k in TS_COLUMNS:
                        d[k] = "<ts>"
                    elif k == "embedding":
                        d[k] = [float(x) for x in np.frombuffer(d[k], dtype=np.float32)]
                    elif k in JSON_COLUMNS and d[k] is not None:
                        d[k] = json.loads(d[k])
                rows.append(d)
            tables[t] = rows

        def clean(v):
            if isinstance(v, dict):
                return {k: ("<ts>" if k in TS_COLUMNS else json.loads(x) if (k in JSON_COLUMNS and isinstance(x, str)) else clean(x))
                        for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            if isinstance(v, np.ndarray):
                return [float(x) for x in v]
            return v

        expected = {
            "tables": tables,
            "all_voice": {k: [float(x) for x in v] for k, v in db.get_all_voice_embeddings().items()},
            "all_face": {k: [float(x) for x in v] for k, v in db.get_all_face_embeddings().items()},
            "all_voice_inactive_too": {k: [float(x) for x in v] for k, v in db.get_all_voice_embeddings(active_only=False).items()},
            "list_speakers": clean(db.list_speakers()),
            "list_sessions": clean(db.list_sessions()),
            "get_session": clean(db.get_session("20260101_101010")),
            "missing_session": db.get_session("nope"),
            "get_speaker": clean(db.get_speaker("Manuela")),
            "count_voice": db.count_voice_embeddings(speakers["Manuela"]),
            "get_segments": clean(db.get_segments(sessions["20260101_101010"])),
            "trend": clean(db.get_session_accuracy_trend()),
        }
        db.close()

    out = ROOT / "src/test/resources/parity/db_reference.json"
    out.write_text(json.dumps({"ops": ops, "expected": expected}, ensure_ascii=False), encoding="utf-8")
    print("gravado", out, {t: len(r) for t, r in tables.items()})


if __name__ == "__main__":
    main()
