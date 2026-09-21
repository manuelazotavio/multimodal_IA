"""Exporta o parser de dependencias do spaCy (en_core_web_sm / pt_core_news_sm) para o app Android.

O av-tracker so le `doc.ents`, mas no spaCy o NER nao permite que uma entidade cruze a fronteira de sentenca que o parser
marcou ("Thanks, Pedro. Silva said..." -> PER "Pedro", nao "Pedro. Silva said"). Por isso o app precisa do parser para
dar os mesmos nomes. Aqui saem os dados; o Kotlin (SpacyParser) reimplementa a inferencia: tok2vec compartilhado (6
atributos) -> projecao -> maquina de transicoes arc-eager (gulosa) -> desprojetivizacao -> inicio de sentencas.

Gera em src/main/assets/ner/<lang>/:
  parser_weights.bin  tensores float32 little-endian, na ordem de parser.json["tensors"]
  parser.json         dimensoes, sementes/linhas dos embeddings, acoes (movimento, rotulo), ordem dos tensores

Uso:  python scripts/export_parser.py [--models en_core_web_sm pt_core_news_sm]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import spacy

LANG_DIR = {"en_core_web_sm": "en", "pt_core_news_sm": "pt"}


def export_model(name: str, out_root: Path) -> None:
    nlp = spacy.load(name)
    parser = nlp.get_pipe("parser")
    tok2vec = nlp.get_pipe("tok2vec").model
    out = out_root / LANG_DIR[name]
    out.mkdir(parents=True, exist_ok=True)

    hash_embeds = [l for l in tok2vec.walk() if l.name == "hashembed"]
    maxouts = [l for l in tok2vec.walk() if l.name == "maxout"]
    layernorms = [l for l in tok2vec.walk() if l.name == "layernorm"]
    columns = [l for l in tok2vec.walk() if l.name == "extract_features"][0].attrs["columns"]
    assert columns == ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"], columns

    root = parser.model
    proj = root.layers[0].layers[-1]  # tok2vec-listener >> list2array >> linear(96 -> 64)
    lower, upper = root.layers[1], root.layers[2]
    assert root.attrs["has_upper"] and not root.attrs["unseen_classes"]

    tensors: list[tuple[str, np.ndarray]] = []
    for i, h in enumerate(hash_embeds):
        tensors.append((f"embed{i}.E", h.get_param("E")))
    for i, (mo, ln) in enumerate(zip(maxouts, layernorms)):
        tensors += [(f"mix{i}.W", mo.get_param("W")), (f"mix{i}.b", mo.get_param("b")),
                    (f"mix{i}.G", ln.get_param("G")), (f"mix{i}.beta", ln.get_param("b"))]
    tensors += [("proj.W", proj.get_param("W")), ("proj.b", proj.get_param("b")),
                ("lower.W", lower.get_param("W")), ("lower.b", lower.get_param("b")), ("lower.pad", lower.get_param("pad")),
                ("upper.W", upper.get_param("W")), ("upper.b", upper.get_param("b"))]

    with open(out / "parser_weights.bin", "wb") as f:
        for _, arr in tensors:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    actions = []
    for i in range(parser.moves.n_moves):
        label = parser.moves.get_class_name(i)  # "S", "D", "L-nsubj", "R-relcl||pobj", "B-ROOT"
        move, _, dep = label.partition("-")
        actions.append({"move": move, "label": dep})

    parser_json = {
        "name": name,
        "width": int(hash_embeds[0].get_dim("nO")),
        "seeds": [int(h.attrs["seed"]) for h in hash_embeds],
        "rows": [int(h.get_dim("nV")) for h in hash_embeds],
        "encoderDepth": len(maxouts) - 1,
        "encoderPad": 4,
        "maxoutPieces": int(maxouts[0].get_dim("nP")),
        "hiddenWidth": int(lower.get_dim("nO")),
        "lowerPieces": int(lower.get_dim("nP")),
        "numFeatures": int(lower.get_dim("nF")),
        "rootLabel": "ROOT",
        "actions": actions,
        "tensors": [{"name": n, "shape": list(a.shape)} for n, a in tensors],
    }
    (out / "parser.json").write_text(json.dumps(parser_json, ensure_ascii=False), encoding="utf-8")
    size = sum(a.size for _, a in tensors) * 4 / 1e6
    print(f"{name}: {len(tensors)} tensores, {size:.1f} MB | {len(actions)} acoes | maxout pieces {parser_json['maxoutPieces']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(LANG_DIR))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "src/main/assets/ner")
    a = ap.parse_args()
    for m in a.models:
        export_model(m, a.out)
