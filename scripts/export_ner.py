"""Exporta o NER do spaCy (en_core_web_sm / pt_core_news_sm) para o app Android.

O av-tracker usa `spacy.load(...)` so para achar nomes de pessoas (PERSON/PER) no texto transcrito. Aqui o
componente `ner` e o tokenizador sao exportados como dados; o Kotlin (SpacyNer) reimplementa a inferencia:
tokenizador -> atributos lexicais (NORM/PREFIX/SUFFIX/SHAPE) -> HashEmbed -> encoder CNN -> maquina de transicoes BILUO.

Gera em src/main/assets/ner/<lang>/:
  weights.bin       tensores float32 little-endian, na ordem de model.json["tensors"]
  model.json        dimensoes, ordem/forma dos tensores, acoes (movimento, rotulo), labels de pessoa
  tokenizer.json    regex de prefixo/sufixo/infixo/url, regras de excecao (ORTH/NORM), faster_heuristics
  lexeme_norm.json  tabela de normalizacao (NORM = tabela.get(texto) ou texto.lower())

Uso:  python scripts/export_ner.py [--models en_core_web_sm pt_core_news_sm]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import spacy
from spacy import symbols
from spacy.lang.norm_exceptions import BASE_NORMS

LANG_DIR = {"en_core_web_sm": "en", "pt_core_news_sm": "pt"}


def find(model, name):
    return [layer for layer in model.walk() if layer.name == name]


def export_model(name: str, out_root: Path) -> None:
    nlp = spacy.load(name)
    ner = nlp.get_pipe("ner")
    root = ner.model
    out = out_root / LANG_DIR[name]
    out.mkdir(parents=True, exist_ok=True)

    tok2vec = root.layers[0]
    hash_embeds = [l for l in tok2vec.walk() if l.name == "hashembed"]
    maxouts = [l for l in tok2vec.walk() if l.name == "maxout"]
    layernorms = [l for l in tok2vec.walk() if l.name == "layernorm"]
    linear = tok2vec.layers[-1]  # projection 96 -> 64 after list2array
    lower, upper = root.layers[1], root.layers[2]

    tensors: list[tuple[str, np.ndarray]] = []
    for i, h in enumerate(hash_embeds):
        tensors.append((f"embed{i}.E", h.get_param("E")))
    # 1 maxout+layernorm mixing the concatenated embeddings, then 4 encoder layers
    for i, (mo, ln) in enumerate(zip(maxouts, layernorms)):
        tensors += [(f"mix{i}.W", mo.get_param("W")), (f"mix{i}.b", mo.get_param("b")),
                    (f"mix{i}.G", ln.get_param("G")), (f"mix{i}.beta", ln.get_param("b"))]
    tensors += [("proj.W", linear.get_param("W")), ("proj.b", linear.get_param("b")),
                ("lower.W", lower.get_param("W")), ("lower.b", lower.get_param("b")), ("lower.pad", lower.get_param("pad")),
                ("upper.W", upper.get_param("W")), ("upper.b", upper.get_param("b"))]

    with open(out / "weights.bin", "wb") as f:
        for _, arr in tensors:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    actions = []
    for i in range(ner.moves.n_moves):
        label = ner.moves.get_class_name(i)
        move, _, entity = label.partition("-")
        actions.append({"move": move, "label": entity})

    model_json = {
        "name": name,
        "width": int(hash_embeds[0].get_dim("nO")),
        "seeds": [int(h.attrs["seed"]) for h in hash_embeds],
        "rows": [int(h.get_dim("nV")) for h in hash_embeds],
        "encoderDepth": len(maxouts) - 1,
        "encoderPad": 4,
        "maxoutPieces": 3,
        "hiddenWidth": int(lower.get_dim("nO")),
        "lowerPieces": int(lower.get_dim("nP")),
        "numFeatures": int(lower.get_dim("nF")),
        "personLabels": [l for l in ner.labels if l in ("PER", "PERSON")],
        "actions": actions,
        # StringStore.add() returns a fixed symbol id, not a hash, for these 457 strings (e.g. the word shape "X" -> 101).
        "symbols": {k: int(v) for k, v in symbols.IDS.items()},
        "tensors": [{"name": n, "shape": list(a.shape)} for n, a in tensors],
    }
    (out / "model.json").write_text(json.dumps(model_json), encoding="utf-8")

    tok = nlp.tokenizer
    rules = []
    for string, specs in tok.rules.items():
        rules.append({"string": string, "tokens": [{"orth": s[65], **({"norm": s[67]} if 67 in s else {})} for s in specs]})
    tok_json = {
        "prefix": tok.prefix_search.__self__.pattern,
        "suffix": tok.suffix_search.__self__.pattern,
        "infix": tok.infix_finditer.__self__.pattern,
        "urlMatch": tok.url_match.__self__.pattern if tok.url_match else None,
        "tokenMatch": tok.token_match.__self__.pattern if tok.token_match else None,
        "fasterHeuristics": bool(tok.faster_heuristics),
        "rules": rules,
    }
    (out / "tokenizer.json").write_text(json.dumps(tok_json, ensure_ascii=False), encoding="utf-8")
    # A tabela lexeme_norm do spaCy e indexada pelo ID da string (hash ou simbolo), nao pela string. NORM =
    # lexeme_norm[id(texto)] se existir, senao BASE_NORMS[texto], senao texto.lower().
    table = {str(int(k)): v for k, v in nlp.vocab.lookups.get_table("lexeme_norm").items()}
    (out / "lexeme_norm.json").write_text(json.dumps({"table": table, "base": dict(BASE_NORMS)}, ensure_ascii=False), encoding="utf-8")
    size = sum(a.size for _, a in tensors) * 4 / 1e6
    print(f"{name}: {len(tensors)} tensores, {size:.1f} MB | {len(actions)} acoes | {len(rules)} regras | lexeme_norm {len(table)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(LANG_DIR))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "src/main/assets/ner")
    a = ap.parse_args()
    for m in a.models:
        export_model(m, a.out)
