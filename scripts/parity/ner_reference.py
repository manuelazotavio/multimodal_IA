"""Referencia em numpy do NER do spaCy, lida a partir dos arquivos exportados por scripts/export_ner.py.

Serve de especificacao executavel para o SpacyNer em Kotlin: valida que os dados exportados bastam e que a
maquina de estados, o HashEmbed e o encoder foram entendidos direito (comparando com o `ner` real do spaCy).
"""
import json
from pathlib import Path

import numpy as np

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF


def load(model_dir: Path):
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    raw = np.fromfile(model_dir / "weights.bin", dtype="<f4")
    tensors, pos = {}, 0
    for t in meta["tensors"]:
        n = int(np.prod(t["shape"]))
        tensors[t["name"]] = raw[pos:pos + n].reshape(t["shape"])
        pos += n
    assert pos == raw.size
    return meta, tensors


def murmur3_x86_128_u64(val, seed):
    """thinc MurmurHash3_x86_128_uint64: 4 chaves de 32 bits."""
    h1 = (val * 0x87c37b91114253d5) & M64
    h1 = ((h1 << 31) | (h1 >> 33)) & M64
    h1 = (h1 * 0x4cf5ad432745937f) & M64
    h1 ^= seed
    h1 ^= 8
    h2 = seed ^ 8
    h1 = (h1 + h2) & M64
    h2 = (h2 + h1) & M64
    h1 ^= h1 >> 33; h1 = (h1 * 0xff51afd7ed558ccd) & M64; h1 ^= h1 >> 33; h1 = (h1 * 0xc4ceb9fe1a85ec53) & M64; h1 ^= h1 >> 33
    h2 ^= h2 >> 33; h2 = (h2 * 0xff51afd7ed558ccd) & M64; h2 ^= h2 >> 33; h2 = (h2 * 0xc4ceb9fe1a85ec53) & M64; h2 ^= h2 >> 33
    h1 = (h1 + h2) & M64
    h2 = (h2 + h1) & M64
    return [h1 & M32, h1 >> 32, h2 & M32, h2 >> 32]


def layer_norm(x, g, b):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True) + 1e-8
    return (x - mu) * var ** -0.5 * g + b


def maxout(x, w, b):
    n_o, n_p, n_i = w.shape
    y = x @ w.reshape(n_o * n_p, n_i).T + b.reshape(n_o * n_p)
    return y.reshape(x.shape[0], n_o, n_p).max(axis=2)


def seq2col(x, n_w=1):
    n = x.shape[0]
    pad = np.zeros((n_w, x.shape[1]), dtype=x.dtype)
    ext = np.concatenate([pad, x, pad])
    return np.concatenate([ext[i:i + n] for i in range(2 * n_w + 1)], axis=1)


def tok2vec(meta, t, attr_keys):
    """attr_keys: uint64 [n, 4] (NORM, PREFIX, SUFFIX, SHAPE) -> tokvecs [n, hiddenWidth]."""
    n = attr_keys.shape[0]
    cols = []
    for i in range(4):
        e = t[f"embed{i}.E"]
        rows = meta["rows"][i]
        vec = np.zeros((n, e.shape[1]), dtype=np.float32)
        for k in range(n):
            for key in murmur3_x86_128_u64(int(attr_keys[k, i]), meta["seeds"][i]):
                vec += 0  # (placeholder to keep the loop shape obvious)
                vec[k] += e[key % rows]
        cols.append(vec)
    x = np.concatenate(cols, axis=1)
    x = layer_norm(maxout(x, t["mix0.W"], t["mix0.b"]), t["mix0.G"], t["mix0.beta"])

    pad = meta["encoderPad"]
    flat = np.zeros((n + 2 * pad, x.shape[1]), dtype=np.float32)
    flat[pad:pad + n] = x
    for d in range(1, meta["encoderDepth"] + 1):
        y = layer_norm(maxout(seq2col(flat, 1), t[f"mix{d}.W"], t[f"mix{d}.b"]), t[f"mix{d}.G"], t[f"mix{d}.beta"])
        flat = flat + y
    x = flat[pad:pad + n]
    return x @ t["proj.W"].T + t["proj.b"]


def run_ner(meta, t, tokvecs, is_space):
    n = tokvecs.shape[0]
    if n == 0:
        return []
    nf, nh, npieces = meta["numFeatures"], meta["hiddenWidth"], meta["lowerPieces"]
    lw = t["lower.W"].reshape(nf * nh * npieces, tokvecs.shape[1])
    cached = (tokvecs @ lw.T).reshape(n, nf, nh, npieces)
    pad = t["lower.pad"].reshape(nf, nh, npieces)
    bias = t["lower.b"].reshape(nh, npieces)
    actions = meta["actions"]

    b_i, ents = 0, []  # ents: [start, end, label]; end == -1 while open
    while b_i < n:
        def B(i): return b_i + i if b_i + i < n else -1
        open_ent = bool(ents) and ents[-1][1] == -1
        ids = [B(0) if B(0) >= 0 else -1, ents[-1][0] if open_ent else -1, -1]
        ids[2] = -1 if ids[0] == -1 or ids[1] == -1 else ids[0] - 1
        acc = np.zeros((nh, npieces), dtype=np.float32)
        for f, tid in enumerate(ids):
            acc += pad[f] if tid < 0 else cached[tid, f]
        hidden = (acc + bias).max(axis=1)
        scores = hidden @ t["upper.W"].T + t["upper.b"]

        buffer_length = n - b_i
        best = -1
        for i, a in enumerate(actions):
            mv, lab = a["move"], a["label"]
            if mv == "B": valid = (not open_ent) and buffer_length >= 2 and lab != "" and not is_space[b_i]
            elif mv == "I": valid = open_ent and buffer_length >= 2 and lab != "" and ents[-1][2] == lab
            elif mv == "L": valid = lab != "" and open_ent and ents[-1][2] == lab
            elif mv == "U": valid = lab != "" and (not open_ent) and not is_space[b_i]
            elif mv == "O": valid = not open_ent
            else: valid = False
            if valid and (best == -1 or scores[i] > scores[best]):
                best = i
        assert best >= 0, "sem acao valida"
        mv, lab = actions[best]["move"], actions[best]["label"]
        if mv == "B": ents.append([b_i, -1, lab])
        elif mv == "U": ents.append([b_i, b_i + 1, lab])
        elif mv == "L": ents[-1][1] = b_i + 1
        b_i += 1
    return [(s, e, l) for s, e, l in ents if s != -1 and e != -1]
