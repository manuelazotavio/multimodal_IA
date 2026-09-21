"""Exporta o Whisper (openai/whisper-<size>) para ONNX, pronto para o app Android.

Gera em src/main/assets/whisper/:
  encoder.onnx   audio [1, 480000] (30 s, 16 kHz, zero-padded) -> cross_k_i / cross_v_i de cada camada do decoder
                 (o log-mel de 80 bandas roda dentro do grafo, igual ao WhisperFeatureExtractor)
  decoder.onnx   um passo do decoder com cache KV explicito: token, position, self_k_i/self_v_i, cross_k_i/cross_v_i
                 -> logits [1, vocab], novos self_k_i/self_v_i
  tokens.txt     base64 dos bytes de cada token (linha = id; vazio para tokens especiais)
  meta.json      ids especiais, idiomas, listas de supressao, dimensoes

Uso (env conda "tracker"): python scripts/export_whisper.py --model-dir <pasta openai/whisper-base>
  (baixe antes com huggingface_hub.snapshot_download("openai/whisper-base", local_dir=...))
"""
import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPSET = 17
N_SAMPLES = 480_000  # 30 s
N_FFT, HOP = 400, 160


class WhisperEncoderKv(nn.Module):
    """audio [1, 480000] -> log-mel -> encoder -> cross-attention K/V of every decoder layer."""

    def __init__(self, hf_model, mel_filters: np.ndarray):
        super().__init__()
        self.encoder = hf_model.model.encoder
        self.layers = hf_model.model.decoder.layers
        self.heads = hf_model.config.decoder_attention_heads
        window = torch.hann_window(N_FFT, periodic=True).double()
        n = torch.arange(N_FFT, dtype=torch.float64)[None, :]
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        angle = 2 * math.pi * k * n / N_FFT
        kernel = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window], dim=0)
        self.register_buffer("kernel", kernel.float()[:, None, :])
        self.register_buffer("mel", torch.from_numpy(mel_filters.T.copy()).float())  # [80, 201]

    def log_mel(self, audio):
        x = F.pad(audio[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")
        spec = F.conv1d(x, self.kernel, stride=HOP)  # [1, 402, 3001]
        n_freq = N_FFT // 2 + 1
        power = (spec[:, :n_freq] ** 2 + spec[:, n_freq:] ** 2)[:, :, :-1]  # drop the last frame -> 3000
        log_spec = torch.log10(torch.clamp(torch.matmul(self.mel, power), min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0

    def forward(self, audio):
        hidden = self.encoder(self.log_mel(audio)).last_hidden_state  # [1, 1500, C]
        outs = []
        for layer in self.layers:
            attn = layer.encoder_attn
            for proj in (attn.k_proj, attn.v_proj):
                t = proj(hidden)
                outs.append(t.view(1, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2))
        return tuple(outs)  # cross_k_0, cross_v_0, cross_k_1, ...


class WhisperDecoderStep(nn.Module):
    """One greedy-decoding step with explicit KV cache. Mirrors WhisperDecoder (pre-LN, GELU)."""

    def __init__(self, hf_model):
        super().__init__()
        dec = hf_model.model.decoder
        self.embed = dec.embed_tokens
        self.pos = dec.embed_positions
        self.layers = dec.layers
        self.ln = dec.layer_norm
        self.heads = hf_model.config.decoder_attention_heads
        # proj_out is tied to the embedding table: one initializer serves both the lookup and the logits Gemm.
        self.scale = (dec.embed_tokens.weight.shape[1] // self.heads) ** -0.5
        self.register_buffer("zero_bias", torch.zeros(dec.embed_tokens.weight.shape[0]))

    def split(self, t):
        return t.view(1, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2)

    def attend(self, q, k, v):
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)), dim=-1)
        return torch.matmul(att, v).transpose(1, 2).reshape(1, 1, -1)

    def forward(self, token, position, *cache):
        n = len(self.layers)
        self_kv, cross_kv = cache[: 2 * n], cache[2 * n:]
        x = self.embed(token) + self.pos.weight[position][None]  # [1, 1, C]
        new_kv = []
        for i, layer in enumerate(self.layers):
            r = x
            h = layer.self_attn_layer_norm(x)
            sa = layer.self_attn
            q = self.split(sa.q_proj(h) * self.scale)
            k = torch.cat([self_kv[2 * i], self.split(sa.k_proj(h))], dim=2)
            v = torch.cat([self_kv[2 * i + 1], self.split(sa.v_proj(h))], dim=2)
            new_kv += [k, v]
            x = r + sa.out_proj(self.attend(q, k, v))

            r = x
            h = layer.encoder_attn_layer_norm(x)
            ca = layer.encoder_attn
            q = self.split(ca.q_proj(h) * self.scale)
            x = r + ca.out_proj(self.attend(q, cross_kv[2 * i], cross_kv[2 * i + 1]))

            r = x
            h = layer.final_layer_norm(x)
            x = r + layer.fc2(F.gelu(layer.fc1(h)))
        # F.linear with a bias on 2-D input exports as Gemm(transB=1), which reads the embedding table directly;
        # without a bias it becomes MatMul plus a constant-folded transposed copy of the 106 MB table.
        logits = F.linear(self.ln(x)[:, 0], self.embed.weight, self.zero_bias)  # [1, vocab]
        return (logits, *new_kv)


def byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode, used by Whisper's byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokens_and_meta(model_dir: Path, out_dir: Path, hf_model) -> None:
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((model_dir / "added_tokens.json").read_text(encoding="utf-8"))
    gen = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    dec = byte_decoder()
    size = hf_model.config.vocab_size
    lines = [""] * size
    eot = gen["eos_token_id"]
    for token, idx in vocab.items():
        if idx < eot:  # ids >= eot are special tokens: never printed
            lines[idx] = base64.b64encode(bytes(dec[c] for c in token)).decode()
    (out_dir / "tokens.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    cfg = hf_model.config
    meta = {
        "vocabSize": size,
        "eot": eot,
        "sot": gen["decoder_start_token_id"],
        "transcribe": gen["task_to_id"]["transcribe"],
        "translate": gen["task_to_id"]["translate"],
        "noTimestamps": gen["no_timestamps_token_id"],
        "noSpeech": added["<|nocaptions|>"],
        "languages": {k.strip("<|>"): v for k, v in gen["lang_to_id"].items()},
        "suppressTokens": gen["suppress_tokens"],
        "beginSuppressTokens": gen["begin_suppress_tokens"],
        "layers": cfg.decoder_layers,
        "heads": cfg.decoder_attention_heads,
        "headDim": cfg.d_model // cfg.decoder_attention_heads,
        "maxTargetPositions": cfg.max_target_positions,
        "nSamples": N_SAMPLES,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")


def export(model_dir: Path, out_dir: Path, shrink: bool = True) -> None:
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    hf = WhisperForConditionalGeneration.from_pretrained(str(model_dir), attn_implementation="eager").eval()
    fe = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    n, h, d = hf.config.decoder_layers, hf.config.decoder_attention_heads, hf.config.d_model // hf.config.decoder_attention_heads

    encoder = WhisperEncoderKv(hf, np.asarray(fe.mel_filters)).eval()
    torch.onnx.export(
        encoder, torch.zeros(1, N_SAMPLES), str(out_dir / "encoder.onnx"),
        input_names=["audio"], output_names=[f"cross_{kv}_{i}" for i in range(n) for kv in "kv"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )

    decoder = WhisperDecoderStep(hf).eval()
    self_names = [f"self_{kv}_{i}" for i in range(n) for kv in "kv"]
    cross_names = [f"cross_{kv}_{i}" for i in range(n) for kv in "kv"]
    args = (
        torch.tensor([[50258]]), torch.tensor([0]),
        *[torch.zeros(1, h, 3, d) for _ in self_names], *[torch.zeros(1, h, 1500, d) for _ in cross_names],
    )
    torch.onnx.export(
        decoder, args, str(out_dir / "decoder.onnx"),
        input_names=["token", "position", *self_names, *cross_names],
        output_names=["logits", *[f"new_{s}" for s in self_names]],
        dynamic_axes={**{s: {2: "past"} for s in self_names}, **{f"new_{s}": {2: "total"} for s in self_names}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    export_tokens_and_meta(model_dir, out_dir, hf)

    if shrink:
        # int8 dynamic quantization was tried and rejected: it breaks the encoder (CER ~33% on real speech).
        # fp16 weight storage keeps fp32 math, so transcripts stay identical to HuggingFace's.
        from onnx_utils import shrink_fp16_storage

        for name in ("encoder", "decoder"):
            before, after = shrink_fp16_storage(out_dir / f"{name}.onnx")
            print(f"{name}: {before:.1f} MB -> {after:.1f} MB (fp16 storage)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "src/main/assets/whisper")
    parser.add_argument("--no-shrink", action="store_true", help="keep fp32 weights (about twice the size)")
    a = parser.parse_args()
    export(a.model_dir, a.out, shrink=not a.no_shrink)
