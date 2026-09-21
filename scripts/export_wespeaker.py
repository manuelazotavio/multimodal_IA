"""Exporta o embedding de falante do pyannote 3.1 (pyannote/wespeaker-voxceleb-resnet34-LM) para ONNX.

O pipeline de diarizacao calcula um embedding por (janela de 10 s, falante local), ponderado pela mascara de atividade
desse falante. O grafo exportado faz tudo de uma vez: fbank estilo Kaldi (a mesma conta do torchaudio, com a DFT como
multiplicacao de matrizes), subtracao da media, ResNet34 e a pooling de estatisticas ponderada (a mascara de 589 quadros
e reduzida ao numero de quadros do ResNet com "nearest", como o StatsPool faz).

  wespeaker_resnet34.onnx   waveform [B, 160000] (10 s, 16 kHz, em [-1, 1])  +  weights [B, 589]  ->  embedding [B, 256]

Uso (env conda "tracker"; precisa de onnx/onnxslim via PYTHONPATH):
  python scripts/export_wespeaker.py
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
CHUNK = 160000  # 10 s
WINDOW, SHIFT, FFT = 400, 160, 512
FRAMES = 1 + (CHUNK - WINDOW) // SHIFT  # 998
MASK_FRAMES = 589
MEL_BINS = 80
OPSET = 17


def kaldi_mel_bank(num_bins=MEL_BINS, fft=FFT, sample_freq=SR, low_freq=20.0, high_freq=0.0):
    """torchaudio.compliance.kaldi.get_mel_banks sem VTLN, ja com a coluna de Nyquist zerada: [num_bins, fft // 2 + 1]."""
    from torchaudio.compliance.kaldi import get_mel_banks

    bins, _ = get_mel_banks(num_bins, fft, sample_freq, low_freq, high_freq, 100.0, -500.0, 1.0)
    return F.pad(bins, (0, 1)).float()


class WeSpeakerOnnx(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        n = torch.arange(FFT, dtype=torch.float64)[:, None]
        k = torch.arange(FFT // 2 + 1, dtype=torch.float64)[None, :]
        angle = 2 * math.pi * n * k / FFT
        self.register_buffer("cos", torch.cos(angle).float())
        self.register_buffer("sin", torch.sin(angle).float())
        self.register_buffer("window", torch.hamming_window(WINDOW, periodic=False, alpha=0.54, beta=0.46))
        self.register_buffer("mel", kaldi_mel_bank().T.contiguous())  # [257, 80]
        self.eps = float(np.finfo(np.float32).eps)

    def fbank(self, waveform):
        x = waveform * 32768.0  # waveforms * (1 << 15)
        frames = x.unfold(1, WINDOW, SHIFT)  # [B, 998, 400]
        frames = frames - frames.mean(dim=2, keepdim=True)  # remove_dc_offset
        shifted = torch.cat([frames[:, :, :1], frames[:, :, :-1]], dim=2)  # replicate pad on the left
        frames = (frames - 0.97 * shifted) * self.window  # preemphasis, then the window
        frames = F.pad(frames, (0, FFT - WINDOW))  # to 512
        real = frames @ self.cos
        imag = frames @ self.sin
        power = real * real + imag * imag
        mel = torch.clamp(power @ self.mel, min=self.eps).log()
        return mel - mel.mean(dim=1, keepdim=True)  # [B, 998, 80]

    def forward(self, waveform, weights):
        r = self.resnet
        x = self.fbank(waveform).permute(0, 2, 1).unsqueeze(1)  # [B, 1, 80, 998]
        out = F.relu(r.bn1(r.conv1(x)))
        out = r.layer4(r.layer3(r.layer2(r.layer1(out))))  # [B, 256, 10, 125]
        seq = out.flatten(1, 2)  # (dimension channel) frames -> [B, 2560, 125]

        w = F.interpolate(weights.unsqueeze(1), size=seq.shape[-1], mode="nearest")  # [B, 1, 125]
        v1 = w.sum(dim=2) + 1e-8
        mean = (seq * w).sum(dim=2) / v1
        dx2 = (seq - mean.unsqueeze(2)) ** 2
        v2 = (w * w).sum(dim=2)
        var = (dx2 * w).sum(dim=2) / (v1 - v2 / v1 + 1e-8)
        stats = torch.cat([mean, var.sqrt()], dim=1)
        return r.seg_1(stats)


def export(out_path: Path, shrink: bool) -> None:
    from pyannote.audio import Model

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").eval()
    wrapper = WeSpeakerOnnx(model.resnet).eval()

    # the wrapper must agree with pyannote's own forward before anything is exported
    torch.manual_seed(0)
    wav = torch.randn(2, 1, CHUNK) * 0.1
    weights = (torch.rand(2, MASK_FRAMES) > 0.4).float()
    with torch.no_grad():
        reference = model(wav, weights=weights)
        ours = wrapper(wav[:, 0], weights)
    cos = F.cosine_similarity(reference, ours).min().item()
    print(f"wrapper vs pyannote: min cosine {cos:.6f}, max abs diff {(reference - ours).abs().max().item():.2e}")
    assert cos > 0.9999

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (wav[:1, 0], weights[:1]), str(out_path),
        input_names=["waveform", "weights"], output_names=["embedding"],
        dynamic_axes={"waveform": {0: "batch"}, "weights": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    if shrink:
        from onnx_utils import shrink_fp16_storage

        before, after = shrink_fp16_storage(out_path)
        print(f"{out_path.name}: {before:.1f} MB -> {after:.1f} MB (fp16 storage)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "src/main/assets/models/wespeaker_resnet34.onnx")
    ap.add_argument("--no-shrink", action="store_true")
    a = ap.parse_args()
    export(a.out, shrink=not a.no_shrink)
