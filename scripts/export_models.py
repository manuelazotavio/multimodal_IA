"""Exporta modelos e perfis do av-tracker para os assets do app Android.

Gera em src/main/assets/:
  models/yolov8n_face.onnx   (detector de faces, entrada 1x3x640x640 RGB /255)
  models/edgeface_xxs.onnx   (embedding facial 512-d, entrada Nx3x112x112, batch dinamico)
  models/ecapa_voxceleb.onnx (embedding de voz 192-d, entrada 1xT waveform 16 kHz em [-1, 1])
  models/light_asd.onnx      (quem esta falando: 25 recortes 112x112 0-255 + 1 s de audio -> 25 probabilidades)
  models/pyannote_segmentation.onnx (diarizacao: waveform 1x1xT -> log-probs powerset 1xFx7, T dinamico)
  face_embeddings/<slug>.json (um JSON por arquivo .npy de data/face_embeddings: {"file", "embedding"})
  voice_embeddings/<slug>.json (um JSON por arquivo .npy de data/embeddings, formato StoredVoice.kt)
  models/silero_vad_v6.onnx  (VAD do faster-whisper, copiado do pacote instalado)
  models/wespeaker_resnet34.onnx (embedding de falante do pyannote 3.1: waveform 1x160000 + mascara 1x589; scripts/export_wespeaker.py)

Uso (no env conda "tracker", que tem torch/ultralytics/timm/speechbrain):
  python scripts/export_models.py --av-tracker C:/Users/manu/av-tracker
Requer os pacotes `onnx` e `onnxslim` importaveis (ex.: pip install --target <dir> e PYTHONPATH).
"""
import argparse
import json
import math
import os
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx_utils import shrink_fp16_storage

OPSET = 17


class EdgeFaceXXS(nn.Module):
    """Mesmo wrapper de av-tracker/src/personid_tracker.py (_EdgeFaceXXS)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model("edgenext_xx_small")
        self.model.reset_classifier(512)

    def forward(self, x):
        return self.model(x)


def export_yolo_face(pt_path: Path, out_path: Path) -> None:
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO

    exported = YOLO(str(pt_path)).export(
        format="onnx", imgsz=640, opset=OPSET, simplify=True, dynamic=False, half=False, device="cpu"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, out_path)


def export_edgeface(pt_path: Path, out_path: Path) -> None:
    net = EdgeFaceXXS()
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    net.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        torch.randn(1, 3, 112, 112),
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


class EcapaWaveform(nn.Module):
    """speechbrain/spkrec-ecapa-voxceleb de ponta a ponta: waveform [1, T] -> embedding [1, 192].

    Replica EncoderClassifier.encode_batch (Fbank 80 mels -> normalizacao de media por sentenca
    -> ECAPA_TDNN). O STFT vira uma conv1d com base DFT * janela Hamming (padding de zeros nas
    pontas, center=True), equivalente ao torch.stft do speechbrain, mas exportavel para ONNX.
    """

    N_FFT, HOP = 400, 160

    def __init__(self, classifier):
        super().__init__()
        fbank = classifier.mods.compute_features
        assert (fbank.compute_STFT.n_fft, fbank.compute_STFT.hop_length) == (self.N_FFT, self.HOP)
        window = fbank.compute_STFT.window.double()
        n = torch.arange(self.N_FFT, dtype=torch.float64)[None, :]
        k = torch.arange(self.N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        angle = 2 * math.pi * k * n / self.N_FFT
        kernel = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window], dim=0)
        self.register_buffer("kernel", kernel.float()[:, None, :])  # [402, 1, 400]
        self.mel = fbank.compute_fbanks
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, waveform):
        x = F.pad(waveform[:, None, :], (self.N_FFT // 2, self.N_FFT // 2))
        spec = F.conv1d(x, self.kernel, stride=self.HOP)
        n_freq = self.N_FFT // 2 + 1
        power = (spec[:, :n_freq] ** 2 + spec[:, n_freq:] ** 2).transpose(1, 2)  # [1, frames, 201]
        feats = self.mel(power)  # [1, frames, 80] log-mel com top_db, como no speechbrain
        feats = feats - feats.mean(dim=1, keepdim=True)  # InputNormalization(sentence, std_norm=False)
        return self.embedding_model(feats).squeeze(1)  # [1, 192]


def export_ecapa(out_path: Path) -> None:
    import tempfile

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    with tempfile.TemporaryDirectory() as cache:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=cache,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # symlinks exigem privilegio de admin no Windows
        )
        model = EcapaWaveform(classifier).eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 48000),
        str(out_path),
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={"waveform": {1: "samples"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


class MfccFrontend(nn.Module):
    """librosa.feature.mfcc(y, sr=16000, n_mfcc=13, n_fft=400, hop_length=160, center=False), as used by
    LightASDDetector._compute_mfcc, followed by the detector's zero-pad/trim to 100 frames.

    Input: waveform [1, N]. Output: [1, 100, 13].
    """

    N_FFT, HOP, FRAMES, N_MELS, N_MFCC = 400, 160, 100, 128, 13

    def __init__(self):
        super().__init__()
        import librosa
        import scipy.fftpack
        import scipy.signal

        window = scipy.signal.get_window("hann", self.N_FFT, fftbins=True)
        n = np.arange(self.N_FFT)[None, :]
        k = np.arange(self.N_FFT // 2 + 1)[:, None]
        angle = 2 * np.pi * k * n / self.N_FFT
        kernel = np.concatenate([np.cos(angle) * window, -np.sin(angle) * window], axis=0)
        self.register_buffer("kernel", torch.from_numpy(kernel).float()[:, None, :])  # [402, 1, 400]
        mel = librosa.filters.mel(sr=16000, n_fft=self.N_FFT, n_mels=self.N_MELS)  # slaney, [128, 201]
        self.register_buffer("mel", torch.from_numpy(mel).float())
        dct = scipy.fftpack.dct(np.eye(self.N_MELS), axis=0, type=2, norm="ortho")[: self.N_MFCC]
        self.register_buffer("dct", torch.from_numpy(dct).float())

    def forward(self, audio):
        spec = F.conv1d(audio[:, None, :], self.kernel, stride=self.HOP)  # center=False: no padding
        n_freq = self.N_FFT // 2 + 1
        power = spec[:, :n_freq] ** 2 + spec[:, n_freq:] ** 2  # [1, 201, F]
        mel = torch.matmul(self.mel, power)  # [1, 128, F]
        db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)  # power_to_db(top_db=80)
        mfcc = torch.matmul(self.dct, db).transpose(1, 2)  # [1, F, 13]
        mfcc = mfcc[:, : self.FRAMES]
        return F.pad(mfcc, (0, 0, 0, self.FRAMES - mfcc.shape[1]))


class LightAsdOnnx(nn.Module):
    """Light-ASD end to end: 25 grayscale 112x112 face crops (0-255) + 1 s of audio -> 25 speaking probabilities."""

    def __init__(self, asd, fc):
        super().__init__()
        self.asd, self.fc, self.mfcc = asd, fc, MfccFrontend()

    def forward(self, video, audio):
        v = self.asd.forward_visual_frontend(video)
        a = self.asd.forward_audio_frontend(self.mfcc(audio))
        out = self.asd.forward_audio_visual_backend(a, v)
        return torch.softmax(self.fc(out), dim=-1)[:, 1]


def load_light_asd(av_tracker: Path, onnx_friendly: bool = True):
    import sys

    sys.path.insert(0, str(av_tracker))
    from src.light_asd_detector import _ASD_Model

    state = torch.load(
        av_tracker / "pretrained_models/Light-ASD-repo/weight/pretrain_AVA_CVPR.model", map_location="cpu", weights_only=False
    )
    asd = _ASD_Model()
    # O checkpoint oficial chama as GRUs de gru_forward/gru_backward; a copia da arquitetura em
    # av-tracker/src/light_asd_detector.py usa fwd/bwd, entao la o load_state_dict falha e o
    # av-tracker cai sempre no fallback de pixel-diff. Aqui as chaves sao remapeadas.
    renames = {"GRU.gru_forward.": "GRU.fwd.", "GRU.gru_backward.": "GRU.bwd."}

    def remap(key: str) -> str:
        for old, new in renames.items():
            key = key.replace(old, new)
        return key

    asd.load_state_dict({remap(k[len("model."):]): v for k, v in state.items() if k.startswith("model.")})
    if onnx_friendly:
        # O encoder de audio aplica MaxPool3d((1,1,3), stride (1,1,2)) a tensores 4D [B,C,F,T]; o PyTorch
        # trata isso como pooling so no eixo do tempo, mas o exportador ONNX gera um MaxPool invalido.
        # MaxPool2d((1,3), stride (1,2), padding (0,1)) e numericamente identico.
        asd.audioEncoder.pool1 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
        asd.audioEncoder.pool2 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
    fc = nn.Linear(128, 2)
    fc.weight.data, fc.bias.data = state["lossAV.FC.weight"], state["lossAV.FC.bias"]
    return LightAsdOnnx(asd.eval(), fc.eval()).eval()


def export_light_asd(av_tracker: Path, out_path: Path) -> None:
    model = load_light_asd(av_tracker)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.rand(1, 25, 112, 112) * 255, torch.randn(1, 16000) * 0.1),
        str(out_path),
        input_names=["video", "audio"],
        output_names=["scores"],
        dynamic_axes={"audio": {1: "samples"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def export_segmentation(out_path: Path) -> None:
    """pyannote/segmentation-3.0 (powerset, 3 locais): waveform [1, 1, T] -> log-probabilidades [1, frames, 7]."""
    from pyannote.audio import Model

    model = Model.from_pretrained("pyannote/segmentation-3.0").eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 1, 160000),
        str(out_path),
        input_names=["waveform"],
        output_names=["log_probs"],
        dynamic_axes={"waveform": {2: "samples"}, "log_probs": {1: "frames"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "person"


def export_face_embeddings(emb_dir: Path, out_dir: Path) -> int:
    """Um JSON por arquivo .npy de data/face_embeddings (inclusive os genericos Person_N).

    O app repete no aparelho o mesmo carregamento do PersonIDTracker (nome pelo arquivo, media por pessoa,
    consolidacao, renomeacao e limpeza de arquivos), entao precisa dos arquivos crus, nao de um resumo por pessoa.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    used: set[str] = set()
    count = 0
    for file in sorted(os.listdir(emb_dir)):
        if not file.endswith(".npy"):
            continue
        slug = slugify(file[:-4])
        while slug in used:  # colisao de slug (acentos removidos)
            slug += "_x"
        used.add(slug)
        emb = np.load(emb_dir / file).astype(np.float32).flatten()
        (out_dir / f"{slug}.json").write_text(
            json.dumps({"file": file, "embedding": [float(v) for v in emb]}, ensure_ascii=False), encoding="utf-8"
        )
        count += 1
    return count


def export_voice_embeddings(emb_dir: Path, out_dir: Path) -> int:
    """Um JSON por arquivo .npy de data/embeddings (inclusive os genericos spk_N/Person_N).

    O app repete no aparelho o mesmo carregamento do MultiSpeakerVerifier (agrupamento de homonimos, media dos
    embeddings, renomeacao de arquivos), entao precisa dos arquivos crus, nao de um resumo por pessoa.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    used: set[str] = set()
    count = 0
    for file in sorted(os.listdir(emb_dir)):
        if not file.endswith(".npy"):
            continue
        base = file[:-4]
        slug = slugify(base)
        while slug in used:  # colisao de slug (acentos removidos)
            slug += "_x"
        used.add(slug)
        emb = np.load(emb_dir / file).astype(np.float32).flatten()
        (out_dir / f"{slug}.json").write_text(
            json.dumps({"name": base, "embedding": [float(v) for v in emb]}, ensure_ascii=False), encoding="utf-8"
        )
        count += 1
    return count


def export_silero_vad(out: Path) -> None:
    """O av-tracker transcreve com vad_filter=True: o faster-whisper usa este modelo Silero (ONNX, ~1.2 MB)."""
    import faster_whisper

    src = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-tracker", type=Path, required=True, help="Pasta do projeto av-tracker")
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parent.parent / "src/main/assets")
    args = parser.parse_args()

    export_yolo_face(args.av_tracker / "od_model/yolov8n-face.pt", args.assets / "models/yolov8n_face.onnx")
    export_edgeface(args.av_tracker / "od_model/edgeface_xxs.pt", args.assets / "models/edgeface_xxs.onnx")
    n = export_face_embeddings(args.av_tracker / "data/face_embeddings", args.assets / "face_embeddings")
    print(f"Embeddings faciais exportados: {n} arquivos")
    export_ecapa(args.assets / "models/ecapa_voxceleb.onnx")
    before, after = shrink_fp16_storage(args.assets / "models/ecapa_voxceleb.onnx")
    print(f"ecapa_voxceleb.onnx: {before:.1f} MB -> {after:.1f} MB (fp16 storage)")
    export_light_asd(args.av_tracker, args.assets / "models/light_asd.onnx")
    export_segmentation(args.assets / "models/pyannote_segmentation.onnx")
    export_silero_vad(args.assets / "models/silero_vad_v6.onnx")
    from export_wespeaker import export as export_wespeaker

    export_wespeaker(args.assets / "models/wespeaker_resnet34.onnx", shrink=True)
    n = export_voice_embeddings(args.av_tracker / "data/embeddings", args.assets / "voice_embeddings")
    print(f"Embeddings de voz exportados: {n} arquivos")


if __name__ == "__main__":
    main()
