"""Utilitarios para reduzir o tamanho dos modelos ONNX embutidos no APK."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def shrink_fp16_storage(path: Path, min_elems: int = 4096) -> tuple[float, float]:
    """Guarda pesos fp32 grandes como fp16 + um Cast para fp32.

    O ONNX Runtime faz constant-folding do Cast ao carregar, entao a computacao continua em fp32
    (sem perda de velocidade nem os erros da quantizacao int8, que degradou muito o encoder do
    Whisper); so o arquivo, e portanto o APK, fica com metade do tamanho.
    Retorna (MB antes, MB depois).
    """
    path = Path(path)
    before = path.stat().st_size / 1e6
    model = onnx.load(str(path))
    graph = model.graph

    kept, converted, casts = [], [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            arr = numpy_helper.to_array(init)
            half = arr.astype(np.float16)
            if np.isfinite(half).all():
                low_name = f"{init.name}__fp16"
                converted.append(numpy_helper.from_array(half, low_name))
                casts.append(helper.make_node("Cast", [low_name], [init.name], to=TensorProto.FLOAT, name=f"cast_{init.name}"))
                continue
        kept.append(init)

    del graph.initializer[:]
    graph.initializer.extend(kept + converted)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)  # Casts only read initializers, so they can safely come first
    onnx.save(model, str(path))
    return before, path.stat().st_size / 1e6
