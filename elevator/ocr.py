"""RapidOCR 3.9.2 recognition-only CPU / existing PyTorch CUDA backend.

No package installation, model downloading or camera/robot access occurs here.
Only already present, explicitly named model files are loaded. The GPU path
uses RapidOCR's native TorchInferSession and verifies its real tensor devices
and a completed recognition warmup. Explicit ``cuda`` never falls back to CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Any, Optional
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models/261104_my_rapidocr_torch"
TORCH_MODEL_NAME = "PP-OCRv6_rec_small.pth"
TORCH_MODEL_SHA256 = "0107b2ad694ccc9b1db7cf9ed3ffbc93d1795d9e08d9cf823127243a87bce516"
CPU_MODEL_NAME = "PP-OCRv6_rec_small.onnx"
CPU_MODEL_SHA256 = "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
DICTIONARY_NAME = "ppocrv6_dict.txt"
# Recorded from the pinned official dictionary download; the upstream manifest
# provides the model SHA256 but does not publish a separate dictionary hash.
DICTIONARY_SHA256 = "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"
MODEL_URL = ("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
             "torch/PP-OCRv6/rec/PP-OCRv6_rec_small.pth")
DICTIONARY_URL = ("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
                  "paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt")


@dataclass
class RecognitionOnlyOCR:
    """Retain the existing ``engine.text_rec(TextRecInput(...))`` API."""
    text_rec: Any


def model_download_plan(model_dir: Optional[str | Path] = None) -> dict[str, Any]:
    destination = DEFAULT_MODEL_DIR if model_dir is None else Path(model_dir).expanduser()
    return {
        "automatic_download": False,
        "source_manifest": "RapidAI/RapidOCR v3.9.2 python/rapidocr/default_models.yaml",
        "model": {"url": MODEL_URL, "sha256": TORCH_MODEL_SHA256,
                  "path": str(destination / TORCH_MODEL_NAME)},
        "dictionary": {"url": DICTIONARY_URL, "path": str(destination / DICTIONARY_NAME),
                       "recorded_sha256": DICTIONARY_SHA256},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, expected_hash: Optional[str] = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"OCR 模型/字典缺失：{path}；未自动下载。下载计划：{json.dumps(model_download_plan(), ensure_ascii=False)}")
    actual = _sha256(path)
    if expected_hash is not None and actual != expected_hash:
        raise RuntimeError(f"OCR 文件 SHA256 不匹配：{path}；拒绝加载")
    return actual


def _recognizer_config(threads: int, backend: str, model: Path, gpu_index: int,
                       dictionary: Optional[Path] = None) -> Any:
    import rapidocr
    from rapidocr import EngineType
    from rapidocr.utils.parse_parameters import ParseParams
    cfg = ParseParams.load(Path(rapidocr.__file__).resolve().parent / "config.yaml")
    cfg = ParseParams.update_batch(cfg, {
        "Rec.engine_type": EngineType.TORCH if backend == "torch" else EngineType.ONNXRUNTIME,
        "Rec.model_path": str(model),
        "Rec.rec_batch_num": 12,
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.torch.use_cuda": backend == "torch",
        "EngineConfig.torch.cuda_ep_cfg.device_id": gpu_index,
        "EngineConfig.torch.use_mps": False,
        "EngineConfig.torch.use_npu": False,
    })
    # Pin the same model family used by the installed ONNX recognizer. This is
    # a local in-memory config; no package configuration file is modified.
    if cfg.Rec.ocr_version.value != "PP-OCRv6" or cfg.Rec.model_type.value != "small":
        raise RuntimeError("本适配器要求当前 RapidOCR PP-OCRv6 small 配置；未自动更换模型")
    cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
    cfg.Rec.model_root_dir = str(model.parent)
    cfg.Rec.font_path = None
    cfg.Rec.rec_keys_path = str(dictionary) if dictionary is not None else None
    return cfg.Rec


def _warmup(recognizer: Any) -> float:
    from rapidocr.ch_ppocr_rec import TextRecInput
    tick = time.perf_counter()
    output = recognizer(TextRecInput(img=[np.full((48, 96, 3), 255, np.uint8)]))
    if output.txts is None or output.scores is None or len(output.txts) != 1 or len(output.scores) != 1:
        raise RuntimeError("OCR warmup 未返回一条合法识别结果")
    return (time.perf_counter()-tick)*1000


def _build_cpu(threads: int) -> tuple[RecognitionOnlyOCR, dict[str, Any]]:
    import rapidocr
    from rapidocr.ch_ppocr_rec import TextRecognizer
    from rapidocr.utils.log import logger
    logger.setLevel("WARNING")
    model = Path(rapidocr.__file__).resolve().parent / "models" / CPU_MODEL_NAME
    model_hash = _require_file(model, CPU_MODEL_SHA256)
    rec = TextRecognizer(_recognizer_config(threads, "onnxruntime", model, 0))
    session = rec.session
    providers = list(session.session.get_providers())
    if type(session).__name__ != "OrtInferSession" or providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"CPU OCR session/provider 不符合请求：{type(session).__name__}/{providers}")
    warmup_ms = _warmup(rec)
    return RecognitionOnlyOCR(rec), {
        "actual_backend": "onnxruntime_cpu", "actual_device": "cpu",
        "actual_session_class": f"{type(session).__module__}.{type(session).__name__}",
        "providers": providers, "parameter_devices": [], "warmup_ms": warmup_ms,
        "model_path": str(model), "model_sha256": model_hash,
    }


def _build_cuda(threads: int, gpu_index: int, model_dir: Path) -> tuple[RecognitionOnlyOCR, dict[str, Any]]:
    import torch
    from rapidocr.ch_ppocr_rec import TextRecognizer
    from rapidocr.utils.log import logger
    logger.setLevel("WARNING")
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"当前进程无法使用 cuda:{gpu_index}；可见设备数 {torch.cuda.device_count()}")
    torch.set_num_threads(threads)
    model, dictionary = model_dir / TORCH_MODEL_NAME, model_dir / DICTIONARY_NAME
    model_hash = _require_file(model, TORCH_MODEL_SHA256)
    dictionary_hash = _require_file(dictionary, DICTIONARY_SHA256)
    lines = dictionary.read_text(encoding="utf-8").splitlines()
    # The model architecture emits blank + 18,708 dictionary entries + space.
    if len(lines) != 18708 or not all(lines):
        raise RuntimeError("PP-OCRv6 small 字典条目数不匹配；拒绝加载")
    rec = TextRecognizer(_recognizer_config(threads, "torch", model, gpu_index, dictionary))
    session = rec.session
    expected = f"cuda:{gpu_index}"
    if type(session).__name__ != "TorchInferSession" or str(session.device) != expected:
        raise RuntimeError(f"请求 CUDA 却获得 {type(session).__name__}/{getattr(session, 'device', None)}")
    tensors = list(session.predictor.parameters()) + list(session.predictor.buffers())
    devices = sorted({str(tensor.device) for tensor in tensors})
    if not tensors or devices != [expected]:
        raise RuntimeError(f"OCR 模型张量没有全部驻留在 {expected}：{devices}")
    # A real model forward proves CUDA kernels work for this architecture and
    # cuDNN build. Availability alone is insufficient.
    warmup_ms = _warmup(rec)
    torch.cuda.synchronize(gpu_index)
    return RecognitionOnlyOCR(rec), {
        "actual_backend": "torch_cuda", "actual_device": expected,
        "actual_session_class": f"{type(session).__module__}.{type(session).__name__}",
        "providers": [], "parameter_devices": devices, "warmup_ms": warmup_ms,
        "model_path": str(model), "model_sha256": model_hash,
        "dictionary_path": str(dictionary), "dictionary_sha256": dictionary_hash,
        "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(gpu_index),
        "cuda_model_forward_verified": True,
    }


def build_ocr(threads: int = 4, device: str = "auto", gpu_index: int = 0, *,
              model_dir: Optional[str | Path] = None) -> tuple[RecognitionOnlyOCR, dict[str, Any]]:
    """Initialize exactly one recognition backend and return verified metadata.

    ``auto`` tries the existing torch CUDA backend, then reports why CPU was
    selected if that attempt fails. ``cuda`` is strict and raises any failure.
    This function never downloads models or changes installed packages. Detection
    and orientation models are not created because callers already crop glyphs.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("OCR device 必须为 auto/cpu/cuda")
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 32:
        raise ValueError("threads 必须为 1 到 32 的整数")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise ValueError("gpu_index 必须为非负整数")
    if version("rapidocr") != "3.9.2":
        raise RuntimeError("此后端按 RapidOCR 3.9.2 验证，请先复核新版本接口；未自动安装或升级")
    reason = "明确选择 CPU"
    if device in {"auto", "cuda"}:
        try:
            engine, info = _build_cuda(threads, gpu_index,
                                       DEFAULT_MODEL_DIR if model_dir is None else Path(model_dir).expanduser())
            reason = "已核验原生 Torch CUDA session、模型张量及真实识别 forward"
        except Exception as exc:
            if device == "cuda":
                raise RuntimeError(f"OCR CUDA 初始化/推理核验失败：{type(exc).__name__}: {exc}；未回退 CPU") from exc
            reason = f"CUDA OCR 不可用，auto 选择 CPU：{type(exc).__name__}: {exc}"
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
            engine, info = _build_cpu(threads)
    else:
        engine, info = _build_cpu(threads)
    info.update(requested_device=device, gpu_index=gpu_index, reason=reason,
                recognition_only=True, enabled_stages=["rec"], rapidocr_version=version("rapidocr"))
    return engine, info
