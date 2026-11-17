"""OCR backend configuration and refusal checks; no real inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from elevator import ocr as backend
class BackendChecks(unittest.TestCase):
    def test_eighty_percent_threshold_filters_variant_scores_without_hiding_conflicts(self):
        from elevator.perception import choose_floor
        self.assertEqual(choose_floor([('3',.8),('3',.8)],.8),('3',.8))
        self.assertEqual(choose_floor([('3',.799),('3',.799)],.8),(None,None))
        self.assertEqual(choose_floor([('3',.81),('3',.82),('8',.80)],.8),(None,None))

    def test_invalid_arguments_fail_before_backend_construction(self):
        for kwargs in ({"device": "gpu"}, {"threads": 0}, {"threads": True},
                       {"gpu_index": -1}, {"gpu_index": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend.build_ocr(**kwargs)

    def test_explicit_cuda_failure_never_silently_falls_back(self):
        with patch.object(backend, "_build_cuda", side_effect=RuntimeError("kernel failed")), \
                patch.object(backend, "_build_cpu") as cpu:
            with self.assertRaisesRegex(RuntimeError, "未回退 CPU"):
                backend.build_ocr(device="cuda")
            cpu.assert_not_called()

    def test_auto_fallback_reports_real_failure_reason(self):
        fake = SimpleNamespace(text_rec=object())
        with patch.object(backend, "_build_cuda", side_effect=FileNotFoundError("missing pth")), \
                patch.object(backend, "_build_cpu", return_value=(fake, {"actual_backend": "onnxruntime_cpu", "actual_device": "cpu"})), \
                warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            engine, info = backend.build_ocr(device="auto")
        self.assertIs(engine, fake)
        self.assertEqual(info["actual_device"], "cpu")
        self.assertIn("missing pth", info["reason"])
        self.assertTrue(any("auto 选择 CPU" in str(item.message) for item in captured))

    def test_explicit_cpu_does_not_probe_or_load_cuda(self):
        fake = SimpleNamespace(text_rec=object())
        with patch.object(backend, "_build_cuda", side_effect=AssertionError("CPU path probed CUDA")), \
                patch.object(backend, "_build_cpu", return_value=(fake, {"actual_device": "cpu"})):
            engine, info = backend.build_ocr(device="cpu")
        self.assertIs(engine, fake)
        self.assertEqual(info["requested_device"], "cpu")
        self.assertTrue(info["recognition_only"])

    def test_missing_model_fails_without_downloading(self):
        from rapidocr.utils.download_file import DownloadFile
        with patch.object(Path, "is_file", return_value=False), \
                patch.object(DownloadFile, "run", side_effect=AssertionError("unexpected model download")):
            with self.assertRaisesRegex(FileNotFoundError, "未自动下载"):
                backend._require_file(ROOT / "missing_recognizer.pth")

    def test_hash_mismatch_is_rejected_before_model_load(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(backend, "_sha256", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "SHA256 不匹配"):
                backend._require_file(ROOT / "example.pth", "expected")

    def test_download_plan_is_fixed_and_separate_from_original_weights(self):
        plan = backend.model_download_plan()
        self.assertFalse(plan["automatic_download"])
        self.assertIn("/resolve/v3.9.2/", plan["model"]["url"])
        self.assertEqual(Path(plan["model"]["path"]).parent, ROOT / "models/261104_my_rapidocr_torch")
        self.assertEqual(plan["model"]["sha256"], backend.TORCH_MODEL_SHA256)

    def test_real_cpu_recognizer_has_only_verified_cpu_session_and_no_download(self):
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.download_file import DownloadFile
        with patch.object(DownloadFile, "run", side_effect=AssertionError("unexpected model download")):
            engine, info = backend.build_ocr(device="cpu")
            output = engine.text_rec(TextRecInput(img=[np.full((48, 96, 3), 255, np.uint8)]))
        self.assertEqual(info["actual_device"], "cpu")
        self.assertEqual(info["providers"], ["CPUExecutionProvider"])
        self.assertEqual(info["enabled_stages"], ["rec"])
        self.assertFalse(hasattr(engine, "text_det"))
        self.assertFalse(hasattr(engine, "text_cls"))
        self.assertEqual(len(output.txts), 1)
        self.assertEqual(len(output.scores), 1)
