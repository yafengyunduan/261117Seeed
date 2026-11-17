"""Current YOLOE + ring geometry + CUDA/CPU OCR. No robot access."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any, Optional
import cv2
import numpy as np
from utils.camera_utils import load_config
from utils.yolo_utils import detect_objects, load_yolo, resolve_yolo_model_path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

os.environ["YOLO_AUTOINSTALL"] = "false"
from . import ocr as backend

BBox = tuple[int, int, int, int]
@dataclass(frozen=True)
class Ring:
    x: float
    y: float
    radius: float
    edge_support: float

    @property
    def center(self) -> tuple[int, int]:
        return round(self.x), round(self.y)

@dataclass
class ButtonObservation:
    ring: Ring
    text: Optional[str] = None
    confidence: Optional[float] = None
    text_bbox: Optional[BBox] = None
    variants: list[tuple[str, float]] = field(default_factory=list)
    status: str = "NO_TEXT"
    observations: int = 0
    text_region_source: str = "bright_glyph_components"

    @property
    def text_center_px(self) -> Optional[tuple[int, int]]:
        if self.text is None or self.text_bbox is None:
            return None
        x1, y1, x2, y2 = self.text_bbox
        return round((x1 + x2) / 2), round((y1 + y2) / 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "button_candidate_center_px": self.ring.center,
            "radius_px": round(self.ring.radius, 2),
            "ring_edge_support": round(self.ring.edge_support, 3),
            "text": self.text,
            "confidence": self.confidence,
            "text_bbox_xyxy": self.text_bbox if self.text is not None else None,
            "text_center_px": self.text_center_px,
            "text_region_source": self.text_region_source if self.text_center_px else None,
            "status": self.status,
            "consistent_observations": self.observations,
            "ocr_variants": [{"text": t, "confidence": s} for t, s in self.variants],
        }

@dataclass
class _FrameResult:
    captured_at: float
    image_shape: tuple[int, int]
    panel_bbox: Optional[BBox] = None
    panel_source: str = "NO_PANEL"
    buttons: list[ButtonObservation] = field(default_factory=list)
    yolo_detections: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_shape_hw": self.image_shape,
            "panel_source": self.panel_source,
            "panel_bbox_xyxy": self.panel_bbox,
            "button_candidate_count": len(self.buttons),
            "recognized_numbers": [b.text for b in self.buttons if b.text is not None],
            "buttons": [b.to_dict() for b in self.buttons],
            "yolo_detections": self.yolo_detections,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "errors": self.errors,
            "press_target_validated": False,
        }

def clip_box(box: Any, shape: tuple[int, ...]) -> Optional[BBox]:
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        return None
    h, w = shape[:2]
    x1, y1 = np.floor(values[:2]).astype(int)
    x2, y2 = np.ceil(values[2:]).astype(int)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return (int(x1), int(y1), int(x2), int(y2)) if x2 > x1 and y2 > y1 else None

def contains(box: BBox, ring: Ring) -> bool:
    return box[0] <= ring.x < box[2] and box[1] <= ring.y < box[3]

def find_rings(image: np.ndarray, max_side: int = 1200) -> list[Ring]:
    """Find rims, verify angular edge coverage and suppress nested duplicates.

    This inexpensive search may cover the frame; OCR never does. Coordinates
    are mapped back with the actual independent x/y resize factors.
    """
    h, w = image.shape[:2]
    if min(h, w) < 24:
        return []
    scale = min(1.0, max_side / max(h, w))
    ww, wh = max(1, round(w * scale)), max(1, round(h * scale))
    work = cv2.resize(image, (ww, wh), interpolation=cv2.INTER_AREA)
    sx, sy = ww / w, wh / h
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)
    short = min(ww, wh)
    min_radius = max(6, round(short * 0.016))
    max_radius = max(min_radius + 2, round(short * 0.115))
    raw = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(10, round(short * 0.033)),
        param1=100, param2=min(38, max(18, round(short * 0.042))),
        minRadius=min_radius, maxRadius=max_radius,
    )
    if raw is None:
        return []
    edges = cv2.Canny(blur, 70, 160)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    radial = np.linspace(0.94, 1.06, 7)[:, None]
    candidates = []
    for x, y, radius in raw[0]:
        xs = np.clip(np.rint(x + np.cos(theta) * radius * radial).astype(int), 0, ww - 1)
        ys = np.clip(np.rint(y + np.sin(theta) * radius * radial).astype(int), 0, wh - 1)
        support = float(np.mean(distance[ys, xs].min(axis=0) < 1.6))
        if support >= 0.85:
            candidates.append(Ring(float(x / sx), float(y / sy), float(radius / math.sqrt(sx * sy)), support))
    kept: list[Ring] = []
    for ring in sorted(candidates, key=lambda c: -c.radius):
        if not any(math.hypot(ring.x - other.x, ring.y - other.y) < max(ring.radius, other.radius) for other in kept):
            kept.append(ring)
    return sorted(kept, key=lambda c: (c.y, c.x))

def repeated_button_group(rings: list[Ring]) -> list[Ring]:
    """Require repeated sizes and row/column alignment, without floor-layout priors."""
    adjacency: list[list[int]] = [[] for _ in rings]
    for i, first in enumerate(rings):
        for j in range(i):
            second = rings[j]
            radius = (first.radius + second.radius) / 2
            dx, dy = abs(first.x - second.x), abs(first.y - second.y)
            if (min(first.radius, second.radius) / max(first.radius, second.radius) >= 0.72
                    and 1.8 * radius <= math.hypot(dx, dy) <= 5.6 * radius
                    and min(dx, dy) <= 0.7 * radius):
                adjacency[i].append(j)
                adjacency[j].append(i)
    components = []
    unseen = set(range(len(rings)))
    while unseen:
        todo, component = [unseen.pop()], []
        while todo:
            node = todo.pop()
            component.append(rings[node])
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    todo.append(neighbor)
        components.append(component)
    strong = [c for c in components if len(c) >= 4 and
              max(r.y for r in c) - min(r.y for r in c) >= 4 * float(np.median([r.radius for r in c]))]
    if not strong:
        return []
    # Text-sized circles on a sign should not beat the repeated larger rims.
    main = max(strong, key=lambda c: len(c) * float(np.median([r.radius for r in c])))
    typical_radius = float(np.median([r.radius for r in main]))
    selected = list(main)
    for component in components:
        if component is main or len(component) < 3:
            continue
        size_ratio = float(np.median([r.radius for r in component])) / typical_radius
        gap = min(math.hypot(a.x - b.x, a.y - b.y) for a in main for b in component)
        if 0.72 <= size_ratio <= 1.35 and gap <= 6.5 * typical_radius:
            selected.extend(component)
    return sorted(selected, key=lambda c: (c.y, c.x))

def group_box(rings: list[Ring], shape: tuple[int, ...]) -> Optional[BBox]:
    if not rings:
        return None
    return clip_box((min(r.x - 1.35 * r.radius for r in rings),
                     min(r.y - 1.35 * r.radius for r in rings),
                     max(r.x + 1.35 * r.radius for r in rings),
                     max(r.y + 1.35 * r.radius for r in rings)), shape)

def crop(image: np.ndarray, box: Any) -> tuple[Optional[np.ndarray], Optional[BBox]]:
    clipped = clip_box(box, image.shape)
    if clipped is None:
        return None, None
    x1, y1, x2, y2 = clipped
    return image[y1:y2, x1:x2].copy(), clipped

def glyph_variants(image: np.ndarray, ring: Ring) -> tuple[list[np.ndarray], Optional[BBox]]:
    """Localize bright raised glyphs separately from the rim and braille.

    Recognition-only OCR does not return text boxes. The reported text box is
    explicitly estimated from these image components, with its own offset.
    """
    x, y, r = ring.x, ring.y, ring.radius
    base, _ = crop(image, (x - .58*r, y - .66*r, x + .58*r, y + .38*r))
    region, offset_box = crop(image, (x - .66*r, y - .85*r, x + .66*r, y + .50*r))
    if base is None or region is None:
        return [], None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if int(gray.max()) - int(gray.min()) < 15:
        return [base], None
    median = float(np.median(gray))
    threshold = max(float(np.percentile(gray, 90)), median + .35*(int(gray.max())-median))
    mask = np.uint8(gray >= threshold) * 255
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    chosen = []
    for index, (bx, by, bw, bh, area) in enumerate(stats[1:], 1):
        if (.4*r < bh < 1.35*r and area > .015*r*r and bx > 0 and by > 0
                and bx+bw < region.shape[1] and by+bh < region.shape[0]):
            chosen.append(index)
    if not chosen:
        return [base], None
    main = max(chosen, key=lambda k: stats[k, 4])
    cx, cy = centers[main]
    # Include an actual disconnected minus stroke; never infer a sign by row.
    for index, (bx, by, bw, bh, area) in enumerate(stats[1:], 1):
        if (index not in chosen and bw > .22*r and bh < .23*r and area > .008*r*r
                and abs(centers[index, 1] - cy) < .3*r
                and abs(centers[index, 0] - cx) < .85*r):
            chosen.append(index)
    bx, by = min(stats[k, 0] for k in chosen), min(stats[k, 1] for k in chosen)
    ex = max(stats[k, 0] + stats[k, 2] for k in chosen)
    ey = max(stats[k, 1] + stats[k, 3] for k in chosen)
    binary = np.uint8(~np.isin(labels, chosen)) * 255
    tight = region[by:ey, bx:ex]
    binary = cv2.cvtColor(binary[by:ey, bx:ex], cv2.COLOR_GRAY2BGR)
    ox, oy = offset_box[:2]
    return [base, tight, binary], (int(ox+bx), int(oy+by), int(ox+ex), int(oy+ey))

def normalize_floor(text: str) -> Optional[str]:
    value = unicodedata.normalize("NFKC", text).strip().replace(" ", "")
    value = value.replace("−", "-").replace("–", "-")
    # A substring in a phone number or an I/H glyph must not become floor 3/1.
    return value if re.fullmatch(r"-?[0-9]{1,2}", value) else None

def choose_floor(variants: list[tuple[str, float]], min_score: float) -> tuple[Optional[str], Optional[float]]:
    valid = [(normalize_floor(t), float(s)) for t, s in variants
             if np.isfinite(s) and s >= min_score and normalize_floor(t) is not None]
    counts = Counter(text for text, _ in valid)
    supported = [text for text, count in counts.items() if count >= 2]
    if len(supported) != 1:
        return None, None
    label = supported[0]
    # Another high-confidence numeric reading is an ambiguity, not a vote to hide.
    if any(text != label for text, _ in valid):
        return None, None
    return label, float(np.mean([score for text, score in valid if text == label]))

class _FrameRecognition:
    """Reuse identical padded pixels within ONE frame, including bright/dark branches.

    Pixel bytes (plus shape/dtype) are exact keys, not perceptual similarity.
    Logical view order/voting is preserved. Nothing survives into the next frame.
    """
    def __init__(self, engine, input_type):
        self.engine, self.input_type = engine, input_type
        self.cache = {}
        self.stats = dict(requested_patches=0, inferred_patches=0, reused_patches=0, batches=0)

    def recognize(self, patches):
        keys, pending, unique = [], [], {}
        for patch in patches:
            key = (patch.shape, patch.dtype.str, patch.tobytes())
            keys.append(key)
            if key not in self.cache and key not in unique:
                unique[key] = len(pending)
                pending.append(patch)
        self.stats["requested_patches"] += len(keys)
        self.stats["reused_patches"] += len(keys)-len(pending)
        if pending:
            raw = self.engine.text_rec(self.input_type(img=pending))
            if (raw.txts is None or raw.scores is None or len(raw.txts) != len(pending)
                    or len(raw.scores) != len(pending)):
                raise RuntimeError("Unexpected RapidOCR recognition batch structure")
            values = [(str(text), float(score)) for text, score in zip(raw.txts, raw.scores)]
            self.cache.update(zip(unique, values))
            self.stats["inferred_patches"] += len(pending)
            self.stats["batches"] += 1
        return [self.cache[key] for key in keys]


class _OCRBatch:
    def recognize(self, image: np.ndarray, rings: list[Ring], panel_box: BBox,
                  *, _frame=None) -> list[ButtonObservation]:
        frame = _frame if _frame is not None else _FrameRecognition(self.engine, self.input_type)
        self.last_inference_stats = frame.stats
        self.last_patch_candidates = 0
        panel_image, panel_box = crop(image, panel_box)
        if panel_image is None:
            return []
        ox, oy = panel_box[:2]
        observations, patches, owners, retries = [], [], [], {}
        for index, ring in enumerate(rings):
            local_ring = Ring(ring.x-ox, ring.y-oy, ring.radius, ring.edge_support)
            variants, glyph_box = glyph_variants(panel_image, local_ring)
            if glyph_box is not None:
                x1, y1, x2, y2 = glyph_box
                glyph_box = (x1+ox, y1+oy, x2+ox, y2+oy)
            observations.append(ButtonObservation(ring, text_bbox=glyph_box))
            if glyph_box is None:
                continue  # Extract once within the panel; no glyph, no OCR work.
            self.last_patch_candidates += 1
            # Start with the independent original/binary views. Only retry the
            # tight color view if a plausible number still lacks agreement.
            if len(variants) == 3:
                retries[index] = variants[1]
                variants = [variants[0], variants[2]]
            for patch in variants:
                border = max(3, round(patch.shape[0] * .15))
                patches.append(cv2.copyMakeBorder(patch, border, border, border, border,
                                                  cv2.BORDER_CONSTANT, value=(255, 255, 255)))
                owners.append(index)
        if patches:
            for owner, reading in zip(owners, frame.recognize(patches)):
                observations[owner].variants.append(reading)
        retry_patches, retry_owners = [], []
        for index, patch in retries.items():
            views = observations[index].variants
            label, _ = choose_floor(views, self.min_score)
            if label is None and any(normalize_floor(t) is not None and s >= self.min_score for t, s in views):
                border = max(3, round(patch.shape[0] * .15))
                retry_patches.append(cv2.copyMakeBorder(patch, border, border, border, border,
                                                        cv2.BORDER_CONSTANT, value=(255, 255, 255)))
                retry_owners.append(index)
        if retry_patches:
            for owner, reading in zip(retry_owners, frame.recognize(retry_patches)):
                observations[owner].variants.append(reading)
        for observation in observations:
            if observation.text_bbox is None:
                observation.status = "NO_GLYPH"
                continue
            observation.text, observation.confidence = choose_floor(observation.variants, self.min_score)
            observation.status = "UNSTABLE" if observation.text is not None else "UNKNOWN"
        return observations

def _draw_boxes(image: np.ndarray, result: Optional[FrameResult], fps: float = 0, offline: bool = False,
                max_age: float = 4.0, targets: tuple[str, ...] = ("3", "4")) -> np.ndarray:
    display = image.copy()
    age = 0 if offline or result is None else max(0, time.perf_counter() - result.captured_at)
    valid = result is not None and age <= max_age
    if valid:
        if result.panel_bbox is not None:
            x1, y1, x2, y2 = result.panel_bbox
            color = (0, 220, 0) if result.panel_source == "YOLOE" else (0, 210, 255)
            cv2.rectangle(display, (x1, y1), (x2-1, y2-1), color, 2)
        for observation in result.buttons:
            ring = observation.ring
            known = observation.text is not None
            stable = observation.status == "STABLE"
            color = (0, 230, 0) if stable else ((0, 180, 255) if known else (140, 140, 140))
            thickness = 3 if observation.text in targets else 2
            cv2.circle(display, ring.center, round(ring.radius), color, thickness, cv2.LINE_AA)
            cv2.drawMarker(display, ring.center, (255, 80, 255), cv2.MARKER_CROSS, 9, 1)
            if known and observation.text_bbox:
                bx, by, ex, ey = observation.text_bbox
                cv2.rectangle(display, (bx, by), (ex-1, ey-1), (255, 180, 0), 2)
                cv2.circle(display, observation.text_center_px, 3, (0, 0, 255), -1)
            label = f"{observation.text} {observation.confidence:.1%}" if known else "?"
            cv2.putText(display, label, (max(0, round(ring.x-ring.radius)), max(18, round(ring.y-ring.radius-8))),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        recognized = ",".join(b.text for b in result.buttons if b.text is not None) or "NO_TEXT"
        summary = f"panel={result.panel_source} rings={len(result.buttons)} numbers={recognized}"
        timing = f"OCR {result.timings_ms.get('ocr', 0):.0f} ms / total {result.timings_ms.get('total', 0):.0f} ms"
    else:
        summary, timing = ("EXPIRED" if result else "WAITING") + " / no current coordinates", ""
    lines = [summary, timing, "OFFLINE SINGLE_SHOT" if offline else f"display FPS {fps:.1f} / result age {age:.2f}s",
             "cyan=text box; magenta=rim candidate center; visual proposals only"]
    if valid:
        for observation in result.buttons:
            if observation.text in targets:
                lines.append(f"{observation.text}: text_center_px={observation.text_center_px} "
                             f"rim_candidate={observation.ring.center} "
                             f"{'SINGLE_SHOT' if offline else observation.status}")
    if valid and result.errors:
        lines.append(result.errors[0][:110])
    for i, line in enumerate(lines):
        cv2.putText(display, line, (10, 24+22*i), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, 24+22*i), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    return display

@dataclass
class FrameResult(_FrameResult):
    tracking: dict[str, Any] = field(default_factory=dict)
    ocr_backend: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = super().to_dict()
        record.update(tracking=self.tracking, ocr_backend=self.ocr_backend,
                      labels_source="CURRENT_OCR_ONLY")
        return record

class ButtonOCR(_OCRBatch):
    """One recognition-only backend; preserve verified legacy batch voting."""
    def __init__(self, threads: int = 4, min_score: float = .80,
                 device: str = "auto", gpu_index: int = 0, model_dir: Optional[Path] = None):
        from rapidocr.ch_ppocr_rec import TextRecInput
        self.input_type = TextRecInput
        self.engine, self.backend_info = backend.build_ocr(threads=threads, device=device,
                                                           gpu_index=gpu_index, model_dir=model_dir)
        self.min_score = min_score
        self.last_patch_candidates = 0
        self.last_inference_stats = {}

def dark_glyph_variants(image, ring):
    """与旧亮字提取互补：Otsu 深色连通域，拒绝边缘背景与矮小盲文。

    不缩放推理坐标；返回字形估计框，不能当作 OCR 检测器输出或按钮中心。
    """
    x, y, r = ring.x, ring.y, ring.radius
    region, offset = crop(image, (x-.66*r, y-.85*r, x+.66*r, y+.50*r))
    base, _ = crop(image, (x-.58*r, y-.66*r, x+.58*r, y+.38*r))
    if region is None or base is None:
        return [], None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if int(gray.max())-int(gray.min()) < 15:
        return [], None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    chosen = [i for i, (bx, by, bw, bh, area) in enumerate(stats[1:], 1)
              if .4*r < bh < 1.35*r and area > .015*r*r and bx > 0 and by > 0
              and bx+bw < gray.shape[1] and by+bh < gray.shape[0]]
    if not chosen:
        return [], None
    # 保留实际存在的负号，不依据楼层排列推断符号或数字。
    main = max(chosen, key=lambda k: stats[k, 4])
    cx, cy = centers[main]
    for i, (bx, by, bw, bh, area) in enumerate(stats[1:], 1):
        if (i not in chosen and bw > .22*r and bh < .23*r and area > .008*r*r
                and abs(centers[i, 1]-cy) < .3*r and abs(centers[i, 0]-cx) < .85*r):
            chosen.append(i)
    bx, by = np.min(stats[chosen, :2], axis=0)
    ex, ey = np.max(stats[chosen, :2]+stats[chosen, 2:4], axis=0)
    binary = np.uint8(~np.isin(labels, chosen))*255
    patches = [base, region[by:ey, bx:ex],
               cv2.cvtColor(binary[by:ey, bx:ex], cv2.COLOR_GRAY2BGR)]
    ox, oy = offset[:2]
    return patches, (int(ox+bx), int(oy+by), int(ox+ex), int(oy+ey))


def dark_vote(variants, threshold):
    """沿用两视图一致规则，且原始色彩图必须支持同一数字。

    防止仅靠裁剪/二值化将图标内部的一笔误认为数字。
    三视图是同一张照片的预处理，不是三个独立帧的稳定性证据。
    """
    label, confidence = choose_floor(variants, threshold)
    if (label is None or not variants or not np.isfinite(variants[0][1])
            or variants[0][1] < threshold or normalize_floor(variants[0][0]) != label):
        return None, None
    return label, confidence


class GeneralizedOCR:
    """亮字优先、深色字形补充；共享一个识别器，实时稳定性交给时序跟踪器。"""
    def __init__(self, original, *, offline=False):
        self.original = original
        self.offline = offline
        self.diagnostics = []
        self.last_patch_candidates = 0
        self.last_inference_stats = {}

    @property
    def backend_info(self):
        return self.original.backend_info

    def recognize(self, image, rings, panel_box):
        self.diagnostics = []
        self.last_patch_candidates = 0
        frame = _FrameRecognition(self.original.engine, self.original.input_type)
        self.last_inference_stats = frame.stats
        if isinstance(self.original, _OCRBatch):
            observations = list(self.original.recognize(image, rings, panel_box, _frame=frame))
        else:
            observations = list(self.original.recognize(image, rings, panel_box))
        if len(observations) != len(rings):
            raise RuntimeError("OCR returned a different number of current ring candidates")
        self.diagnostics = [{"source": "bright_glyph_components",
                             "bright": b.to_dict(), "dark_variants": []} for b in observations]
        panel, panel_box = crop(image, panel_box)
        if panel is None:
            return []
        ox, oy = panel_box[:2]
        patches, owners, boxes, retries = [], [], {}, {}
        for i, b in enumerate(observations):
            if b.text is not None:
                continue  # 保留已被旧路径接受的结果。
            r = b.ring
            variants, box = dark_glyph_variants(panel, replace(r, x=r.x-ox, y=r.y-oy))
            if box is None:
                continue
            boxes[i] = (box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy)
            # Same staged strategy as bright OCR: original + binary first.
            # Tight color is a fallback, not another mandatory model forward.
            retries[i] = variants[1]
            for patch in (variants[0], variants[2]):
                pad = max(3, round(patch.shape[0]*.15))
                patches.append(cv2.copyMakeBorder(patch, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255)))
                owners.append(i)
        self.last_patch_candidates = sum(b.text_bbox is not None or i in boxes
                                         for i, b in enumerate(observations))
        if patches:
            for i, reading in zip(owners, frame.recognize(patches)):
                self.diagnostics[i]["dark_variants"].append(reading)
        retry_patches, retry_owners = [], []
        for i, patch in retries.items():
            views = self.diagnostics[i]["dark_variants"]
            first, score = views[0]
            label = normalize_floor(first)
            # The original color crop must support the digit. A known numeric
            # conflict cannot be repaired by a third vote and stays rejected.
            original_valid = label is not None and np.isfinite(score) and score >= self.original.min_score
            conflict = any(np.isfinite(s) and s >= self.original.min_score
                           and normalize_floor(t) not in (None, label) for t, s in views)
            if original_valid and not conflict and dark_vote(views, self.original.min_score)[0] is None:
                pad = max(3, round(patch.shape[0]*.15))
                retry_patches.append(cv2.copyMakeBorder(patch, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255)))
                retry_owners.append(i)
        if retry_patches:
            for i, reading in zip(retry_owners, frame.recognize(retry_patches)):
                self.diagnostics[i]["dark_variants"].append(reading)
        for i, box in boxes.items():
            variants = self.diagnostics[i]["dark_variants"]
            label, confidence = dark_vote(variants, self.original.min_score)
            # 不隐藏旧路径里与新数字冲突的高置信度数字证据。
            conflict = label is not None and any(
                s >= self.original.min_score and np.isfinite(s)
                and normalize_floor(t) not in (None, label)
                for t, s in observations[i].variants)
            if label is not None and not conflict:
                observations[i] = ButtonObservation(observations[i].ring, label,
                    confidence, box, variants, status="UNSTABLE", text_region_source="dark_glyph_components")
                self.diagnostics[i]["source"] = "dark_glyph_components"
            else:
                self.diagnostics[i]["dark_rejection"] = "BRIGHT_DARK_CONFLICT" if conflict else "NO_CONSENSUS"
        status = "SINGLE_SHOT" if self.offline else "UNSTABLE"
        return [replace(b, status=status) if b.text is not None else b for b in observations]


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def _tracking_gray(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]
    scale = min(1., 900/max(h, w))
    ww, hh = max(1, round(w*scale)), max(1, round(h*scale))
    work = image if (ww, hh) == (w, h) else cv2.resize(image, (ww, hh), interpolation=cv2.INTER_AREA)
    return _gray(work), ww/w, hh/h

def _edge_distance(gray: np.ndarray) -> np.ndarray:
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)
    return cv2.distanceTransform(255 - cv2.Canny(blur, 65, 150), cv2.DIST_L2, 3)

def verify_ring(ring: Ring, distance: np.ndarray, minimum: float = .72) -> Optional[Ring]:
    """Require current angular coverage; refine only near a tracked prediction."""
    h, w = distance.shape
    if ring.radius < 5 or ring.x-ring.radius < 1 or ring.y-ring.radius < 1 or ring.x+ring.radius >= w-1 or ring.y+ring.radius >= h-1:
        return None
    theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    cosine, sine = np.cos(theta), np.sin(theta)
    best = (0.0, ring.x, ring.y, ring.radius, np.zeros(96, dtype=bool))
    shift = max(.75, min(2.5, ring.radius*.025))
    for dx, dy in ((0., 0.), (-shift, 0.), (shift, 0.), (0., -shift), (0., shift)):
        for scale in (.97, 1., 1.03):
            radius = ring.radius*scale
            radial = radius*np.linspace(.94, 1.06, 5)[:, None]
            xs = np.clip(np.rint(ring.x+dx+radial*cosine).astype(int), 0, w-1)
            ys = np.clip(np.rint(ring.y+dy+radial*sine).astype(int), 0, h-1)
            supported = np.min(distance[ys, xs], axis=0) < max(1.7, .035*radius)
            score = float(np.mean(supported))
            if score > best[0]:
                best = score, ring.x+dx, ring.y+dy, radius, supported
    score, x, y, radius, supported = best
    if score < minimum or min(float(np.mean(part)) for part in np.array_split(supported, 4)) < .45:
        return None
    return Ring(float(x), float(y), float(radius), score)

class LayoutTracker:
    """A similarity transform of observed rims, supported by current pixels.

    Lost rims are never returned as buttons. A tracked ROI may preserve the
    transformed layout extent only while >=65% and >=4 real rims are verified.
    """
    def __init__(self, min_support_ratio: float = .65):
        self.min_support_ratio = min_support_ratio
        self.reset()

    def reset(self) -> None:
        self.previous = None
        self.input_shape = None
        self.rings: list[Ring] = []
        self.box = None

    def seed(self, image: np.ndarray, rings: list[Ring]) -> None:
        self.previous, _, _ = _tracking_gray(image)
        self.input_shape, self.rings = image.shape[:2], list(rings)
        self.box = group_box(rings, image.shape)

    def track(self, image: np.ndarray) -> tuple[list[Ring], Optional[BBox], dict[str, Any]]:
        info: dict[str, Any] = {"geometry_mode": "REACQUIRE", "current_ring_count": 0}
        gray, sx, sy = _tracking_gray(image)
        radius_scale = math.sqrt(sx*sy)
        if self.previous is None or self.input_shape != image.shape[:2] or self.previous.shape != gray.shape or len(self.rings) < 4 or self.box is None:
            return [], None, info
        mask = np.zeros(gray.shape, np.uint8)
        for r in self.rings:
            center = round(r.x*sx), round(r.y*sy)
            cv2.circle(mask, center, max(2, round(r.radius*radius_scale*1.18)), 255, -1)
            cv2.circle(mask, center, max(1, round(r.radius*radius_scale*.70)), 0, -1)
        first = cv2.goodFeaturesToTrack(self.previous, maxCorners=240, qualityLevel=.012,
                                       minDistance=4, mask=mask, blockSize=5)
        if first is None or len(first) < 8:
            return [], None, info
        second, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, first, None,
                                                       winSize=(25, 25), maxLevel=3)
        if second is None or status is None:
            return [], None, info
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, second, None,
                                                         winSize=(25, 25), maxLevel=3)
        if backward is None or back_status is None:
            return [], None, info
        valid = (status[:, 0] > 0) & (back_status[:, 0] > 0)
        valid &= np.linalg.norm(backward[:, 0]-first[:, 0], axis=1) < 1.5
        if error is not None:
            valid &= error[:, 0] < 28
        if int(valid.sum()) < 8:
            return [], None, info
        transform, inliers = cv2.estimateAffinePartial2D(first[valid, 0], second[valid, 0],
                                                        method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if transform is None or inliers is None or float(np.mean(inliers)) < .65:
            return [], None, info
        transform = np.diag([1/sx, 1/sy]) @ transform @ np.diag([sx, sy, 1.])
        scale = math.hypot(transform[0, 0], transform[1, 0])
        angle = abs(math.atan2(transform[1, 0], transform[0, 0]))
        if not .8 <= scale <= 1.25 or angle > math.radians(18):
            return [], None, info
        distance = _edge_distance(gray)
        predicted = []
        verified = []
        for ring in self.rings:
            center = transform @ np.array([ring.x, ring.y, 1.])
            prediction = Ring(float(center[0]), float(center[1]), ring.radius*scale, 0.)
            predicted.append(prediction)
            work_ring = Ring(prediction.x*sx, prediction.y*sy, prediction.radius*radius_scale, 0.)
            current = verify_ring(work_ring, distance)
            if current is not None:
                verified.append(Ring(current.x/sx, current.y/sy, current.radius/radius_scale, current.edge_support))
        ratio = len(verified)/len(self.rings)
        if len(verified) < 4 or ratio < self.min_support_ratio:
            return [], None, info
        # The full layout extent is registered using current features and rims.
        # It stabilizes the ROI when one outside rim is briefly occluded, while
        # that missing rim contributes no coordinates, glyph crop or text.
        box = group_box(predicted, image.shape)
        if box is None:
            return [], None, info
        self.previous, self.rings, self.box = gray, predicted, box
        info.update(geometry_mode="TRACKED_CURRENT_EDGES", current_ring_count=len(verified),
                    layout_ring_count=len(predicted), current_support_ratio=ratio,
                    flow_inliers=int(inliers.sum()), scale=float(scale))
        return sorted(verified, key=lambda r: (r.y, r.x)), box, info

class PanelPerception(object):
    """Reuse legacy geometry/OCR, adding a current-evidence layout tracker."""
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace):
        # Standalone legacy image checks keep their bright-only baseline. Both
        # application entries explicitly pass the workflow's preprocessing mode.
        self.ocr_preprocessing = getattr(args, "ocr_preprocessing", "bright")
        if self.ocr_preprocessing not in ("bright", "bright_dark"):
            raise ValueError("ocr_preprocessing需为bright或bright_dark")
        # Legacy __init__ hardcodes its CPU OCR class. Injecting this backend
        # directly avoids constructing two engines or changing the old module.
        self.roi = getattr(args, "roi", None)
        self.max_side = int(getattr(args, "max_side", 1200))
        self.model, self.yolo_options = None, {}
        self.yolo_interval = max(1, int(cfg.get("grasp_pipeline", {}).get("infer_every_live", 3)))
        self.calls, self.hough_calls = 0, 0
        self.refresh_every = max(2, int(getattr(args, "hough_refresh_every", 8)))
        self.layout = LayoutTracker()
        threads = int(getattr(args, "ocr_threads", 4))
        if not getattr(args, "geometry_only", False):
            model_path = resolve_yolo_model_path(cfg.get("yolo", {}).get("model_name", ""), ROOT)
            if not model_path.is_file():
                raise FileNotFoundError(f"Existing YOLO weights missing: {model_path}")
            import torch
            torch.set_num_threads(threads)
            self.model, self.yolo_options = load_yolo(cfg, project_root=ROOT,
                                                       device_override=getattr(args, "device", None))
        self.ocr = ButtonOCR(threads, float(getattr(args, "ocr_min_score", .80)),
                             str(getattr(args, "ocr_device", "auto")), int(getattr(args, "ocr_gpu_index", 0)),
                             getattr(args, "ocr_model_dir", None))
        if self.ocr_preprocessing == "bright_dark":
            self.ocr = GeneralizedOCR(self.ocr)
        self.backend_info = self.ocr.backend_info
        cv2.setNumThreads(min(4, threads))

    def warmup(self, height=720, width=1280):
        """Prepare YOLO CUDA kernels before the first real observation.

        This blank input supplies no evidence and never updates the tracker,
        OCR observations or frame counters. OCR already warms up on creation.
        """
        if self.model is not None:
            detect_objects(self.model,np.zeros((int(height),int(width),3),np.uint8),self.yolo_options)

    def process(self, image: np.ndarray, captured_at: Optional[float] = None, offline: bool = False) -> FrameResult:
        tick = time.perf_counter()
        at = tick if captured_at is None else float(captured_at)
        result = FrameResult(at, image.shape[:2], ocr_backend=dict(self.backend_info))
        self.calls += 1
        if offline:
            self.layout.reset()  # Independent photos are never a temporal sequence.
        yolo = []
        if self.model is not None and (offline or self.calls == 1 or self.calls % self.yolo_interval == 0):
            start = time.perf_counter()
            try:
                _, detections = detect_objects(self.model, image, self.yolo_options)
                yolo = [d for d in detections if "panel" in d.class_name.lower()]
                result.yolo_detections = [{"class_name": d.class_name, "confidence": float(d.conf),
                                           "bbox_xyxy": d.bbox_xyxy, "evidence": "CURRENT_YOLO"} for d in yolo]
            except Exception as exc:
                result.tracking["yolo_error"] = f"{type(exc).__name__}: {exc}"
            result.timings_ms["yolo"] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        rings, box, evidence = self.layout.track(image)
        tracking_valid = box is not None
        refresh = offline or not tracking_valid or self.calls % self.refresh_every == 0
        if refresh:
            self.hough_calls += 1
            proposals = repeated_button_group(find_rings(image, self.max_side))
            manual = clip_box(self.roi, image.shape) if self.roi is not None else None
            if self.roi is not None:
                proposals = [r for r in proposals if manual is not None and contains(manual, r)]
            if not tracking_valid or len(proposals) >= len(self.layout.rings)+2:
                rings = proposals
                box = group_box(rings, image.shape)
                evidence = {"geometry_mode": "CURRENT_HOUGH", "current_ring_count": len(rings),
                            "layout_ring_count": len(rings), "current_support_ratio": 1. if rings else 0.}
                if box is not None:
                    self.layout.seed(image, rings)
                else:
                    self.layout.reset()
        result.tracking.update(evidence, hough_this_frame=refresh, total_hough_calls=self.hough_calls,
                               labels_source="CURRENT_OCR_ONLY", ocr_preprocessing=self.ocr_preprocessing)
        if box is not None and rings:
            result.panel_bbox, result.panel_source = box, "GEOMETRY_PROPOSAL"
            if any(sum(contains(d.bbox_xyxy, r) for r in rings) >= .65*len(rings) for d in yolo):
                result.panel_source = "YOLOE"
        result.timings_ms["geometry"] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        if result.panel_bbox is not None:
            try:
                result.buttons = self.ocr.recognize(image, rings, result.panel_bbox)
            except Exception as exc:
                result.errors.append(f"OCR_ERROR: {type(exc).__name__}: {exc}")
                result.buttons = [ButtonObservation(r, status="OCR_ERROR") for r in rings]
        result.tracking["ocr_ring_candidates"] = getattr(self.ocr, "last_patch_candidates", len(rings))
        result.tracking["ocr_inference"] = (dict(getattr(self.ocr, "last_inference_stats", {}))
                                             if result.panel_bbox is not None else {})
        result.timings_ms["ocr"] = (time.perf_counter()-start)*1000
        result.timings_ms["total"] = (time.perf_counter()-tick)*1000
        return result

def draw_result(image: np.ndarray, result: Optional[FrameResult], fps: float = 0, offline: bool = False,
                max_age: float = 4., targets: tuple[str, ...] = ("3", "4")) -> np.ndarray:
    if result is not None and result.tracking.get('workflow_evidence') and (offline or time.perf_counter()-result.captured_at<=max_age):
        from .panel_261112 import draw_workflow_result
        return draw_workflow_result(image,result)
    display = _draw_boxes(image, result, fps, offline, max_age, targets)
    if result is not None:
        mode = result.tracking.get("geometry_mode", "NO_PANEL")
        backend = result.ocr_backend.get("actual_device", "unknown")
        line = f"{mode} | OCR device={backend} | CURRENT labels only; cached labels shown separately"
        y = display.shape[0]-12
        cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 3)
        cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
    return display

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT/"config/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ocr-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ocr-gpu-index", type=int, default=0)
    parser.add_argument("--ocr-model-dir", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--ocr-threads", type=int, default=4)
    parser.add_argument("--ocr-min-score", type=float, default=.80)
    parser.add_argument("--ocr-preprocessing", choices=("bright", "bright_dark"), default="bright")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--hough-refresh-every", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    args.roi = None
    try:
        perception = PanelPerception(load_config(args.config), args)
        for path in args.images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            result = perception.process(image, offline=True)
            print(json.dumps({"image": str(path), **result.to_dict()}, ensure_ascii=False), flush=True)
            if args.show:
                cv2.imshow("261104 offline image", draw_result(image, result, offline=True))
                if cv2.waitKey(0) & 0xFF in (27, ord("q"), ord("Q")):
                    break
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
