"""独立图片实验：复用圆环检测与 RapidOCR，增加深色字形提取。

不修改现有电梯/串联流程，不连接相机或机器人。默认沿用 0.80 阈值。
仅保存指定图片的标注和 JSON；单张图识别不代表时间稳定或有效按压目标。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from elevator import perception as vision


def compact_group(rings):
    """剔除远离主按钮组的背景圆环；只筛已有候选，不补造缺失按钮。

    旧分组允许合并距离 6.5r 的另一组，这张原图把左侧木纹也合入。
    这里用原邻接规则的 5.6r 距离构建紧凑分量，不使用图片坐标/楼层模板。
    """
    unseen = set(range(len(rings)))
    components = []
    while unseen:
        todo, component = [unseen.pop()], []
        while todo:
            i = todo.pop()
            component.append(rings[i])
            a = rings[i]
            neighbors = [j for j in unseen
                         if min(a.radius, rings[j].radius)/max(a.radius, rings[j].radius) >= .72
                         and math.hypot(a.x-rings[j].x, a.y-rings[j].y)
                         <= 5.6*(a.radius+rings[j].radius)/2]
            unseen.difference_update(neighbors)
            todo.extend(neighbors)
        if len(component) >= 4:
            components.append(component)
    if not components:
        return []
    selected = max(components, key=lambda c: len(c)*float(np.median([r.radius for r in c])))
    return sorted(selected, key=lambda r: (r.y, r.x))


dark_glyph_variants = vision.dark_glyph_variants
dark_vote = vision.dark_vote


class GeneralizedOCR(vision.GeneralizedOCR):
    """图片实验保留 SINGLE_SHOT 标记；与实时入口复用同一字形算法。"""
    def __init__(self, original):
        super().__init__(original, offline=True)


def frame_record(image, rings, observations, duration_ms, backend_info, geometry_ms=0.):
    return vision.FrameResult(time.perf_counter(), image.shape[:2], vision.group_box(rings, image.shape),
        "GEOMETRY_PROPOSAL" if rings else "NO_PANEL", observations,
        timings_ms={"geometry": geometry_ms, "ocr": duration_ms, "total": geometry_ms+duration_ms},
        tracking={"geometry_mode": "OFFLINE_CURRENT_GEOMETRY" if rings else "NO_PANEL"},
        ocr_backend=backend_info)


def preview(image, result):
    """绘制在缩略图上，所有坐标统一缩放；报告始终用原图坐标。"""
    h, w = image.shape[:2]
    scale = min(1., 1400/max(h, w))
    ww, hh = round(w*scale), round(h*scale)
    sx, sy = ww/w, hh/h
    def box(b):
        return None if b is None else tuple(round(v*s) for v, s in zip(b, (sx, sy, sx, sy)))
    buttons = [replace(b, ring=replace(b.ring, x=b.ring.x*sx, y=b.ring.y*sy,
                                     radius=b.ring.radius*math.sqrt(sx*sy)), text_bbox=box(b.text_bbox))
               for b in result.buttons]
    scaled = replace(result, image_shape=(hh, ww), panel_bbox=box(result.panel_bbox), buttons=buttons)
    return vision.draw_result(cv2.resize(image, (ww, hh)), scaled, offline=True, targets=())


def score_arg(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("置信度必须在 (0, 1] 内")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--ocr-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--ocr-gpu-index", type=int, default=0)
    parser.add_argument("--ocr-min-score", type=score_arg, default=.80,
                        help="仅本实验阈值；不读取或修改现有工作流 YAML")
    parser.add_argument("--compare-thresholds", nargs="+", type=score_arg, default=[.80, .60],
                        help="原识别流程的实际重放阈值")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"readme/261110_panel_generalization")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if not 240 <= args.max_side <= 1600:
        parser.error("max-side 需在 240..1600 范围内")
    try:
        cv2.setNumThreads(4)
        original = vision.ButtonOCR(threads=4, min_score=args.ocr_min_score,
                                   device=args.ocr_device, gpu_index=args.ocr_gpu_index)
        experiment = GeneralizedOCR(original)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"experiment": "261110_panel_generalization", "offline_only": True,
                  "workflow_modified": False, "threshold": args.ocr_min_score,
                  "geometry_max_side": args.max_side, "ocr_backend": original.backend_info,
                  "notes": ["复用现有几何候选和识别模型；本图片实验不运行YOLO。",
                            "两种预处理视图不等于多帧稳定证据；未验证按压目标。",
                            "截图可能包含绘制痕迹，不能代替原始相机帧评估。"], "images": []}
        scores = list(dict.fromkeys([args.ocr_min_score, *args.compare_thresholds]))
        for index, path in enumerate(args.images, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"无法读取图片: {path}")
            tick = time.perf_counter()
            rings = vision.repeated_button_group(vision.find_rings(image, args.max_side))
            box = vision.group_box(rings, image.shape)
            geometry_ms = (time.perf_counter()-tick)*1000
            row = {"image": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "shape_hw": list(image.shape[:2]), "baseline": [], "geometry_ms": geometry_ms}
            first = None
            for score in scores:
                original.min_score = score
                tick = time.perf_counter()
                observations = original.recognize(image, rings, box) if box else []
                baseline = frame_record(image, rings, observations, (time.perf_counter()-tick)*1000,
                                        original.backend_info, geometry_ms)
                row["baseline"].append({"threshold": score, **baseline.to_dict()})
                if first is None:
                    first = baseline
            original.min_score = args.ocr_min_score
            selected = compact_group(rings)
            tick = time.perf_counter()
            observations = experiment.recognize(image, selected, vision.group_box(selected, image.shape)) if selected else []
            enhanced = frame_record(image, selected, observations, (time.perf_counter()-tick)*1000,
                                    original.backend_info, geometry_ms)
            record = enhanced.to_dict()
            for b, diag in zip(record["buttons"], experiment.diagnostics if selected else []):
                b["text_region_source"] = diag["source"] if b["text"] else None
                b["preprocessing_audit"] = diag
            row["generalized"] = record
            for kind, result in (("baseline", first), ("generalized", enhanced)):
                target = args.output_dir/f"{index:02d}_{path.stem}_{kind}.jpg"
                if target.resolve() in [p.resolve() for p in args.images]:
                    raise ValueError("输出文件不能覆盖输入图片")
                if not cv2.imwrite(str(target), preview(image, result)):
                    raise OSError(f"无法写入标注图: {target}")
                row[f"{kind}_overlay"] = str(target.resolve())
            report["images"].append(row)
            print(json.dumps({"image": path.name,
                  "baseline": [{"threshold": b["threshold"], "numbers": b["recognized_numbers"]} for b in row["baseline"]],
                  "generalized": record["recognized_numbers"], "rings": len(selected)}, ensure_ascii=False), flush=True)
            if args.show:
                cv2.imshow("Independent panel OCR experiment", preview(image, enhanced))
                if cv2.waitKey(0) & 0xff in (27, ord("q"), ord("Q")):
                    break
        output = args.output_dir/"report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告: {output.resolve()}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
