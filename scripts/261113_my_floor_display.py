"""独立识别电梯内黑色面板的当前楼层：CUDA优先，无机械臂动作。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from elevator.floor_display_261113 import DisplayOCR, FloorDisplayReader, draw_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--camera', action='store_true', help='只运行相机预览')
    group.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--config', type=Path, default=ROOT/'config/default.yaml')
    parser.add_argument('--ocr-device', choices=('auto', 'cuda', 'cpu'), default='auto',
                        help='auto先尝试真实CUDA，失败时明确报告CPU；cuda失败直接退出')
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='可选：原图中完整LCD的矩形，包含左箭头和右数字，不是整块黑面板')
    parser.add_argument('--min-confidence', type=float, default=.80)
    parser.add_argument('--stable-reads', type=int, default=3, help='独立OCR结果的连续一致次数；缓存帧不计数')
    parser.add_argument('--ocr-interval', type=float, default=.20)
    parser.add_argument('--max-age', type=float, default=.80)
    parser.add_argument('--refresh-s', type=float, default=1.5)
    parser.add_argument('--max-side', type=int, default=960)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'readme/261113_floor_display_results')
    parser.add_argument('--show', action='store_true', help='离线图片也显示窗口')
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8 or args.gpu_index < 0:
        parser.error('threads需1..8；gpu-index需非负整数')
    try:
        make_reader(None, args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.camera and args.images is None:
        args.images = sorted((ROOT/'readme/PairDrop_files_20261112_1539').glob('*.jpg'))
        if not args.images:
            parser.error('参考照片目录中没有jpg图片')
    return args


def make_reader(ocr, args):
    return FloorDisplayReader(ocr, min_confidence=args.min_confidence, required=args.stable_reads,
                             interval_s=args.ocr_interval, max_age_s=args.max_age, refresh_s=args.refresh_s,
                             max_side=args.max_side, roi=args.roi)


def run_camera(reader, args):
    from drivers.camera import make_camera
    from utils.camera_utils import load_config
    camera = make_camera(load_config(args.config))
    pool = ThreadPoolExecutor(max_workers=1)
    future = latest = rendered = None
    epoch, failures = 0, 0
    last_print, last_state = -float('inf'), None
    window = 'Cabin floor display - NO ROBOT'
    try:
        camera.open()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.get_frame()
            image = None if frame is None else frame[0]
            if image is None:
                failures += 1
                epoch += 1
                latest = rendered = None
                blank = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank, 'NO CAMERA FRAME - NO CURRENT FLOOR', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 100, 255), 2)
                cv2.imshow(window, blank)
                if cv2.waitKey(10) & 255 in (27, ord('q'), ord('Q')):
                    break
                if failures >= 10:
                    raise RuntimeError('连续10次相机取帧失败，退出预览')
                continue
            failures = 0
            if future is not None and future.done():
                source, result, processed_epoch = future.result()
                future = None
                now = time.perf_counter()
                if processed_epoch == epoch and result.captured_at is not None and 0 <= now-result.captured_at <= args.max_age:
                    latest, rendered = result, draw_result(source, result)
                    key = (result.state, result.observed_floor, result.current_floor)
                    if key != last_state or now-last_print >= 1.:
                        print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)
                        last_print, last_state = now, key
                else:
                    latest = rendered = None
            if future is None:
                source, at, submitted_epoch = image.copy(), time.perf_counter(), epoch
                def process(im=source, when=at, generation=submitted_epoch):
                    return im, reader.process(im, when, generation=generation), generation
                future = pool.submit(process)
            age = None if latest is None else time.perf_counter()-latest.captured_at
            display = rendered.copy() if rendered is not None and age is not None and age <= args.max_age else image.copy()
            if age is not None:
                cv2.putText(display, f'result age={age:.3f}s', (8, 95), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('r'), ord('R')):
                epoch += 1
                latest = rendered = None
    finally:
        try:
            camera.close()
        finally:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            finally:
                cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args = parse_args(argv)
    cv2.setNumThreads(args.threads)
    try:
        start = time.perf_counter()
        print(f'[Startup] 初始化一次OCR，requested={args.ocr_device}；仅本地模型，无自动下载', flush=True)
        ocr = DisplayOCR(device=args.ocr_device, gpu_index=args.gpu_index, threads=args.threads)
        startup_s = time.perf_counter()-start
        print('[OCR] '+json.dumps(ocr.info, ensure_ascii=False), flush=True)
        print(f'[Startup] OCR初始化/预热={startup_s:.2f}s', flush=True)
        reader = make_reader(ocr, args)
        if args.camera:
            return run_camera(reader, args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(branch='CABIN_FLOOR_DISPLAY', release_date='2026-11-13', hardware_camera_or_robot_accessed=False,
                      actions_enabled=False, backend=ocr.info, startup_s=startup_s,
                      min_confidence=args.min_confidence, independent_stable_reads=args.stable_reads,
                      ocr_interval_s=args.ocr_interval, images=[])
        inputs = {path.resolve() for path in args.images}
        for index, path in enumerate(args.images, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'无法读取图片：{path}')
            result = reader.process(image, offline=True)
            output = args.output_dir/f'{index:02d}_{path.stem}_floor.jpg'
            if output.resolve() in inputs:
                raise ValueError('输出不能覆盖输入照片')
            if not cv2.imwrite(str(output), draw_result(image, result)):
                raise OSError(f'无法保存结果图：{output}')
            record = result.to_dict()
            record.update(image=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), overlay=str(output.resolve()))
            if result.crop is not None:
                crop_path = args.output_dir/f'{index:02d}_{path.stem}_display.jpg'
                if crop_path.resolve() in inputs:
                    raise ValueError('LCD裁剪输出不能覆盖输入照片')
                if not cv2.imwrite(str(crop_path), result.crop):
                    raise OSError(f'无法保存LCD区域：{crop_path}')
                record['display_crop'] = str(crop_path.resolve())
            report['images'].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.show:
                cv2.imshow('Cabin floor display - NO ROBOT', draw_result(image, result))
                if cv2.waitKey(0) & 255 in (27, ord('q'), ord('Q')):
                    break
        (args.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
