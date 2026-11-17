"""两键外呼面板独立预览：画面上方=UP、下方=DOWN，无机械臂动作。"""
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

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from elevator.call_panel_261112 import CallPanelDetector,draw_result


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--camera',action='store_true',help='仅相机预览，不连接机械臂')
    group.add_argument('--images',type=Path,nargs='+')
    parser.add_argument('--config',type=Path,default=ROOT/'config/default.yaml')
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X1','Y1','X2','Y2'),help='原图像素选区；仍检查周围额外圆环')
    parser.add_argument('--max-side',type=int,default=1200,help='获取分辨率上限；先640，失败后逐步提高')
    parser.add_argument('--stable-frames',type=int,default=3)
    parser.add_argument('--refresh-s',type=float,default=2.,help='全图复查间隔；场景变化会提前复查')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'readme/261112_call_panel_branch')
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.threads<=8:parser.error('threads需为1..8')
    try:CallPanelDetector(max_side=args.max_side,required=args.stable_frames,refresh_s=args.refresh_s,roi=args.roi)
    except ValueError as exc:parser.error(str(exc))
    if not args.camera and args.images is None:
        base=ROOT/'readme/PairDrop_files_20261112_1513'
        args.images=[base/'电梯上下按键全.jpg',base/'电梯上下按键大.jpg']
    return args


def run_camera(detector,args):
    from drivers.camera import make_camera
    from utils.camera_utils import load_config
    camera=make_camera(load_config(args.config));pool=ThreadPoolExecutor(max_workers=1)
    future=None;latest=None;rendered=None;epoch=0;failures=0;last_print=-float('inf');last_state=None
    window='Two-key call panel - NO ROBOT'
    try:
        camera.open();cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        while True:
            frame=camera.get_frame();image=None if frame is None else frame[0]
            if image is None:
                failures+=1;epoch+=1;latest=rendered=None
                blank=np.zeros((480,640,3),np.uint8)
                cv2.putText(blank,'NO CAMERA FRAME - NO CURRENT TARGET',(10,40),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,100,255),2)
                cv2.imshow(window,blank)
                if cv2.waitKey(10)&255 in (27,ord('q'),ord('Q')):break
                if failures>=10:raise RuntimeError('连续10次相机取帧失败，退出预览')
                continue
            failures=0
            if future is not None and future.done():
                source,result,processed_epoch=future.result();future=None
                now=time.perf_counter()
                if processed_epoch==epoch and result.captured_at is not None and 0<=now-result.captured_at<=1.:
                    latest=result;rendered=draw_result(source,result)
                    if result.state!=last_state or now-last_print>=1.:
                        print(json.dumps(result.to_dict(),ensure_ascii=False),flush=True)
                        last_print=now;last_state=result.state
                else:latest=rendered=None
            if future is None:
                source=image.copy();at=time.perf_counter();submitted_epoch=epoch
                def process(im=source,when=at,generation=submitted_epoch):
                    return im,detector.process(im,when,generation=generation),generation
                future=pool.submit(process)
            age=None if latest is None else time.perf_counter()-latest.captured_at
            display=rendered.copy() if rendered is not None and age is not None and age<=1. else image.copy()
            if age is not None:
                cv2.putText(display,f'result age={age:.3f}s',(8,74),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,255),1,cv2.LINE_AA)
            cv2.imshow(window,display);key=cv2.waitKey(1)&255
            if key in (27,ord('q'),ord('Q')):break
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key in (ord('r'),ord('R')):epoch+=1;latest=rendered=None
    finally:
        try:camera.close()
        finally:
            try:pool.shutdown(wait=True,cancel_futures=True)
            finally:cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args=parse_args(argv);cv2.setNumThreads(args.threads)
    detector=CallPanelDetector(max_side=args.max_side,required=args.stable_frames,refresh_s=args.refresh_s,roi=args.roi)
    try:
        if args.camera:return run_camera(detector,args)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        report=dict(branch='TWO_KEY_CALL_PANEL',date='2026-11-12',hardware_accessed=False,
            position_rule='Upper=UP, lower=DOWN; upright camera/standard two-key layout assumed.',
            models_loaded=[],images=[],source_config_modified=False,actions_enabled=False)
        inputs={p.resolve() for p in args.images}
        for index,path in enumerate(args.images,1):
            image=cv2.imread(str(path))
            if image is None:raise ValueError(f'无法读取图片：{path}')
            result=detector.process(image,offline=True)
            output=args.output_dir/f'{index:02d}_{path.stem}_call_panel.jpg'
            if output.resolve() in inputs:raise ValueError('标注输出不能覆盖输入照片')
            if not cv2.imwrite(str(output),draw_result(image,result)):raise OSError(f'无法保存标注：{output}')
            record=result.to_dict();record.update(image=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),overlay=str(output.resolve()))
            report['images'].append(record)
            print(json.dumps(record,ensure_ascii=False),flush=True)
            if args.show:
                cv2.imshow('Two-key call panel - NO ROBOT',draw_result(image,result))
                if cv2.waitKey(0)&255 in (27,ord('q'),ord('Q')):break
        (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return 0
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True);return 2
    finally:
        if args.show:cv2.destroyAllWindows()


if __name__=='__main__':raise SystemExit(main())
