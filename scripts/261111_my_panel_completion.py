"""独立面板实验：缺失楼层候选、开门/关门符号。

原始 OCR 不改写。候选依据当前圆环与像素，缺失/跳层/0楼歧义保留。
图片模式保存本次指定输入的标注和 JSON；相机预览不保存每帧图像。
--camera --head-up额外复用已验证抬头动作；默认预览和图片模式不连接机械臂。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from elevator import perception as vision

# Re-export the shared recognition API for existing experiment imports/tests.
from elevator.panel_261112 import (
    THRESHOLDS, classify_door, glyph_shape_evidence, floor_value, function_word,
    layout_candidates, target_candidates, analyze, draw, region_patch, patch_matches,
    isolated_glyph_views, RegionOCR, EvidenceFrameResult, panel_search_box,
    grid_steps, confirm_rim, recover_panel_rims, FastPanelPerception,
)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--images',type=Path,nargs='+')
    group.add_argument('--camera',action='store_true',help='单相机预览；只有额外指定--head-up才初始化机械臂')
    parser.add_argument('--head-up',action='store_true',help='真实动作：相机预览前按现有配置抬头/闭合夹爪，退出失能；不按按钮')
    parser.add_argument('--workflow-config',type=Path,default=ROOT/'config/261105_my_elevator.yaml',help='只供--head-up复用startup_view/gripper配置')
    parser.add_argument('--config',type=Path,default=ROOT/'config/default.yaml')
    parser.add_argument('--ocr-device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--device',default='0')
    parser.add_argument('--ocr-min-score',type=float,default=.70)
    parser.add_argument('--max-side',type=int,default=1200)
    parser.add_argument('--ocr-buttons-per-update',type=int,default=4,help='实时每轮最多识别的按钮局部数')
    parser.add_argument('--ocr-refresh-s',type=float,default=2.,help='稳定数字重新OCR的间隔，仍逐轮核对当前局部像素')
    parser.add_argument('--use-yolo',action='store_true',help='额外复用原YOLOE面板检测；默认仅复用圆环+OCR')
    parser.add_argument('--floor-list',help='人工核对的可选楼层集合，如 --floor-list=-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12')
    parser.add_argument('--inventory-complete',action='store_true',help='人工确认楼层集合完整且所有数字圆环均已检出，才启用全量匹配')
    parser.add_argument('--target-floor',type=int,help='仅打印该楼层候选像素，绝不执行动作')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'readme/261111_panel_completion')
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args(argv)
    if args.head_up and not args.camera:parser.error('--head-up只允许与--camera一起使用，图片模式不执行动作')
    if not math.isfinite(args.ocr_min_score) or not 0<args.ocr_min_score<=1:parser.error('OCR阈值需在(0,1]')
    if not 240<=args.max_side<=1600:parser.error('max-side需在240..1600')
    if not 1<=args.ocr_buttons_per_update<=12:parser.error('ocr-buttons-per-update需为1..12')
    if not math.isfinite(args.ocr_refresh_s) or not .5<=args.ocr_refresh_s<=10:parser.error('ocr-refresh-s需为0.5..10秒')
    try:args.floor_list=None if args.floor_list is None else [int(v) for v in args.floor_list.split(',')]
    except ValueError:parser.error('floor-list为逗号分隔的整数')
    if args.inventory_complete and args.floor_list is None:parser.error('inventory-complete需要floor-list')
    if args.floor_list is not None and (not args.floor_list or len(set(args.floor_list))!=len(args.floor_list)
                                        or any(abs(x)>99 for x in args.floor_list)):parser.error('floor-list需为不同的-99..99整数')
    if not args.camera and args.images is None:
        args.images=[ROOT/'readme/mmexport1793686157333.jpg',ROOT/'readme/mmexport1793686160331.jpg']
    args.geometry_only=not args.use_yolo
    args.ocr_preprocessing='bright_dark';args.ocr_threads=4;args.ocr_gpu_index=0
    return args


def make_head_robot(cfg,args):
    # Lazy imports: offline and ordinary camera preview do not load a control
    # stack. Reuse the working workflow; no SDK/driver logic is duplicated.
    import copy
    from elevator.app import load_workflow
    from elevator.motion import Robot
    options=load_workflow(args.workflow_config)
    robot_cfg=copy.deepcopy(cfg);robot_cfg['trajectory']=options['trajectory']
    return Robot(robot_cfg),options


def run_camera(detector,args):
    from drivers.camera import make_camera
    cfg=vision.load_config(args.config)
    camera=make_camera(cfg);worker=ThreadPoolExecutor(max_workers=1);future=None;latest=None;latest_display=None;age=0.;failures=0
    robot=None;head_cancelled=False;head_enabled=getattr(args,'head_up',False)
    window='261111 panel candidates - '+('HEAD PREVIEW / NO PRESS' if head_enabled else 'NO ROBOT')
    try:
        camera.open()
        if head_enabled:
            robot,options=make_head_robot(cfg,args)
            def cancelled():
                nonlocal head_cancelled
                frame=camera.get_frame()
                if frame is None or frame[0] is None:raise RuntimeError('抬头时相机帧获取失败，取消后续识别')
                preview=frame[0].copy()
                cv2.putText(preview,'HEAD UP: Q/ESC/SPACE cancel',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,200,255),2)
                cv2.imshow(window,preview)
                head_cancelled=cv2.waitKey(1)&0xff in (27,ord('q'),ord('Q'),32)
                return head_cancelled
            robot.start()
            head=list(options['startup_view']['joints_deg'][:6])
            if options['gripper']['initialize']:head.append(options['gripper']['closed_angle_deg'])
            print(f'[PreviewHead] 真实抬头目标deg={head}；参考={options["startup_view"]["duration_s"]}s；不执行按压。退出停止控制并失能，请保持机械支撑。',flush=True)
            robot.head_up(head,options['startup_view']['duration_s'],cancelled,
                          closed_tolerance=options['gripper']['closed_tolerance_rad'],
                          wait_for_arrival=options['startup_view']['wait_for_arrival'])
            detector.request_refresh()
        while True:
            frame=camera.get_frame();image=frame[0] if frame is not None else None
            if image is None:
                failures+=1
                if failures>=10:raise RuntimeError('连续10次无有效相机帧，退出预览')
                if cv2.waitKey(10)&0xff in (27,ord('q'),ord('Q')):break
                continue
            failures=0
            if future is not None and future.done():
                latest_image,latest=future.result();age=latest['captured_at'];future=None
                latest_display=draw(latest_image,latest) if time.perf_counter()-age<4. else None
                if latest_display is not None and args.target_floor is not None:
                    print(json.dumps(target_candidates(latest,args.target_floor),ensure_ascii=False),flush=True)
            if future is None:
                original=image.copy();captured_at=time.perf_counter()
                def process(frame=original,at=captured_at):
                    result=detector.process(frame,captured_at=at)
                    record=analyze(frame,result,floor_list=args.floor_list,inventory_complete=args.inventory_complete,
                                   min_score=args.ocr_min_score)
                    record['head_up_enabled']=head_enabled
                    return frame,record
                future=worker.submit(process)
            # Coordinate overlays belong to the image actually analyzed, not a
            # newer frame after the user moved the camera. No image is saved.
            display=latest_display if latest_display is not None and time.perf_counter()-age<4. else image
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&0xff
            if key in (27,ord('q'),ord('Q')):break
            if key==32 and robot is not None:robot.stop()
            if key in (ord('r'),ord('R')) and hasattr(detector,'request_refresh'):detector.request_refresh()
    except Exception:
        if not head_cancelled:raise
        print('[PreviewHead] 用户取消抬头，退出预览。',flush=True)
    finally:
        pending_exception=sys.exc_info()[0] is not None;errors=[]
        cleanups=([lambda:robot.close(home=False)] if robot is not None else [])
        cleanups.extend([camera.close,lambda:worker.shutdown(wait=True,cancel_futures=True),cv2.destroyAllWindows])
        for cleanup in cleanups:
            try:cleanup()
            except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
        if errors:
            message='退出清理未全部完成：'+'；'.join(errors)
            if not pending_exception:raise RuntimeError(message)
            print('[Exit] '+message,file=sys.stderr,flush=True)
    return 0


def main(argv=None):
    args=parse_args(argv)
    try:
        began=time.perf_counter();detector=FastPanelPerception(vision.load_config(args.config),args)
        initialization_ms=(time.perf_counter()-began)*1000
        if args.camera:return run_camera(detector,args)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        report=dict(experiment='261111_panel_completion',workflow_modified=False,hardware_accessed=False,
                    initialization_ms=initialization_ms,ocr_backend=detector.backend_info,
                    ocr_min_score=args.ocr_min_score,floor_list=args.floor_list,
                    images=[],executable=False)
        for index,path in enumerate(args.images,1):
            image=cv2.imread(str(path))
            if image is None:raise ValueError(f'无法读取图片：{path}')
            result=detector.process(image,offline=True)
            record=analyze(image,result,floor_list=args.floor_list,inventory_complete=args.inventory_complete,
                           min_score=args.ocr_min_score)
            record.update(image=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            output=args.output_dir/f'{index:02d}_{path.stem}_candidates.jpg'
            if output.resolve() in [p.resolve() for p in args.images]:raise ValueError('输出不能覆盖输入图片')
            if not cv2.imwrite(str(output),draw(image,record)):raise OSError(f'无法保存：{output}')
            record['overlay']=str(output.resolve())
            if args.target_floor is not None:record['requested_floor_candidates']=target_candidates(record,args.target_floor)
            report['images'].append(record)
            summary=dict(image=path.name,ocr=result.to_dict()['recognized_numbers'],
                supplemented=[dict(center=b['button_candidate_center_px'],candidates=b['shape_layout_agreement'],status=b['status'])
                              for b in record['layout']['buttons'] if b['status']=='SHAPE_LAYOUT_CANDIDATE'],
                doors=[dict(label=d['label'],center=d['center_px'],score=round(d['score'],3)) for d in record['door_symbols']],
                extra_ms=round(record['timings_ms']['extra_total'],2))
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            if args.show:
                cv2.imshow('261111 panel candidates - NO ROBOT',draw(image,record))
                if cv2.waitKey(0)&0xff in (27,ord('q'),ord('Q')):break
        output=args.output_dir/'report.json'
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'报告：{output.resolve()}',flush=True)
        return 0
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True);return 2
    finally:
        if args.show:cv2.destroyAllWindows()


if __name__=='__main__':
    raise SystemExit(main())

