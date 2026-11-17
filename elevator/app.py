"""Combined original grasp / simplified OCR elevator workflow."""
from __future__ import annotations
import argparse
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import select
import sys
import time
import numpy as np
import yaml
from . import geometry, motion

ROOT=Path(__file__).resolve().parents[1]


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value=yaml.safe_load(stream)
    if not isinstance(value,dict):
        raise ValueError(f"配置必须为映射：{path}")
    return value


def load_workflow(path):
    o=read_yaml(path)
    placement=o.setdefault("grasp_place",{})
    if not isinstance(placement,dict):raise ValueError("grasp_place必须为映射")
    xyz=np.asarray(placement.setdefault("xyz_m",[.25,-.38,.35]),float)
    if xyz.shape!=(3,) or not np.isfinite(xyz).all():
        raise ValueError("grasp_place.xyz_m需要三个有限米值（base坐标系end_link位置）")
    placement["xyz_m"]=xyz.tolist()
    trajectory=o.setdefault("trajectory",{})
    if not isinstance(trajectory,dict):raise ValueError("trajectory必须为映射")
    interval=float(trajectory.setdefault("sample_interval_s",.1))
    if not math.isfinite(interval) or not .01<=interval<=.1:
        raise ValueError("trajectory.sample_interval_s需在[0.01,0.1]秒")
    trajectory["sample_interval_s"]=interval
    for group in ("ocr","temporal","startup_view","gripper","elevator","tool"):
        if not isinstance(o.get(group),dict):
            raise ValueError(f"电梯配置缺少{group}")
    if "initial_poses" in o:
        initial=o["initial_poses"]
        if not isinstance(initial,dict):raise ValueError("initial_poses必须为映射")
        ready=initial.get("grasp_ready")
        if not isinstance(ready,dict):raise ValueError("initial_poses.grasp_ready必须为位姿映射")
        for key in ("x","y","z","roll","pitch","yaw","duration"):
            try:value=float(ready[key])
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError(f"initial_poses.grasp_ready.{key}必须为有限数值") from exc
            if not math.isfinite(value) or key=="duration" and value<=0:
                raise ValueError(f"initial_poses.grasp_ready.{key}必须为有限数值，duration须大于0")
            ready[key]=value
        for key in ("call_view_deg","floor_view_deg","display_view_deg"):
            joints=np.asarray(initial.get(key),float)
            if joints.shape!=(6,) or not np.isfinite(joints).all():
                raise ValueError(f"initial_poses.{key}需要六个有限关节角，单位度")
            initial[key]=joints.tolist()
        # The standalone elevator entries use the floor-button view. Keep the
        # legacy in-memory key so their proven head-up code stays unchanged.
        o["startup_view"]["joints_deg"]=list(initial["floor_view_deg"])
    e=o["elevator"]
    e.setdefault("stable_snapshot_age_s",2.)
    if not isinstance(e.setdefault("floor_cache_until_update",False),bool):
        raise ValueError("elevator.floor_cache_until_update必须为布尔值")
    e.setdefault("level_during_approach",True)
    e.setdefault("target_offset_base_m",[0.,0.,0.])
    geometry.offset_target_base(None,e["target_offset_base_m"])
    if not isinstance(e["level_during_approach"],bool):
        raise ValueError("level_during_approach必须为布尔值")
    for key,lo,hi in (("standoff_m",.03,.10),("move_duration_s",.5,10.),("contact_duration_s",.5,10.),
                      ("retreat_duration_s",.5,10.),("dwell_s",0.,3.),("press_travel_m",0.,.005),
                      ("max_result_age_s",.1,5.),("floor_wait_s",1.,30.),("stable_snapshot_age_s",0.,3.)):
        v=float(e[key])
        if not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f"elevator.{key}必须在[{lo},{hi}]")
        e[key]=v
    if e.get("panel_type") not in ("printed","physical"):
        raise ValueError("panel_type需为printed或physical")
    if e["panel_type"]=="printed" and e["press_travel_m"]!=0:
        raise ValueError("打印面板不允许追加压入")
    score=float(o["ocr"]["min_confidence"])
    if not math.isfinite(score) or not 0<=score<=1:
        raise ValueError("OCR置信度需在[0,1]")
    if o["ocr"].setdefault("preprocessing","bright_dark") not in ("bright","bright_dark"):
        raise ValueError("ocr.preprocessing需为bright或bright_dark")
    budget=o['ocr'].setdefault('buttons_per_update',4)
    refresh=float(o['ocr'].setdefault('refresh_s',2.))
    if isinstance(budget,bool) or int(budget)!=budget or not 1<=budget<=12:
        raise ValueError('ocr.buttons_per_update需为1..12整数')
    if not math.isfinite(refresh) or not .5<=refresh<=10:
        raise ValueError('ocr.refresh_s需在0.5..10秒')
    o['ocr']['buttons_per_update']=int(budget);o['ocr']['refresh_s']=refresh
    offset=np.asarray(o["tool"]["tip_offset_in_end_m"],float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():
        raise ValueError("工具偏移需要三个有限米值")
    joints=np.asarray(o["startup_view"]["joints_deg"],float)
    if not isinstance(o["startup_view"].setdefault("wait_for_arrival",True),bool):
        raise ValueError("startup_view.wait_for_arrival必须为布尔值")
    if joints.shape not in ((6,),(7,)) or not np.isfinite(joints).all():
        raise ValueError("抬头角度需要六或七个有限度数")
    closed_angle=float(o["gripper"].setdefault("closed_angle_deg",0.))
    if not math.isfinite(closed_angle):raise ValueError("gripper.closed_angle_deg必须为有限角度")
    o["gripper"]["closed_angle_deg"]=closed_angle
    if joints.size==7 and not math.isclose(joints[6],closed_angle,abs_tol=1e-9):
        raise ValueError("startup_view第七轴须与gripper.closed_angle_deg一致；也可只配置前六轴")
    value=float(o["startup_view"]["duration_s"])
    if not math.isfinite(value) or value<=0:
        raise ValueError("初始化参考时间必须为有限正数")
    tolerance=float(o["gripper"].setdefault("closed_tolerance_rad",.01))
    if not math.isfinite(tolerance) or not 0<tolerance<=.12:
        raise ValueError("gripper.closed_tolerance_rad需在(0,0.12]rad")
    if not isinstance(o["gripper"]["initialize"],bool):
        raise ValueError("gripper.initialize必须为布尔值")
    return o


def apply_initial_poses(cfg,options):
    """Apply the shared grasp ready pose to a copy, preserving default.yaml."""
    cfg=copy.deepcopy(cfg)
    ready=options.get("initial_poses",{}).get("grasp_ready")
    if ready is not None:
        cfg.setdefault("robot",{})["ready_pose"]=copy.deepcopy(ready)
    return cfg


def _perception_worker(connection,requests,cfg,settings):
    try:
        began=time.perf_counter()
        from .panel_261112 import FastPanelPerception
        settings=dict(settings,workflow_evidence=True)
        detector=FastPanelPerception(cfg,argparse.Namespace(**settings))
        print("[OCR] "+json.dumps(detector.ocr.backend_info,ensure_ascii=False),flush=True)
        print(f"[OCR] preprocessing={settings.get('ocr_preprocessing','bright')}；"
              f"min_confidence={settings.get('ocr_min_score',.80):.2f}",flush=True)
        print(f"[OCR] shared=panel_261112；局部预算={settings.get('ocr_buttons_per_update',4)}；"
              f"稳定刷新={settings.get('ocr_refresh_s',2.):g}s；显示缓存不重复计票",flush=True)
        initialized=time.perf_counter()
        detector.warmup(cfg["camera"].get("color_height",720),cfg["camera"].get("color_width",1280))
        print(f"[VisionReady] 导入/模型初始化={initialized-began:.2f}s；YOLO预热={time.perf_counter()-initialized:.2f}s",flush=True)
        connection.send(("ready",))
        panel_generation=None
        while True:
            request=requests.get()
            if request is None:break
            kind,frame_id,at,image,depth,K,*extra=request
            generation=extra[0] if extra else 0
            if kind=="grasp":
                from utils.ordinary_grasp import estimate_grasps
                opts=detector.yolo_options
                results=detector.model.predict(image,verbose=False,device=opts.get("device","cpu"),
                    conf=float(opts.get("conf",.25)),iou=float(opts.get("iou",.45)))
                result=estimate_grasps(results,depth,K,
                    depth_quantile=float(cfg.get("grasp_pipeline",{}).get("grasp",{}).get("depth_quantile",.75)))
            elif kind=="panel":
                if generation!=panel_generation:
                    detector.request_refresh();panel_generation=generation
                result=detector.process(image,captured_at=at)
            else:raise ValueError(f"Unknown inference task: {kind}")
            connection.send(("result",frame_id,result))
    except (EOFError,BrokenPipeError):
        pass
    except BaseException as exc:
        try:connection.send(("error",f"{type(exc).__name__}: {exc}"))
        except OSError:pass
    finally:connection.close()


class PerceptionWorker:
    """One inference frame in flight; camera display stays responsive."""
    def __init__(self,cfg,settings,timeout=20.):
        ctx=mp.get_context("spawn")
        self.connection,child=ctx.Pipe()
        self.requests=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=_perception_worker,args=(child,self.requests,cfg,settings),daemon=True)
        self.process.start();child.close()
        self.pending=None;self.submitted_at=0.;self.timeout=timeout

    def wait_ready(self,timeout=90.,on_wait=None):
        until=time.perf_counter()+timeout
        while time.perf_counter()<until:
            if self.connection.poll(.05):
                message=self.connection.recv()
                if message[0]=="ready":return
                raise RuntimeError(str(message))
            if not self.process.is_alive():raise RuntimeError("OCR初始化进程退出")
            if on_wait is not None:on_wait()
        raise TimeoutError("OCR初始化超时")

    def submit(self,snapshot,kind="panel"):
        if kind not in ("panel","grasp"):raise ValueError("Unknown inference task")
        if self.pending is None:
            self.requests.put_nowait((kind,snapshot.frame_id,snapshot.captured_at,snapshot.color,
                                     snapshot.depth_mm if kind=="grasp" else None,snapshot.K if kind=="grasp" else None,
                                     snapshot.generation))
            self.pending,self.submitted_at=snapshot,time.perf_counter()

    def receive(self):
        if self.connection.poll():
            message=self.connection.recv()
            if message[0]=="error":raise RuntimeError(message[1])
            if message[0]!="result" or self.pending is None or message[1]!=self.pending.frame_id:
                raise RuntimeError("OCR结果与RGB-D不匹配")
            snapshot,self.pending=self.pending,None
            return snapshot,message[2]
        if not self.process.is_alive():raise RuntimeError("OCR进程退出")
        if self.pending is not None and time.perf_counter()-self.submitted_at>self.timeout:
            raise TimeoutError("OCR推理超时")
        return None

    def close(self):
        self.requests.cancel_join_thread();self.requests.close();self.connection.close()
        if self.process.is_alive():self.process.terminate()
        self.process.join(2.)
        if self.process.is_alive():self.process.kill();self.process.join(1.)


def report_reached(robot,phase,target,offset):
    actual=robot.pose()
    tip=actual[:3,3]+actual[:3,:3]@np.asarray(offset)
    intended=target[:3,3]+target[:3,:3]@np.asarray(offset)
    print(f"[Reached:{phase}] model_tip_base_m={tip.tolist()} "
          f"target_tip_base_m={intended.tolist()} model_error_mm={((tip-intended)*1000).round(2).tolist()}；"
          "依赖当前工具偏移和编码器/FK，非独立尺量",flush=True)


def execute_button(robot,point,level,settings,offset,stage,cancelled=lambda:False, *, position_only=False):
    """main.py-like pregrasp/grasp sequence, now approach/surface/retreat."""
    if stage not in ("approach","contact"):
        raise ValueError("此动作仅用于approach/contact")
    direction=geometry.horizontal_approach_direction(level) if position_only else None
    poses=geometry.button_poses(point,level,offset,settings["standoff_m"],settings["press_travel_m"],forward_base=direction)
    mode='orientation=FREE_POSITION_IK' if position_only else '目标roll/pitch=0/0rad'
    print(f"[Button] surface_base_m={np.asarray(point).tolist()}；{mode}；stage={stage}",flush=True)
    def move(name,duration):
        if position_only:
            goal=poses[name]
            tip=goal[:3,3]+goal[:3,:3]@np.asarray(offset)
            poses[name]=robot.move_position(tip,duration,cancelled,tip_offset=offset)
        else:robot.move(poses[name],duration,cancelled)
    move("approach",settings["move_duration_s"])
    report_reached(robot,"approach",poses["approach"],offset)
    if stage=="approach":
        print("[Button] 到达前置位置，本阶段不触纸",flush=True)
        return poses
    move("surface",settings["contact_duration_s"])
    report_reached(robot,"surface",poses["surface"],offset)
    if settings["press_travel_m"]>0:
        move("press",settings["contact_duration_s"])
    until=time.monotonic()+settings["dwell_s"]
    while time.monotonic()<until:
        if cancelled():
            robot.stop()
            raise motion.MotionCancelled("用户停止")
        time.sleep(.02)
    move("retreat",settings["retreat_duration_s"])
    report_reached(robot,"retreat",poses["retreat"],offset)
    print("[Button] 已到估计表面并撤回；未独立验证按钮触发",flush=True)
    return poses


def _tracker(options):
    from .temporal import TemporalButtonTracker
    t=options["temporal"]
    return TemporalButtonTracker(required=int(t["required"]),min_score=float(options["ocr"]["min_confidence"]),
        memory_age_s=float(t["memory_age_s"]),memory_frames=int(t["memory_frames"]))


def remember_stable(cache,snapshot,result,tracker,min_score):
    if result.errors or result.panel_bbox is None or not tracker.panel.stable:
        cache.clear()
        return
    for b in result.buttons:
        if b.status!="STABLE" or b.text is None:continue
        try:
            floor=geometry.normalize_floor_id(b.text)
            geometry.select_floor(result,None,floor,min_score)
        except ValueError:continue
        cache[floor]=(snapshot,result)


def resolve_observation(request,last,snapshot,cache,tracker,current,settings,min_score):
    """Prefer a new result; otherwise reuse a complete, checked stable bundle."""
    error=ValueError("等待稳定数字/有效深度")
    if snapshot is not None and not last.errors and snapshot.generation==current.generation and \
            request["at"]<snapshot.captured_at<=current.captured_at and \
            current.captured_at-snapshot.captured_at<=settings["max_result_age_s"]:
        try:
            b=geometry.select_floor(last,tracker,request["floor"],min_score,current.captured_at)
            source=('CURRENT_RIM_POSITION_INFERENCE' if geometry.position_floor_indices(last,request['floor'])
                    else 'CURRENT_RIM_LABEL_MEMORY' if getattr(last,'tracking',{}).get('workflow_evidence')
                    and b.text is None else 'NEW_RESULT')
            return snapshot,b,source
        except ValueError as exc:error=exc
    saved=cache.get(request["floor"])
    if saved is None or settings["stable_snapshot_age_s"]<=0 or not tracker.panel.stable:
        raise error
    snap,result=saved
    b=geometry.select_floor(result,None,request["floor"],min_score,current.captured_at)
    if getattr(last,'tracking',{}).get('workflow_evidence'):
        old_index=next(i for i,item in enumerate(result.buttons) if item is b)
        old_evidence=getattr(result,'tracking',{}).get('region_evidence',{}).get(str(old_index),{})
        nearby=[(i,item) for i,item in enumerate(last.buttons)
                if np.linalg.norm(np.asarray(item.ring.center)-b.ring.center)<max(6.,b.ring.radius*.5)]
        if len(nearby)!=1:raise ValueError('当前没有唯一匹配的实测圆环，不能复用旧图像')
        index,current_button=nearby[0]
        evidence=last.tracking.get('region_evidence',{}).get(str(index),{})
        display=last.tracking.get('display_label_cache',{}).get(str(index),{})
        label=current_button.text if evidence.get('source')=='CURRENT_OCR' else display.get('text')
        if (old_evidence.get('track_id') is None or evidence.get('track_id')!=old_evidence['track_id']
                or evidence.get('source') not in ('CURRENT_OCR','PIXEL_MATCHED_CACHE')
                or not evidence.get('pixel_match') or label is None
                or geometry.normalize_floor_id(label)!=request['floor']):
            raise ValueError('当前文字/像素证据已改变，不能复用旧的稳定图像')
        for key,cached in last.tracking.get('display_label_cache',{}).items():
            proof=last.tracking.get('region_evidence',{}).get(key,{})
            if (int(key)!=index and proof.get('pixel_match') and cached.get('confidence',0)>=min_score
                    and geometry.normalize_floor_id(cached['text'])==request['floor']):
                raise ValueError('当前楼层标签对应多个圆环，缓存身份冲突')
    # A recent conflicting reading at the same rim, or a duplicate floor at
    # another rim, must not be hidden by older stable OCR.
    if last is not None:
        if last.errors or last.panel_bbox is None:raise ValueError("当前面板/识别无效，不能复用")
        for other in last.buttons:
            if other.text is None or other.confidence is None or other.confidence<min_score:continue
            try:label=geometry.normalize_floor_id(other.text)
            except ValueError:continue
            same=np.linalg.norm(np.asarray(other.ring.center)-b.ring.center)<max(6.,b.ring.radius*.5)
            if (same and label!=request["floor"]) or (not same and label==request["floor"]):
                raise ValueError("当前数字与缓存圆环冲突，等待重新稳定")
    geometry.check_cached_observation(snap,current,b,settings["stable_snapshot_age_s"])
    return snap,b,"STABLE_CACHE"


def automatic_floor_ready(floor,snapshot,result,tracker,now,generation,min_score,max_age):
    """Automatic sequence requires the requested digit itself to be stable."""
    if snapshot is None or result is None or result.errors or result.panel_bbox is None or not tracker.panel.stable:
        return False
    if snapshot.generation!=generation or not 0<=now-snapshot.captured_at<=max_age:return False
    try:geometry.select_floor(result,None,floor,min_score)
    except ValueError:return False
    return True


def run_elevator(cfg,options,args,*,before_head=None):
    import cv2
    from drivers.camera import make_camera
    from utils.camera_utils import load_hand_eye
    from utils.transforms import rotation_matrix_to_euler_zyx
    from . import perception
    worker=camera=robot=None
    sequence=bool(getattr(args,"sequence_mode",False))
    tracker=_tracker(options);settings=options["elevator"];offset=options["tool"]["tip_offset_in_end_m"]
    window="Elevator - main.py SDK workflow";display=None;rendered_panel=rendered_result=None
    state="INITIALIZING";typed="";request=None;last=None;last_snapshot=None;level=None;stable_cache={}
    generation=0;frame_id=0;next_submit=0.;failures=0;normal=False;stopped=False;quit_motion=False
    def visible():
        try:return cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1
        except cv2.error:return False
    def cancelled():
        nonlocal quit_motion
        if display is not None:
            moving=display.copy()
            cv2.putText(moving,state+" | SPACE=stop Q=cancel/quit",(8,32),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.imshow(window,moving)
        key=cv2.waitKey(1)&255
        if key in (27,ord("q"),ord("Q")) or not visible():
            quit_motion=True;return True
        return key==32
    def moved():
        nonlocal generation,last,last_snapshot,rendered_panel,rendered_result
        generation+=1;tracker.reset();last=last_snapshot=None;stable_cache.clear()
        rendered_panel=rendered_result=None
        camera.warm_up(3)
    def startup_preview():
        nonlocal display
        color,_=camera.get_frame()
        if color is not None:
            display=color.copy()
            cv2.putText(display,"Loading YOLO/OCR | Q=quit",(8,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            cv2.imshow(window,display)
        if cancelled():raise motion.MotionCancelled("初始化已取消")
    try:
        began=time.perf_counter()
        ocr=options["ocr"]
        infer=dict(device=args.device,roi=None,max_side=1200,geometry_only=False,ocr_threads=4,
            ocr_min_score=float(ocr["min_confidence"]),ocr_device=ocr["device"],
            ocr_preprocessing=ocr["preprocessing"],
            ocr_gpu_index=int(ocr["gpu_index"]),ocr_model_dir=ROOT/ocr["model_dir"],
            ocr_buttons_per_update=ocr.get('buttons_per_update',4),ocr_refresh_s=ocr.get('refresh_s',2.))
        worker=PerceptionWorker(cfg,infer)
        camera=make_camera(cfg);camera.open();camera.warm_up(15)
        print(f"[Startup] 相机就绪={time.perf_counter()-began:.2f}s；YOLO/OCR正在并行初始化",flush=True)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        worker.wait_ready(on_wait=startup_preview)
        print(f"[Startup] 相机+视觉就绪={time.perf_counter()-began:.2f}s",flush=True)
        K,D=np.asarray(camera.K).copy(),np.asarray(camera.D).copy()
        hand_eye=None
        if args.execute:
            hand_eye,mode=load_hand_eye(ROOT,str(cfg["camera"]["type"]).lower())
            if hand_eye is None or mode!="eye_in_hand" or np.asarray(hand_eye).shape!=(4,4) or not np.isfinite(hand_eye).all():
                raise RuntimeError("缺少有效eye_in_hand手眼标定")
        if args.execute:
            robot_cfg=copy.deepcopy(cfg)
            robot_cfg["trajectory"]=options["trajectory"]
            robot=motion.Robot(robot_cfg)
            if before_head is None:robot.start()
            else:robot.start(gripper_posvel=False)
        if before_head is not None:
            state="GRASP_AND_PLACE"
            before_head(robot,camera,worker,window)
            if robot is not None:robot.switch_gripper_to_posvel(cancelled)
        if args.execute:
            head=list(options["startup_view"]["joints_deg"][:6])
            if options["gripper"]["initialize"]:head.append(options["gripper"]["closed_angle_deg"])
            state="HEAD_UP"
            head_began=time.perf_counter()
            robot.head_up(head,options["startup_view"]["duration_s"],cancelled,
                          closed_tolerance=options["gripper"]["closed_tolerance_rad"],
                          wait_for_arrival=options["startup_view"]["wait_for_arrival"])
            print(f"[Startup] 抬头指令阶段={time.perf_counter()-head_began:.2f}s；"
                  f"wait_for_arrival={options['startup_view']['wait_for_arrival']}；后续识别另行计时",flush=True)
            moved()
        else:print("[Preview] 仅相机与OCR；未初始化机器人",flush=True)
        state="WAIT_FLOOR"
        print((f"[Keys] 自动等待稳定{args.floor}楼；空格/Q/ESC取消整轮。" if sequence else "[Keys] 输入楼层+回车；R重新选择；空格停止；Q退出。")+
              f"stage={args.stage}；"
              f"静止稳定图像可复用{settings['stable_snapshot_age_s']:.1f}s",flush=True)
        automatic=args.floor
        recognition_began=time.perf_counter();stable_reported=False
        clock=time.perf_counter();frames=0;fps=0.
        while not quit_motion:
            if sequence and time.perf_counter()-recognition_began>args.auto_floor_timeout_s:
                raise TimeoutError(f"等待稳定{args.floor}楼及有效深度超时；未执行按键动作")
            color,depth=camera.get_frame()
            now=time.perf_counter()
            if color is None or depth is None:
                failures+=1
                if failures>=15:raise RuntimeError("连续15次RGB-D取帧失败")
            else:
                failures=0;frame_id+=1;frames+=1
                if now-clock>=1.:fps=frames/(now-clock);frames=0;clock=now
                pose=np.eye(4) if robot is None else robot.pose()
                current=geometry.Snapshot(frame_id,now,color.copy(),depth.copy(),K,D,pose,generation)
                stable_cache={f:pair for f,pair in stable_cache.items()
                              if 0<=now-pair[0].captured_at<=settings["stable_snapshot_age_s"]}
                received=worker.receive()
                if received is not None:
                    snap,result=received
                    if snap.generation==generation and 0<=now-snap.captured_at<=settings["max_result_age_s"]:
                        tracker.update(result,generation,camera_pose=snap.T_base_end,now=now)
                        last,last_snapshot=result,snap
                        remember_stable(stable_cache,snap,result,tracker,ocr["min_confidence"])
                        if stable_cache and not stable_reported:
                            stable_reported=True
                            print(f"[VisionStable] 识别稳定等待={now-recognition_began:.2f}s（不含抬头及模型初始化）；"
                                  f"楼层={sorted(stable_cache)} 本帧分项ms={result.timings_ms}；"
                                  f"OCR计算={result.tracking.get('ocr_inference',{})}",flush=True)
                if request is not None:
                    try:
                        snap,b,source=resolve_observation(request,last,last_snapshot,stable_cache,tracker,current,settings,ocr["min_confidence"])
                        camera_point,raw_base_point=geometry.point_from_button(snap,b,hand_eye,cfg)
                        base_point=geometry.offset_target_base(raw_base_point,settings["target_offset_base_m"])
                    except ValueError as exc:
                        if now-request["reported"]>1.:
                            print(f"[Floor] {exc}",flush=True);request["reported"]=now
                    else:
                        print(f"[Target] floor={request['floor']} source={source} frame={snap.frame_id} "
                              f"image_age_s={now-snap.captured_at:.3f} selection_wait_s={now-request['at']:.3f} "
                              f"text_center_px={b.text_center_px} rim_candidate_px={b.ring.center} "
                              f"camera_m={camera_point.tolist()} base_m={None if base_point is None else base_point.tolist()}",flush=True)
                        if robot is not None:
                            print("[TargetTransform] "+json.dumps(dict(
                                T_base_end_at_capture=snap.T_base_end.tolist(),
                                base_compensation_m=(cfg.get("calibration") or {}).get("hand_eye_compensation_m",{}),
                                raw_base_m=raw_base_point.tolist(),
                                target_offset_base_m=settings["target_offset_base_m"],
                                corrected_base_m=base_point.tolist(),
                                tip_offset_in_end_m=offset),ensure_ascii=False),flush=True)
                        if not args.execute or args.stage=="observe":
                            request=None;state="DONE"
                            if sequence:
                                print("[Sequence] 预览完成；未执行夹取、搬运或按键",flush=True)
                                normal=True;break
                        else:
                            try:
                                # A complete RGB-D + capture pose is frozen for
                                # this ONE action on the fixed printed panel.
                                level=geometry.level_pose(robot.pose(),offset)
                                if not settings["level_during_approach"]:
                                    state="LEVEL_PITCH_ZERO"
                                    robot.move(level,settings["move_duration_s"],cancelled)
                                else:
                                    print("[Level] 接近前置点时同步调平，到位后再水平向纸面推进",flush=True)
                                state="BUTTON_ACTION"
                                execute_button(robot,base_point,level,settings,offset,args.stage,cancelled)
                                moved();request=None;state="DONE"
                                print("[Pose] actual_rpy_deg="+str(np.degrees(rotation_matrix_to_euler_zyx(robot.pose()[:3,:3])).tolist()),flush=True)
                                if sequence:
                                    print(f"[Sequence] 抓取、放置、{args.floor}楼纸面接触及撤回完成；随后回零退出",flush=True)
                                    normal=True;break
                            except motion.MotionCancelled:
                                if sequence:raise
                                generation+=1;stable_cache.clear();tracker.reset();last=last_snapshot=None
                                stopped=True;request=None;state="STOPPED"
                                print("[Stop] 已取消；R重新选择，Q退出不回零",flush=True)
                            # Never submit the pre-move RGB-D with a post-move pose.
                            continue
                if request is not None and now-request["at"]>settings["floor_wait_s"]:
                    print(f"[Floor] 等待{args.floor}楼重新稳定；未使用旧三维坐标" if sequence else
                          "[Floor] 本次未找到可定位楼层，请重新输入；未使用旧三维坐标",flush=True)
                    request=None;state="WAIT_FLOOR"
                    if sequence:automatic=args.floor
                if worker.pending is None and now>=next_submit:
                    worker.submit(current)
                    next_submit=now+.05
                shown=last if last_snapshot is not None and last_snapshot.generation==generation and now-last_snapshot.captured_at<=settings["max_result_age_s"] else None
                if shown is not None:
                    if rendered_result is not shown:
                        rendered_panel=perception.draw_result(last_snapshot.color,shown,fps,
                            max_age=settings['max_result_age_s'])
                        rendered_result=shown
                    display=rendered_panel.copy()
                else:display=color.copy()
                labels=sorted({b.text for b in shown.buttons if b.status=="STABLE" and b.text is not None}) if shown is not None else []
                memory=[m.label for m in tracker.cached_labels(now=now)]
                positions=sorted({p['label'] for p in shown.tracking.get('position_inference',{}).values()
                                  if p.get('status')=='STABLE_POSITION'}) if shown is not None else []
                age='-' if shown is None else f'{now-last_snapshot.captured_at:.2f}s'
                lines=[f"SDK main.py | {state} | stage={args.stage} | input={typed} | FPS={fps:.1f} image_age={age}",
                       f"OCR>={ocr['min_confidence']:.0%} current={labels} memory={memory} position={positions}",
                       f"AUTO floor={args.floor} | SPACE/Q/Esc=cancel sequence" if sequence else "floor+Enter  R=reset  SPACE=stop  Q/Esc=quit"]
                for i,line in enumerate(lines):
                    cv2.putText(display,line,(8,display.shape[0]-60+i*20),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3)
                    cv2.putText(display,line,(8,display.shape[0]-60+i*20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                cv2.imshow(window,display)
                if automatic is not None and tracker.panel.stable and not stopped:
                    if not sequence or automatic_floor_ready(automatic,last_snapshot,last,tracker,now,generation,
                                                             ocr["min_confidence"],settings["max_result_age_s"]):
                        typed=automatic;automatic=None
                        request=dict(floor=geometry.normalize_floor_id(typed),at=time.perf_counter(),step="locate",reported=0.)
                        typed="";state="LOCATE"
            key=cv2.waitKey(1)&255
            if key in (27,ord("q"),ord("Q")) or not visible():
                if sequence:raise motion.MotionCancelled("用户取消串联流程")
                normal=not stopped;break
            if key==32:
                if sequence:raise motion.MotionCancelled("用户停止串联流程")
                if robot is not None:robot.stop()
                generation+=1;stable_cache.clear();tracker.reset();last=last_snapshot=None
                stopped=True;request=None;state="STOPPED"
            elif not sequence and key in (ord("r"),ord("R")):
                generation+=1;stable_cache.clear()
                tracker.reset();request=None;last=last_snapshot=None;stopped=False;state="WAIT_FLOOR"
            elif not sequence and (ord("0")<=key<=ord("9") or key==ord("-")):typed=(typed+chr(key))[:3]
            elif not sequence and key in (8,127):typed=typed[:-1]
            elif not sequence and key in (10,13) and typed and not stopped:
                request=dict(floor=geometry.normalize_floor_id(typed),at=time.perf_counter(),step="locate",reported=0.)
                typed="";state="LOCATE"
            if sys.stdin.isatty() and select.select([sys.stdin],[],[],0)[0]:
                value=sys.stdin.readline().strip()
                if sequence:
                    if value.lower() in ("q","stop"):raise motion.MotionCancelled("用户取消串联流程")
                    continue
                if value.lower()=="q":normal=not stopped;break
                if value.lower()=="r":
                    generation+=1;stable_cache.clear()
                    tracker.reset();request=None;last=last_snapshot=None;stopped=False;state="WAIT_FLOOR"
                elif value and not stopped:
                    try:floor=geometry.normalize_floor_id(value)
                    except ValueError as exc:print(exc,flush=True)
                    else:request=dict(floor=floor,at=time.perf_counter(),step="locate",reported=0.);state="LOCATE"
        return 0
    except motion.MotionCancelled:
        print("[Stop] 当前流程已取消，不自动回零",flush=True)
        return 130 if sequence else 0
    finally:
        active_error=sys.exc_info()[0] is not None
        cleanup_error=None
        if robot is not None:
            try:robot.close(home=normal)
            except (Exception,KeyboardInterrupt) as exc:
                cleanup_error=exc
                print(f"[Exit] {exc}",file=sys.stderr)
        if camera is not None:
            try:camera.close()
            except Exception as exc:print(f"[Camera close] {exc}",file=sys.stderr)
        cv2.destroyAllWindows()
        if worker is not None:worker.close()
        if cleanup_error is not None and not active_error:
            raise RuntimeError(f"机械臂退出清理失败：{cleanup_error}") from cleanup_error


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"config/default.yaml")
    p.add_argument("--workflow-config",type=Path,default=ROOT/"config/261105_my_elevator.yaml")
    p.add_argument("--mode",choices=("grasp","elevator"))
    g=p.add_mutually_exclusive_group()
    g.add_argument("--execute",action="store_true");g.add_argument("--dry-run",action="store_true")
    g.add_argument("--simulate",action="store_true")
    g.add_argument("--check-bus",action="store_true",help="仅查询电机通信；不加载相机/OCR，不使能或运动")
    p.add_argument("--device");p.add_argument("--ocr-device",choices=("cpu","cuda","auto"))
    p.add_argument("--ocr-preprocessing",choices=("bright_dark","bright"),
                   help="覆盖ocr.preprocessing；bright_dark为亮字+黑字，bright为原亮字流程")
    p.add_argument("--stage",choices=("observe","approach","contact"),default="approach")
    p.add_argument("--floor",help="可选：稳定后自动选择一次的楼层")
    p.add_argument("--standoff-m",type=float)
    p.add_argument("--skip-gripper-init",action="store_true",help="诊断用：只下发六轴抬头，夹爪保持启动时角度")
    args=p.parse_args(argv)
    if args.standoff_m is not None and not .03<=args.standoff_m<=.10:
        p.error("--standoff-m需为0.03～0.10米")
    return args


def main(argv=None):
    args=parse_args(argv)
    try:
        cfg=read_yaml(args.config)
        if args.check_bus:
            return motion.check_bus(cfg)
        if args.mode is None and not args.simulate:
            value=input("选择 1=夹取物体，2=电梯按键，Q=退出：").strip().lower()
            if value in ("q",""):return 0
            args.mode={"1":"grasp","2":"elevator","grasp":"grasp","elevator":"elevator"}.get(value)
            if args.mode is None:raise ValueError("请输入1或2")
        cfg=copy.deepcopy(cfg)
        args.device=str(cfg.get("yolo",{}).get("device","0")) if args.device is None else args.device
        cfg.setdefault("yolo",{})["device"]=args.device
        if args.mode=="grasp" and not args.simulate:
            from . import grasp
            return grasp.run(cfg,args)
        options=load_workflow(args.workflow_config)
        if args.ocr_device is not None:options["ocr"]["device"]=args.ocr_device
        if args.ocr_preprocessing is not None:options["ocr"]["preprocessing"]=args.ocr_preprocessing
        if args.skip_gripper_init:options["gripper"]["initialize"]=False
        if args.standoff_m is not None:options["elevator"]["standoff_m"]=args.standoff_m
        if args.simulate:
            raw_point=np.array([.4,-.1,.4])
            point=geometry.offset_target_base(raw_point,options["elevator"]["target_offset_base_m"])
            current=np.eye(4);current[:3,3]=[.26,0,.192]
            level=geometry.level_pose(current,options["tool"]["tip_offset_in_end_m"])
            poses=geometry.button_poses(point,level,options["tool"]["tip_offset_in_end_m"],
                options["elevator"]["standoff_m"],options["elevator"]["press_travel_m"])
            print(json.dumps(dict(simulation_only=True,pitch_rad=0.,floor=args.floor or "3",
                raw_base_m=raw_point.tolist(),target_offset_base_m=options["elevator"]["target_offset_base_m"],
                corrected_base_m=point.tolist(),
                targets={k:v.tolist() for k,v in poses.items()}),ensure_ascii=False,indent=2))
            return 0
        return run_elevator(cfg,options,args)
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        return 2
