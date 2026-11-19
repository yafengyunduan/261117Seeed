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
from types import SimpleNamespace
import numpy as np
import yaml
from . import detail, geometry, motion, terminal_output, terminal_verbose

ROOT=Path(__file__).resolve().parents[1]
RELOCALIZE_FRAMES=3


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
    robot_control=o.setdefault('robot_control',{})
    scales=motion.joint4_pos_gain_scales(robot_control)
    robot_control.update(scales)
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
        if 'joint4_stage_duration_s' in ready:
            raw=ready['joint4_stage_duration_s']
            message='initial_poses.grasp_ready.joint4_stage_duration_s 必须为有限的非负秒数'
            try: seconds=float(raw)
            except (TypeError,ValueError) as exc: raise ValueError(message) from exc
            if isinstance(raw,bool) or not math.isfinite(seconds) or seconds<0.:
                raise ValueError(message)
            ready['joint4_stage_duration_s']=seconds
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
    if e.get('button_orientation','level') not in ('free','level'):
        raise ValueError('elevator.button_orientation需为free或level')
    for key,lo,hi in (("standoff_m",.03,.25),("move_duration_s",.5,10.),("contact_duration_s",.5,10.),
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
    refinement=e.setdefault("refinement",{})
    if not isinstance(refinement,dict) or not isinstance(refinement.setdefault("enabled",False),bool):
        raise ValueError("elevator.refinement须为映射，enabled须为布尔值")
    if not isinstance(refinement.setdefault("vertical_panel",True),bool):
        raise ValueError("elevator.refinement.vertical_panel须为布尔值")
    if not isinstance(refinement.setdefault("standoff_face_panel",False),bool):
        raise ValueError("elevator.refinement.standoff_face_panel须为布尔值")
    for key,default,lo,hi in (("reacquire_timeout_s",6.,1.,20.),("max_correction_m",.03,.005,.05),
                              ("hold_joint_vlim_rad_s",.15,.05,.30),("standoff_brake_s",0.,0.,10.),
                              ("standoff_hold_vlim_rad_s",.05,.01,.30)):
        value=float(refinement.setdefault(key,default))
        if not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f"elevator.refinement.{key}必须在[{lo},{hi}]")
        refinement[key]=value
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


def robot_runtime_config(cfg,options):
    runtime=copy.deepcopy(cfg)
    runtime['trajectory']=options['trajectory']
    runtime['robot_control']=options.get('robot_control',{})
    return runtime


def _perception_worker(connection,requests,cfg,settings):
    try:
        began=time.perf_counter()
        from .panel_261112 import FastPanelPerception
        settings=dict(settings,workflow_evidence=True)
        detector=FastPanelPerception(cfg,argparse.Namespace(**settings))
        initialized=time.perf_counter()
        detector.warmup(cfg["camera"].get("color_height",720),cfg["camera"].get("color_width",1280))
        with terminal_output(settings.get('verbose',True)):
            report_vision_ready(detector.ocr.backend_info,began,initialized)
            detail(f"[OCR] preprocessing={settings.get('ocr_preprocessing','bright')}；"
                   f"min_confidence={settings.get('ocr_min_score',.80):.2f}；"
                   f"局部预算={settings.get('ocr_buttons_per_update',4)}；"
                   f"刷新={settings.get('ocr_refresh_s',2.):g}s",flush=True)
        connection.send(("ready",))
        panel_generation=None
        while True:
            request=requests.get()
            if request is None:break
            kind,frame_id,at,image,depth,K,*extra=request
            generation=extra[0] if extra else 0
            focus_px=extra[1] if len(extra)>1 else None
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
                result=detector.process(image,captured_at=at,focus_px=focus_px)
            else:raise ValueError(f"Unknown inference task: {kind}")
            connection.send(("result",frame_id,result))
    except (EOFError,BrokenPipeError):
        pass
    except BaseException as exc:
        try:connection.send(("error",f"{type(exc).__name__}: {exc}"))
        except OSError:pass
    finally:connection.close()


class PerceptionError(RuntimeError):
    """Camera inference cannot provide a target for the rest of the task."""


def report_vision_ready(info, began, initialized):
    if terminal_verbose():
        print('[OCR] '+json.dumps(info,ensure_ascii=False),flush=True)
        print(f'[VisionReady] 导入/模型初始化={initialized-began:.2f}s；'
              f'YOLO预热={time.perf_counter()-initialized:.2f}s',flush=True)
    else:
        print(f"[Vision] YOLO/OCR 就绪，OCR={info.get('actual_device', 'unknown')}",flush=True)


class PerceptionWorker:
    """One inference frame in flight; camera display stays responsive."""
    KINDS=('panel','grasp')

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

    def submit(self,snapshot,kind="panel",*,focus_px=None):
        if kind not in self.KINDS:raise ValueError("Unknown inference task")
        if self.pending is None:
            self.requests.put_nowait((kind,snapshot.frame_id,snapshot.captured_at,snapshot.color,
                                     snapshot.depth_mm if kind=="grasp" else None,snapshot.K if kind=="grasp" else None,
                                     snapshot.generation,focus_px))
            self.pending,self.submitted_at=snapshot,time.perf_counter()

    def receive(self):
        if self.connection.poll():
            try:message=self.connection.recv()
            except EOFError as exc:raise PerceptionError("OCR进程连接已关闭") from exc
            if message[0]=="error":raise PerceptionError(message[1])
            if message[0]!="result" or self.pending is None or message[1]!=self.pending.frame_id:
                raise PerceptionError("OCR结果与RGB-D不匹配")
            snapshot,self.pending=self.pending,None
            return snapshot,message[2]
        if not self.process.is_alive():raise PerceptionError("OCR进程退出")
        if self.pending is not None and time.perf_counter()-self.submitted_at>self.timeout:
            raise TimeoutError("OCR推理超时")
        return None

    def close(self):
        self.requests.cancel_join_thread();self.requests.close();self.connection.close()
        if self.process.is_alive():self.process.terminate()
        self.process.join(2.)
        if self.process.is_alive():self.process.kill();self.process.join(1.)


def startup_preview(camera, draw, cancelled, *, interval_s=.20):
    """Throttle loading previews without delaying worker polling or key checks."""
    next_frame=-math.inf
    def poll():
        nonlocal next_frame
        now=time.perf_counter()
        if now>=next_frame:
            next_frame=now+interval_s
            color,_=camera.get_frame()
            if color is not None:draw(color)
        if cancelled():raise motion.MotionCancelled('初始化已取消')
    return poll


def report_reached(robot,phase,target,offset):
    if not terminal_verbose():
        return
    actual=robot.pose()
    tip=actual[:3,3]+actual[:3,:3]@np.asarray(offset)
    intended=target[:3,3]+target[:3,:3]@np.asarray(offset)
    print(f"[Reached:{phase}] model_tip_base_m={tip.tolist()} "
          f"target_tip_base_m={intended.tolist()} model_error_mm={((tip-intended)*1000).round(2).tolist()}；"
          "依赖当前工具偏移和编码器/FK，非独立尺量",flush=True)


class ButtonRefinementError(RuntimeError):
    """The selected button could not be reached from a verified position."""


def _match_reobserved_button(result, label, kind, min_score):
    """Select a uniquely identified rim from this post-move frame."""
    if kind == 'call':
        # The three-frame RGB-D streak below also establishes pair stability.
        if result.state not in ('UNSTABLE','STABLE') or len(result.pair) != 2:
            raise ValueError('等待重新确认外呼上下键')
        button=SimpleNamespace(ring=result.pair[0 if label=='U' else 1])
    else:
        if result.errors or result.panel_bbox is None:
            raise ValueError('等待新画面的数字面板')
        metadata=result.tracking
        sparse=metadata.get('workflow_evidence')
        region_evidence=metadata.get('region_evidence',{})
        display_cache=metadata.get('display_label_cache',{})
        doors=metadata.get('door_audit',{})
        matches=[]
        inferred=geometry.position_floor_indices(result,label)
        for i,b in enumerate(result.buttons):
            key=str(i)
            if doors.get(i,doors.get(key,{})).get('label','UNKNOWN')!='UNKNOWN':
                continue
            evidence=region_evidence.get(key,{})
            text,score=b.text,b.confidence
            if sparse and evidence.get('source')!='CURRENT_OCR':
                # Worker clears its cache on a generation change. A cached
                # label here already has new-view OCR and matching current pixels.
                cached=display_cache.get(key,{})
                text,score=(cached.get('text'),cached.get('confidence')) if (
                    evidence.get('source')=='PIXEL_MATCHED_CACHE' and evidence.get('pixel_match')) else (None,None)
            same=False
            if text is not None and score is not None and math.isfinite(score) and score>=min_score:
                try:same=geometry.normalize_floor_id(text)==label
                except ValueError:pass
            if same or i in inferred:
                matches.append(b)
        if len(matches)!=1:
            raise ValueError(f'楼层{label}对应多个圆环' if matches else f'新画面未读到楼层{label}')
        button=matches[0]
    if button.ring.edge_support < .85:
        raise ValueError('目标圆环尚未清楚定位')
    return button


def report_standoff_feedback(label, status, samples, invalid_samples, elapsed_s):
    if not samples and not invalid_samples:
        return
    report=dict(key=label,status=status,samples=len(samples),invalid_samples=invalid_samples,
                wait_s=round(elapsed_s,3),packet_timestamps=False)
    if samples:
        times,feedback,positions=zip(*samples)
        q=np.array([sample.position_rad for sample in feedback])
        qd=np.array([sample.velocity_rad_s for sample in feedback])
        target=np.array([sample.target_rad for sample in feedback])
        error_deg=np.degrees(q-target)
        xyz=np.array(positions)
        joint_range=np.degrees(np.ptp(q,axis=0)).round(3)
        largest=int(np.argmax(joint_range))
        low_z,high_z=np.argmin(xyz[:,2]),np.argmax(xyz[:,2])
        report.update(host_span_s=round(times[-1]-times[0],3),
                      largest_joint=f'J{largest+1}' if joint_range[largest]>0. else None,
                      joint_range_deg=joint_range.tolist(),
                      hold_target_range_deg=np.degrees(np.ptp(target,axis=0)).round(4).tolist(),
                      final_joint_error_deg=error_deg[-1].round(3).tolist(),
                      raw_peak_rad_s=np.max(np.abs(qd),axis=0).round(4).tolist(),
                      fk_axis_range_mm=(np.ptp(xyz,axis=0)*1000.).round(2).tolist(),
                      joint_delta_low_to_high_z_deg=np.degrees(q[high_z]-q[low_z]).round(3).tolist())
        if terminal_verbose():
            j4_error=error_deg[:,3]
            report['joint4_error_range_deg']=[round(float(j4_error.min()),3),round(float(j4_error.max()),3)]
            torques=[sample.joint4_torque_nm for sample in feedback if sample.joint4_torque_nm is not None]
            if torques:
                report['joint4_torque_range_nm']=[round(min(torques),3),round(max(torques),3)]
    if terminal_verbose():
        print('[StandoffFeedback] '+json.dumps(report,ensure_ascii=False),flush=True)
    elif samples and status!='CANCELLED':
        print(f"[StandoffFeedback] {status} {label}，J4={report['joint_range_deg'][3]:.3f}°，"
              f"Z={report['fk_axis_range_mm'][2]:.2f} mm（{report['samples']} 帧，{elapsed_s:.2f} 秒）",flush=True)
    elif invalid_samples:
        print(f'[StandoffFeedback] {status} {label}，缺失反馈 {invalid_samples} 帧',flush=True)


def reacquire_button(capture, worker, kind, label, options, hand_eye, cfg, cancelled, *,
                     generation, settle_s=.3, max_age_s=1.5, hint_point_base=None):
    """Use the latest three matching RGB-D observations from one stationary view."""
    settings=options['elevator']['refinement']
    min_score=options['ocr']['min_confidence']
    started=time.perf_counter()
    timeout=settings['reacquire_timeout_s']
    deadline=started+timeout
    anchor=None
    pose_low=pose_high=None
    samples=[]
    feedback_samples=[]
    invalid_feedback=0
    last_frame=-1
    reason='等待新的 RGB-D 与静止位姿'
    peak_m=peak_deg=0.
    peak_rim_shift=None
    detail(f'[Relocalize] key={label} timeout_s={timeout:g} settle_s={settle_s:g}',flush=True)
    while time.perf_counter()<deadline:
        if cancelled():
            report_standoff_feedback(label,'CANCELLED',feedback_samples,invalid_feedback,time.perf_counter()-started)
            raise motion.MotionCancelled('用户停止重新定位')
        try:
            current=capture()
        except motion.ArmFeedbackError as exc:
            invalid_feedback+=1
            samples.clear()
            anchor=None
            reason=f'等待有效的六轴反馈（{exc}）'
            continue
        if current is not None and current.captured_at>=deadline:
            break
        if (current is None or current.generation!=generation
                or current.captured_at<started):
            samples.clear()
            anchor=None
            reason='缺少有效的新 RGB-D'
            continue
        position=current.T_base_end[:3,3]
        if current.arm_feedback is not None:
            feedback_samples.append((current.captured_at,current.arm_feedback,position))
        if anchor is None:
            anchor=current
            pose_low=pose_high=position
        pose_low,pose_high=np.minimum(pose_low,position),np.maximum(pose_high,position)
        translation=np.linalg.norm(pose_high-pose_low)
        _,angle=geometry.pose_difference(anchor.T_base_end,current.T_base_end)
        peak_m,peak_deg=max(peak_m,translation),max(peak_deg,np.degrees(angle))
        if translation>.002 or angle>np.radians(.5):
            anchor=current
            pose_low=pose_high=position
            samples.clear()
            reason=f'末端还未停稳（该窗口位移范围 {translation*1000:.1f} mm / {np.degrees(angle):.2f}°）'
        stationary_at=anchor.captured_at+settle_s
        received=worker.receive()
        if received is not None:
            snapshot,result=received
            if (snapshot.generation==generation and snapshot.captured_at>=stationary_at
                    and snapshot.frame_id>last_frame):
                last_frame=snapshot.frame_id
                try:
                    age=current.captured_at-snapshot.captured_at
                    if not 0 <= age <= max_age_s:
                        raise ValueError(f'新帧识别结果已过时（{age:.2f}s）')
                    button=_match_reobserved_button(result,label,kind,min_score)
                    point_camera,point_base=geometry.point_from_button(snapshot,button,hand_eye,cfg,min_depth_samples=9)
                    if point_base is None or not np.isfinite(point_base).all():
                        raise ValueError('未取得有效的按钮 base 坐标')
                except ValueError as exc:
                    reason=str(exc);samples.clear()
                else:
                    u,v=button.ring.center
                    spread=max((np.linalg.norm(point_base-old_point) for old_point,_,_ in samples),default=0.)
                    rim_shift=max((math.hypot(u-old_u,v-old_v) for _,old_u,old_v in samples),default=0.)
                    if samples:
                        peak_rim_shift=max(peak_rim_shift or 0.,rim_shift)
                    if spread>.008:
                        samples.clear()
                    samples.append((point_base,u,v))
                    reason=(f'同键三维估计跳动 {spread*1000:.1f} mm，重新累计' if spread>.008
                            else f'同键稳定 {len(samples)}/{RELOCALIZE_FRAMES}')
                    detail(f'[Relocalize] key={label} stable={len(samples)}/{RELOCALIZE_FRAMES} '
                           f'generation={generation} frame={snapshot.frame_id} raw_base_m={point_base.tolist()} '
                           f'spread_mm={spread*1000:.1f} rim_shift_px={rim_shift:.1f} '
                           f'window_end_span_mm={translation*1000:.1f} window_end_turn_deg={np.degrees(angle):.2f} '
                           f'wait_s={time.perf_counter()-started:.2f}',flush=True)
                    if len(samples)==RELOCALIZE_FRAMES:
                        report_standoff_feedback(label,'STABLE',feedback_samples,invalid_feedback,time.perf_counter()-started)
                        return geometry.ButtonPoint(snapshot,button,point_camera,point_base)
        if worker.pending is None and current.captured_at>=stationary_at:
            focus_px=None
            if kind=='panel' and hint_point_base is not None:
                focus_px=geometry.project_button_hint(current,hint_point_base,hand_eye,cfg)
            hints={} if focus_px is None else {'focus_px':focus_px}
            worker.submit(current,kind=kind,**hints)
    report_standoff_feedback(label,'TIMEOUT',feedback_samples,invalid_feedback,time.perf_counter()-started)
    rim_summary='' if peak_rim_shift is None else f'；同键圆环最大像素位移 {peak_rim_shift:.1f} px'
    raise ButtonRefinementError(f'同键重新定位超时（设置 {timeout:g}s）：{reason}；'
                                f'反馈窗口最大位姿变化 {peak_m*1000:.1f} mm / {peak_deg:.2f}°'+rim_summary)


def refined_button_poses(point, outward, orientation, settings, offset):
    if settings['refinement'].get('standoff_face_panel',False):
        orientation=geometry.panel_facing_pose(outward)
    return geometry.button_poses(point,orientation,offset,settings['standoff_m'],settings['press_travel_m'],
                                 forward_base=-np.asarray(outward,float))


def simulated_button_poses(point, orientation, settings, offset):
    if settings['refinement']['enabled']:
        angle=np.radians(20.)
        return refined_button_poses(point,[-np.cos(angle),np.sin(angle),0.],orientation,settings,offset)
    return geometry.button_poses(point,orientation,offset,settings['standoff_m'],settings['press_travel_m'])


def _dwell_before_retreat(robot,seconds,cancelled):
    if seconds<=0.:
        return
    print(f'[Button] 接触位保持 {seconds:g} 秒后回撤。',flush=True)
    deadline=time.monotonic()+seconds
    while True:
        if cancelled():
            robot.stop()
            raise motion.MotionCancelled('用户停止')
        remaining=deadline-time.monotonic()
        if remaining<=0.:
            return
        time.sleep(min(.02,remaining))


def execute_refined_button(robot, surface, settings, offset, stage, refresh, cancelled=lambda:False, *,
                           position_only=False, target_offset=(0.,0.,0.), remember_retreat=lambda pose:None):
    """From one standoff, resolve the same button once and press the fresh point."""
    if stage not in ('approach','contact'):
        raise ValueError('此动作仅用于approach/contact')
    offset=np.asarray(offset,float)
    start=robot.pose()
    orientation=start if position_only else geometry.level_pose(start,offset)
    refinement=settings['refinement']
    face_panel=refinement.get('standoff_face_panel',False)
    outward=surface.outward_base
    def plan(measured):
        point=geometry.offset_target_base(measured.point_base,target_offset)
        return refined_button_poses(point,outward,orientation,settings,offset)

    poses=plan(surface)
    remember_retreat(start)

    def move(name,duration):
        target=poses[name]
        # Wait for quiet feedback only at non-contact waypoints. Do not extend
        # the dwell against the panel when feedback is unsettled.
        arrival={}
        hold_vlim=refinement.get('hold_joint_vlim_rad_s',.15)
        if name=='approach':
            arrival['settle']=motion.SettledArrival(name,hold_vlim,
                brake_tail_s=refinement.get('standoff_brake_s',0.),
                stationary_vlim_rad_s=refinement.get('standoff_hold_vlim_rad_s',.05),lock_after_settle=True)
        elif name=='retreat':
            arrival['settle']=motion.SettledArrival(name,hold_vlim,hold_measured=True)
        if position_only and not (name=='approach' and face_panel):
            poses[name]=robot.move_position(geometry.tip_position(target,offset),duration,cancelled,tip_offset=offset,
                                           **arrival)
        else:
            robot.move(target,duration,cancelled,**arrival)
        report_reached(robot,name,poses[name],offset)

    def front_clearance():
        error=geometry.tip_position(robot.pose(),offset)-geometry.tip_position(poses['surface'],offset)
        gap=float(error @ outward)
        lateral=np.linalg.norm(error-gap*outward)
        if gap<.02:
            raise ButtonRefinementError(f'末端在估计面板外侧的余量不足：{gap*1000:.1f}mm')
        return gap,lateral

    front_clearance()
    detail(f"[Button:STANDOFF] standoff_m={settings['standoff_m']:.3f} face_panel={face_panel} "
           f"vertical_panel={refinement['vertical_panel']} outward_base={outward.tolist()} "
           f'measured_outward_base={surface.measured_outward_base.tolist()} plane_inliers={surface.plane_inliers} '
           f'plane_rms_mm={surface.plane_rms_m*1000:.2f}',flush=True)
    move('approach',settings['move_duration_s'])
    standoff_pose=robot.pose()
    remember_retreat(standoff_pose)
    facing='，末端水平朝向面板' if face_panel else ''
    print(f'[Button] 已到前置点{facing}，复核同键 {RELOCALIZE_FRAMES} 帧。',flush=True)
    try:
        new=refresh()
    except motion.MotionCancelled:
        robot.stop()
        raise
    if (new.snapshot.generation<=surface.snapshot.generation or new.snapshot.frame_id<=surface.snapshot.frame_id
            or new.snapshot.captured_at<=surface.snapshot.captured_at):
        raise ButtonRefinementError('拒绝使用运动之前的 RGB-D')
    poses=plan(new)
    # Withdraw to the standoff we actually reached, without another view change.
    poses['approach']=poses['retreat']=standoff_pose
    gap,lateral=front_clearance()
    detail(f"[Button:TARGET] gap_mm={gap*1000:.1f} lateral_mm={lateral*1000:.1f} "
           f"tip_base_m={geometry.tip_position(poses['surface'],offset).tolist()}",flush=True)
    if lateral>refinement['max_correction_m'] or abs(gap-settings['standoff_m'])>refinement['max_correction_m']+.01:
        raise ButtonRefinementError('三帧已确认同键，但修正量超出当前前置点的范围')
    if stage=='approach':
        print('[Button] 完成精定位验证；未接触面板',flush=True)
        return poses
    def verify(name):
        error=geometry.tip_position(robot.pose(),offset)-geometry.tip_position(poses[name],offset)
        normal=float(error @ outward)
        sideways=float(np.linalg.norm(error-normal*outward))
        passed=np.linalg.norm(error)<=.008
        printer=detail if passed else print
        printer(f'[ContactCheck] target={name} normal_error_mm={normal*1000:.1f} lateral_error_mm={sideways*1000:.1f} '
                f"status={'GEOMETRY_CONFIRMED' if passed else 'GEOMETRY_OUT_OF_TOLERANCE'} "
                'physical_contact=UNVERIFIED activation=UNVERIFIED',flush=True)
        return passed

    # Keep the old time per metre when the contact leg exceeds 10 cm.
    time_scale=max(1.,float(np.hypot(gap,lateral))/.10)
    move('surface',settings['contact_duration_s']*time_scale)
    verified=verify('surface')
    if verified and settings['press_travel_m']>0.:
        move('press',settings['contact_duration_s'])
        verified=verify('press')
    if verified:
        _dwell_before_retreat(robot,settings['dwell_s'],cancelled)
    move('retreat',settings['retreat_duration_s']*time_scale)
    if not verified:
        raise ButtonRefinementError('接触位反馈超出8mm；已回撤，本次未确认到位')
    print('[Button] 已到估计表面并撤回。',flush=True)
    return poses


def execute_button(robot,point,level,settings,offset,stage,cancelled=lambda:False, *, position_only=False):
    """main.py-like pregrasp/grasp sequence, now approach/surface/retreat."""
    if stage not in ("approach","contact"):
        raise ValueError("此动作仅用于approach/contact")
    direction=geometry.horizontal_approach_direction(level) if position_only else None
    poses=geometry.button_poses(point,level,offset,settings["standoff_m"],settings["press_travel_m"],forward_base=direction)
    mode='orientation=FREE_POSITION_IK' if position_only else '目标roll/pitch=0/0rad'
    detail(f"[Button] surface_base_m={np.asarray(point).tolist()}；{mode}；stage={stage}",flush=True)
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
    _dwell_before_retreat(robot,settings['dwell_s'],cancelled)
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


def finish_elevator(robot,camera,worker,home,*,restore_terminal=None):
    """Release vision, finish the arm session, and return to the caller."""
    import cv2
    active=sys.exc_info()[1]
    if active is not None and robot is not None:
        detail(f'[Exit] {type(active).__name__}: {active}',file=sys.stderr,flush=True)
    errors=[]
    def clean(name,call):
        try:
            call()
        except (Exception,KeyboardInterrupt) as exc:
            errors.append(f'{name}: {exc}')
            print(f'[Cleanup] {errors[-1]}',file=sys.stderr,flush=True)
    if restore_terminal is not None:clean('terminal',restore_terminal)
    if camera is not None:clean('camera',camera.close)
    if worker is not None:clean('worker',worker.close)
    clean('windows',cv2.destroyAllWindows)
    if robot is not None:clean('robot',lambda:robot.finish_task(home=home))
    if errors and active is None:raise RuntimeError('退出清理未全部确认：'+'; '.join(errors))


def run_elevator(cfg,options,args,*,before_head=None):
    import cv2
    from drivers.camera import CameraFrameError, make_camera
    from utils.camera_utils import load_hand_eye
    from utils.transforms import rotation_matrix_to_euler_zyx
    from . import perception
    worker=camera=robot=None
    sequence=bool(getattr(args,"sequence_mode",False))
    tracker=_tracker(options);settings=options["elevator"];offset=options["tool"]["tip_offset_in_end_m"]
    position_only=settings.get('button_orientation','level')=='free'
    window="Elevator - main.py SDK workflow";display=None;rendered_panel=rendered_result=None
    state="INITIALIZING";typed="";request=None;last=None;last_snapshot=None;stable_cache={}
    generation=0;frame_id=0;next_submit=0.;failures=0;home=False;stopped=False;quit_motion=False
    def visible():
        try:return cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1
        except cv2.error:return False
    def cancelled():
        nonlocal quit_motion,home
        if display is not None:
            moving=display.copy()
            cv2.putText(moving,state+" | SPACE=stop Q=home/exit",(8,32),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.imshow(window,moving)
        key=cv2.waitKey(1)&255
        if key in (27,ord("q"),ord("Q")) or not visible():
            quit_motion=True;home=key in (ord("q"),ord("Q"));return True
        return key==32
    def moved(*, warm_up=True):
        nonlocal generation,last,last_snapshot,rendered_panel,rendered_result
        generation+=1;tracker.reset();last=last_snapshot=None;stable_cache.clear()
        rendered_panel=rendered_result=None
        if warm_up:camera.warm_up(3)
    def capture_for_alignment():
        nonlocal frame_id,display
        color,depth=camera.get_frame()
        if color is None or depth is None:return None
        frame_id+=1
        display=color.copy()
        pose,feedback=robot.pose_with_feedback()
        return geometry.Snapshot(frame_id,time.perf_counter(),color.copy(),depth.copy(),K,D,pose,generation,
                                 arm_feedback=feedback)
    def show_startup(color):
        nonlocal display
        display=color.copy()
    try:
        began=time.perf_counter()
        ocr=options["ocr"]
        infer=dict(device=args.device,roi=None,max_side=1200,geometry_only=False,ocr_threads=4,verbose=terminal_verbose(),
            ocr_min_score=float(ocr["min_confidence"]),ocr_device=ocr["device"],
            ocr_preprocessing=ocr["preprocessing"],
            ocr_gpu_index=int(ocr["gpu_index"]),ocr_model_dir=ROOT/ocr["model_dir"],
            ocr_buttons_per_update=ocr.get('buttons_per_update',4),ocr_refresh_s=ocr.get('refresh_s',2.))
        worker=PerceptionWorker(cfg,infer)
        camera_began=time.perf_counter()
        camera=make_camera(cfg);camera.open();camera.warm_up(15)
        camera_ready=time.perf_counter()
        detail(f"[Startup] camera_open_and_warm_s={camera_ready-camera_began:.2f}；同期加载YOLO/OCR",flush=True)
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        worker.wait_ready(on_wait=startup_preview(camera,show_startup,cancelled))
        detail(f"[Startup] vision_remaining_s={time.perf_counter()-camera_ready:.2f} "
              f"vision_and_camera_s={time.perf_counter()-began:.2f}",flush=True)
        K,D=np.asarray(camera.K).copy(),np.asarray(camera.D).copy()
        hand_eye=None
        if args.execute:
            hand_eye,mode=load_hand_eye(ROOT,str(cfg["camera"]["type"]).lower())
            if hand_eye is None or mode!="eye_in_hand" or np.asarray(hand_eye).shape!=(4,4) or not np.isfinite(hand_eye).all():
                raise RuntimeError("缺少有效eye_in_hand手眼标定")
        if args.execute:
            robot=motion.Robot(robot_runtime_config(cfg,options))
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
            detail(f"[Startup] 抬头指令阶段={time.perf_counter()-head_began:.2f}s；"
                  f"wait_for_arrival={options['startup_view']['wait_for_arrival']}；后续识别另行计时",flush=True)
            moved()
        else:print("[Preview] 仅相机与OCR；未初始化机器人",flush=True)
        state="WAIT_FLOOR"
        print(f"[Keys] 自动等待稳定{args.floor}楼，Q 回零退出。" if sequence else
              "[Keys] 输入楼层并回车，R 重选，空格停止，Q 回零退出。",flush=True)
        detail(f"[Button] stage={args.stage}，稳定图像可复用 {settings['stable_snapshot_age_s']:.1f}s",flush=True)
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
                if failures>=15:raise CameraFrameError("连续15次RGB-D取帧失败")
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
                            detail(f"[VisionStable] 识别稳定等待={now-recognition_began:.2f}s（不含抬头及模型初始化）；"
                                  f"楼层={sorted(stable_cache)} 本帧分项ms={result.timings_ms}；"
                                  f"OCR计算={result.tracking.get('ocr_inference',{})}",flush=True)
                if request is not None:
                    surface=None
                    try:
                        snap,b,source=resolve_observation(request,last,last_snapshot,stable_cache,tracker,current,settings,ocr["min_confidence"])
                        if robot is not None and args.stage!='observe' and settings.get('refinement',{}).get('enabled',False):
                            surface=geometry.measure_button_surface(snap,b,hand_eye,cfg,
                                                                    vertical_panel=settings['refinement']['vertical_panel'])
                            camera_point,raw_base_point=surface.point_camera,surface.point_base
                        else:
                            camera_point,raw_base_point=geometry.point_from_button(snap,b,hand_eye,cfg)
                        base_point=geometry.offset_target_base(raw_base_point,settings["target_offset_base_m"])
                    except ValueError as exc:
                        if now-request["reported"]>1.:
                            if not request['reported'] or terminal_verbose():print(f"[Floor] {exc}",flush=True)
                            request["reported"]=now
                    else:
                        if terminal_verbose():
                            print(f"[Target] floor={request['floor']} source={source} frame={snap.frame_id} "
                                  f"image_age_s={now-snap.captured_at:.3f} selection_wait_s={now-request['at']:.3f} "
                                  f"text_center_px={b.text_center_px} rim_candidate_px={b.ring.center} "
                                  f"camera_m={camera_point.tolist()} base_m={None if base_point is None else base_point.tolist()}",flush=True)
                        elif not args.execute or args.stage=='observe':
                            point=base_point if base_point is not None else camera_point
                            frame='base' if base_point is not None else 'camera'
                            print(f"[Target] {request['floor']} 楼，{frame} XYZ={np.asarray(point).round(4).tolist()} m",flush=True)
                        else:
                            print(f"[Target] 已定位 {request['floor']} 楼。",flush=True)
                        if robot is not None and terminal_verbose():
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
                                break
                        else:
                            try:
                                if surface is not None:
                                    selected=request['floor']
                                    def refresh():
                                        nonlocal state
                                        state='RELOCALIZE_3_FRAMES'
                                        moved(warm_up=False)
                                        return reacquire_button(capture_for_alignment,worker,'panel',selected,options,
                                            hand_eye,cfg,cancelled,generation=generation,
                                            hint_point_base=surface.point_base)
                                    state='BUTTON_ACTION'
                                    execute_refined_button(robot,surface,settings,offset,args.stage,refresh,cancelled,
                                                           position_only=position_only,target_offset=settings['target_offset_base_m'])
                                else:
                                    orientation=robot.pose()
                                    if not position_only:
                                        orientation=geometry.level_pose(orientation,offset)
                                        if not settings['level_during_approach']:
                                            state='LEVEL_PITCH_ZERO'
                                            robot.move(orientation,settings['move_duration_s'],cancelled)
                                        else:
                                            detail('[Level] 接近前置点时同步调平，到位后再水平向纸面推进',flush=True)
                                    state="BUTTON_ACTION"
                                    execute_button(robot,base_point,orientation,settings,offset,args.stage,cancelled,
                                                   position_only=position_only)
                                moved();request=None;state="DONE"
                                if terminal_verbose():
                                    print("[Pose] actual_rpy_deg="+str(np.degrees(rotation_matrix_to_euler_zyx(robot.pose()[:3,:3])).tolist()),flush=True)
                                if sequence:
                                    print(f"[Sequence] 抓取、放置、{args.floor}楼纸面接触及撤回完成；随后回零退出",flush=True)
                                    home=True;break
                            except motion.MotionCancelled:
                                if sequence or quit_motion:raise
                                generation+=1;stable_cache.clear();tracker.reset();last=last_snapshot=None
                                stopped=True;request=None;state="STOPPED"
                                print("[Stop] 已取消；R重新选择，Q回零结束任务",flush=True)
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
                home=key in (ord("q"),ord("Q"))
                if sequence:raise motion.MotionCancelled("用户取消串联流程")
                break
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
                    if value.lower() in ("q","stop"):
                        home=value.lower()=="q"
                        raise motion.MotionCancelled("用户取消串联流程")
                    continue
                if value.lower()=="q":home=True;break
                if value.lower()=="r":
                    generation+=1;stable_cache.clear()
                    tracker.reset();request=None;last=last_snapshot=None;stopped=False;state="WAIT_FLOOR"
                elif value and not stopped:
                    try:floor=geometry.normalize_floor_id(value)
                    except ValueError as exc:print(exc,flush=True)
                    else:request=dict(floor=floor,at=time.perf_counter(),step="locate",reported=0.);state="LOCATE"
        return 0
    except (ButtonRefinementError,PerceptionError,motion.TrajectoryPlanningError,motion.TrajectorySendError,
            TimeoutError,CameraFrameError):
        home=True
        raise
    except motion.MotionCancelled:
        print("[Stop] 回零后退出" if home else "[Stop] 停止当前动作后退出",flush=True)
        return 0 if home or not sequence else 130
    finally:
        finish_elevator(robot,camera,worker,home)


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
    p.add_argument("--verbose",action="store_true",help="输出详细识别、轨迹和反馈日志")
    p.add_argument("--ocr-preprocessing",choices=("bright_dark","bright"),
                   help="覆盖ocr.preprocessing；bright_dark为亮字+黑字，bright为原亮字流程")
    p.add_argument("--stage",choices=("observe","approach","contact"),default="approach")
    p.add_argument("--floor",help="可选：稳定后自动选择一次的楼层")
    p.add_argument("--standoff-m",type=float,help="按钮估计表面前的停靠距离，默认读取 elevator.standoff_m")
    p.add_argument("--skip-gripper-init",action="store_true",help="诊断用：只下发六轴抬头，夹爪保持启动时角度")
    args=p.parse_args(argv)
    if args.standoff_m is not None and not .03<=args.standoff_m<=.25:
        p.error("--standoff-m需为0.03～0.25米")
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
            with terminal_output(args.verbose):
                return grasp.run(cfg,args)
        options=load_workflow(args.workflow_config)
        if args.ocr_device is not None:options["ocr"]["device"]=args.ocr_device
        if args.ocr_preprocessing is not None:options["ocr"]["preprocessing"]=args.ocr_preprocessing
        if args.skip_gripper_init:options["gripper"]["initialize"]=False
        if args.standoff_m is not None:options["elevator"]["standoff_m"]=args.standoff_m
        if args.simulate:
            settings=options['elevator']
            position_only=settings.get('button_orientation','level')=='free'
            raw_point=np.array([.4,-.1,.4])
            point=geometry.offset_target_base(raw_point,settings['target_offset_base_m'])
            current=np.eye(4);current[:3,3]=[.26,0,.192]
            poses=simulated_button_poses(point,current,settings,options['tool']['tip_offset_in_end_m'])
            print(json.dumps(dict(simulation_only=True,pitch_rad=None if position_only else 0.,floor=args.floor or "3",
                raw_base_m=raw_point.tolist(),target_offset_base_m=settings['target_offset_base_m'],
                corrected_base_m=point.tolist(),refinement=settings['refinement'],position_only_ik=position_only,
                standoff_m=settings['standoff_m'],relocalize_frames=RELOCALIZE_FRAMES if settings['refinement']['enabled'] else 0,
                move_duration_s=settings['move_duration_s'],dwell_s=settings['dwell_s'],
                targets={k:v.tolist() for k,v in poses.items()}),ensure_ascii=False,indent=2))
            return 0
        with terminal_output(args.verbose):
            return run_elevator(cfg,options,args)
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        return 2
