"""Grasp execution delegates directly to the protected original scripts/main.py."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
from drivers.camera import make_camera
from utils.yolo_utils import load_yolo
from utils.ordinary_grasp import estimate_grasps,select_best_grasp,draw_grasp
from . import detail, terminal_verbose
from elevator.motion import sdk_types

ROOT=Path(__file__).resolve().parents[1]


def original_main():
    from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath
    ensure_rebot_sdk_in_syspath()
    spec=importlib.util.spec_from_file_location("_original_grasp_main",ROOT/"scripts/main.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cfg,args):
    if args.execute:
        module=original_main()
        # Only this in-memory module is changed; the original file stays intact.
        module.load_config=lambda path:cfg
        arm_type, driver_type = sdk_types((cfg.get("robot") or {}).get("repo_root"))
        arms = []
        def make_arm(*a, **kw):
            arm = arm_type(*a, **kw)
            arms.append(arm)
            return arm
        module.RebotArm = make_arm
        module.GraspDriver = driver_type
        argv=sys.argv
        try:
            sys.argv=[str(ROOT/"scripts/main.py"),"--config",str(args.config)]
            return module.main()
        finally:
            sys.argv=argv
            # Original main may swallow end()/disconnect() errors, or be interrupted
            # during home. Cleanup is independent of successful grasp/home completion.
            active_error = sys.exc_info()[0] is not None
            failures = []
            for arm in arms:
                try:
                    arm.disconnect()
                except BaseException as exc:
                    failures.append(str(exc))
                reported = getattr(arm, "_application_shutdown_error", "")
                if reported and reported not in failures:
                    failures.append(reported)
            if failures:
                message = "夹取退出清理未全部确认：" + "; ".join(failures)
                if active_error:
                    print(f"[Exit] {message}", flush=True)
                else:
                    raise RuntimeError(message)
    return preview(cfg,args)


def ready_settings(cfg):
    return cfg.get("robot",{}).get("ready_pose",dict(x=.25,y=0.,z=.35,roll=0.,pitch=1.2,yaw=0.,duration=3.))


def placement_pose(cfg,xyz,rpy_deg=None):
    """Base XYZ; preserve ready roll/pitch and face the placement location."""
    from utils.transforms import pose6d_to_mat4
    xyz=np.asarray(xyz,float)
    if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError("放置坐标需要三个有限米值")
    ready=ready_settings(cfg)
    rpy=np.array([ready.get("roll",0.),ready.get("pitch",1.2),np.arctan2(xyz[1],xyz[0])]) if rpy_deg is None else np.radians(rpy_deg)
    if rpy.shape!=(3,) or not np.isfinite(rpy).all():raise ValueError("放置朝向需要三个有限度数")
    return pose6d_to_mat4(*xyz,*rpy)


def original_motion_adapter(robot,cancelled):
    """Run original grasp steps with cancellable, blocking SDK moves."""
    from utils.transforms import pose6d_to_mat4
    module=original_main()
    def move_to_traj(*values,duration,**kwargs):
        pose=values or tuple(kwargs[k] for k in ("x","y","z","roll","pitch","yaw"))
        robot.move(pose6d_to_mat4(*pose),duration,cancelled)
        return True
    # Robot.move already waits for the original SDK sender and final feedback.
    # This private module instance changes no original source or other mode.
    module._wait_motion=lambda *a,**kw:None
    return module,SimpleNamespace(move_to_traj=move_to_traj)


def enter_ready_pose(robot,cfg,cancelled):
    """Move into the startup grasp pose, optionally delaying the fourth axis."""
    from utils.transforms import pose6d_to_mat4
    module,controller=original_motion_adapter(robot,cancelled)
    ready=ready_settings(cfg)
    joint4_seconds=ready.get('joint4_stage_duration_s',0.)
    if joint4_seconds>0.:
        target=pose6d_to_mat4(*(ready[key] for key in ('x','y','z','roll','pitch','yaw')))
        robot.move_ready_in_stages(target,ready['duration'],joint4_seconds,cancelled)
    else:
        module._move_ready(controller,ready)
    return module


def pick_and_place(robot,grasp6d,pre6d,cfg,place_target,place_duration=3.,release_wait=.5,cancelled=lambda:False):
    from .motion import MotionCancelled
    module,controller=original_motion_adapter(robot,cancelled)
    driver=robot.driver
    previous=getattr(driver,"_application_cancelled",None)
    driver._application_cancelled=cancelled
    def open_gripper():
        if cancelled():raise MotionCancelled("用户取消松爪")
        driver.open_gripper()
        if not driver._position_done():raise TimeoutError("夹爪张开未完成，停止串联流程")
    try:
        commands=SimpleNamespace(open_gripper=open_gripper,grasp=driver.grasp)
        if not module._execute_grasp(controller,commands,grasp6d,pre6d,ready_settings(cfg),False):
            print("[Sequence] 原驱动未确认抓取成功；留在夹取阶段，可重新按G",flush=True)
            return False
        if cancelled():raise MotionCancelled("用户取消搬运")
        detail(f"[Sequence:PLACE] end_target_base_m={place_target[:3,3].tolist()}；保持抓取力，先到位再松爪",flush=True)
        robot.move(place_target,place_duration,cancelled)
        open_gripper()
        until=time.monotonic()+release_wait
        while time.monotonic()<until:
            if cancelled():raise MotionCancelled("用户取消串联流程")
            time.sleep(.02)
        print("[Sequence] 已到放置点并张开夹爪。",flush=True)
        return True
    finally:driver._application_cancelled=previous


def grasp_then_place(robot,camera,worker,window,cfg,args):
    """One G-triggered original grasp; one shared camera/model/robot session."""
    from . import geometry,motion
    from utils.camera_utils import compose_cam_to_base_transform,load_hand_eye
    from utils.transforms import transform_grasp_pose_to_base,rotation_matrix_to_euler_zyx
    display=None
    def cancelled():
        if display is not None:cv2.imshow(window,display)
        key=cv2.waitKey(1)&255
        try:visible=cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1
        except cv2.error:visible=False
        return key in (27,32,ord("q"),ord("Q")) or not visible
    place=placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg)
    detail(f"[Sequence] 放置end_link base XYZ={place[:3,3].tolist()}m，"
           f"rpy_deg={np.degrees(rotation_matrix_to_euler_zyx(place[:3,:3])).round(3).tolist()}",flush=True)
    if robot is not None:
        module=enter_ready_pose(robot,cfg,cancelled)
        hand_eye,mode=load_hand_eye(ROOT,str(cfg["camera"]["type"]).lower())
        if hand_eye is None or mode!="eye_in_hand":raise RuntimeError("抓取缺少有效手眼标定")
    else:
        module=original_main();hand_eye=None
        print("[Sequence:PREVIEW] 仅相机识别；不抓取、搬运、松爪或抬头",flush=True)
    print(f"[Sequence] G 抓放后按 {args.floor} 楼。R 重选，空格/Q/ESC 取消。",flush=True)
    K,D=np.asarray(camera.K).copy(),np.asarray(camera.D).copy()
    every=max(1,int(cfg.get("grasp_pipeline",{}).get("infer_every_live",3)))
    pregrasp=float(cfg.get("grasp_pipeline",{}).get("grasp",{}).get("pregrasp_offset_m",.08))
    frame=failures=0;grasps=[];want_grasp=False;selected_id=None
    while True:
        color,depth=camera.get_frame()
        if color is None or depth is None:
            failures+=1
            if cancelled():raise motion.MotionCancelled("取消夹取阶段")
            if failures>=15:raise RuntimeError("连续15次抓取RGB-D取帧失败")
            continue
        failures=0;frame+=1
        pose=np.eye(4) if robot is None else robot.pose()
        current=geometry.Snapshot(frame,time.perf_counter(),color.copy(),depth.copy(),K,D,pose,-1)
        received=worker.receive()
        if received is not None:
            snap,grasps=received
            if snap.frame_id==selected_id:
                selected_id=None;want_grasp=False
                best=select_best_grasp(grasps)
                if best is None:
                    print("[Sequence] 本次没有有效抓取目标，未进入搬运或电梯阶段",flush=True)
                else:
                    if terminal_verbose():module._print_best_grasp(best)
                    display=module._render_display(snap.color,grasps,best,"GRASP -> PLACE | SPACE/Q=cancel")
                    if robot is None:
                        print(f"[Sequence:PREVIEW] grasp_camera_m={best.position.tolist()}；后续仅预览{args.floor}楼识别",flush=True)
                        return
                    T=compose_cam_to_base_transform(snap.T_base_end,hand_eye,cfg)
                    grasp6d,pre6d=transform_grasp_pose_to_base(best.position,best.tcp_rotation,T,pregrasp)
                    detail(f"[Sequence:GRASP_TARGET] frame={snap.frame_id} image_age_s={time.perf_counter()-snap.captured_at:.3f}；"
                           "使用按G后的同帧RGB-D和拍摄位姿，不叠加电梯目标修正",flush=True)
                    if pick_and_place(robot,grasp6d,pre6d,cfg,place,args.place_duration_s,args.release_wait_s,cancelled):
                        return
                    # Never submit a pre-motion image with a post-motion pose.
                    grasps=[];continue
        display=module._render_display(color,grasps,select_best_grasp(grasps),
            "WAIT FRESH GRASP" if want_grasp else f"G=grasp then auto place/floor{args.floor} | R=refresh Q=cancel")
        cv2.imshow(window,display)
        if worker.pending is None and (want_grasp or frame%every==0):
            worker.submit(current,kind="grasp")
            if want_grasp:selected_id=current.frame_id
        key=cv2.waitKey(1)&255
        if key in (27,32,ord("q"),ord("Q")) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
            raise motion.MotionCancelled("取消夹取阶段")
        if key in (ord("g"),ord("G")) and not want_grasp:want_grasp=True
        elif key in (ord("r"),ord("R")):want_grasp=False;selected_id=None;grasps=[]


def preview(cfg,args):
    """Original G/R/Q perception loop, without main.py's unconditional robot init."""
    camera=None;window="Grasp preview";frozen=False;display=None;results=[];grasps=[];index=0
    try:
        camera=make_camera(cfg);camera.open();camera.warm_up(15)
        K=np.asarray(camera.K,np.float32)
        model,opts=load_yolo(cfg,project_root=ROOT,device_override=args.device)
        every=max(1,int(cfg.get("grasp_pipeline",{}).get("infer_every_live",3)))
        quantile=float(cfg.get("grasp_pipeline",{}).get("grasp",{}).get("depth_quantile",.5))
        cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
        print("[Grasp] 仅预览；G抓帧定位，R继续，Q退出；不初始化机器人",flush=True)
        failures=0
        while True:
            color,depth=camera.get_frame()
            if color is None or depth is None:
                failures+=1
                if failures>=15:raise RuntimeError("连续取帧失败")
                continue
            failures=0;index+=1
            if not frozen:
                if index%every==0 or not results:
                    results=model.predict(color,verbose=False,device=opts.get("device","cpu"),
                        conf=float(opts.get("conf",.25)),iou=float(opts.get("iou",.45)))
                    grasps=estimate_grasps(results,depth,K,depth_quantile=quantile)
                display=color.copy()
                for item in grasps:draw_grasp(display,item)
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&255
            if key in (27,ord("q"),ord("Q")) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key in (ord("r"),ord("R")):frozen=False
            if key in (ord("g"),ord("G")):
                color,depth=camera.get_frame()
                if color is None or depth is None:continue
                results=model.predict(color,verbose=False,device=opts.get("device","cpu"),
                    conf=float(opts.get("conf",.25)),iou=float(opts.get("iou",.45)))
                grasps=estimate_grasps(results,depth,K,depth_quantile=quantile)
                best=select_best_grasp(grasps)
                print("[Grasp] camera_xyz_m="+str(best.position.tolist()) if best is not None else "[Grasp] 无有效目标",flush=True)
                display=color.copy()
                for item in grasps:draw_grasp(display,item)
                frozen=True
        return 0
    finally:
        if camera is not None:camera.close()
        cv2.destroyAllWindows()
