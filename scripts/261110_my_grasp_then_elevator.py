"""按G抓取 → 放置松爪 → 共用panel_261112识别 → 接触--floor指定纸面按键。"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from elevator import app,grasp,geometry


def floor_argument(value):
    try:return geometry.normalize_floor_id(value)
    except ValueError as exc:raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/default.yaml")
    parser.add_argument("--workflow-config",type=Path,default=ROOT/"config/261105_my_elevator.yaml")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--execute",action="store_true",help="真实动作：按一次G后自动执行整轮")
    mode.add_argument("--dry-run",action="store_true",help="只运行相机识别，不初始化机器人（默认）")
    mode.add_argument("--simulate",action="store_true",help="只输出合成几何计划，不打开相机/机器人")
    parser.add_argument("--device")
    parser.add_argument("--ocr-device",choices=("cuda","cpu","auto"))
    parser.add_argument("--verbose",action="store_true",help="输出详细识别、轨迹和反馈日志")
    parser.add_argument("--ocr-preprocessing",choices=("bright_dark","bright"),
                        help="覆盖ocr.preprocessing；bright_dark为亮字+黑字，bright为原亮字流程")
    parser.add_argument("--floor",type=floor_argument,required=True,
                        help="必填：抓取放置后自动选择的整数楼层，如3、-1、12；需稳定识别后执行")
    parser.add_argument("--place-xyz-m",type=float,nargs=3,metavar=("X","Y","Z"),
                        help="临时覆盖workflow-config中的grasp_place.xyz_m；base坐标系end_link XYZ，单位米")
    parser.add_argument("--place-rpy-deg",type=float,nargs=3,metavar=("ROLL","PITCH","YAW"),
                        help="默认保留准备位roll/pitch，yaw朝向放置点")
    parser.add_argument("--place-duration-s",type=float,default=3.)
    parser.add_argument("--release-wait-s",type=float,default=.5)
    parser.add_argument("--auto-floor-timeout-s",type=float,default=60.)
    parser.add_argument("--standoff-m",type=float,help="按钮估计表面前的停靠距离，默认读取 elevator.standoff_m")
    args=parser.parse_args(argv)
    try:
        cfg=app.read_yaml(args.config);options=app.load_workflow(args.workflow_config)
        cfg=app.apply_initial_poses(cfg,options)
        if args.place_xyz_m is None:args.place_xyz_m=list(options["grasp_place"]["xyz_m"])
        args.device=str(cfg.get("yolo",{}).get("device","0")) if args.device is None else args.device
        cfg.setdefault("yolo",{})["device"]=args.device
        if args.ocr_device is not None:options["ocr"]["device"]=args.ocr_device
        if args.ocr_preprocessing is not None:options["ocr"]["preprocessing"]=args.ocr_preprocessing
        for name,lo,hi in (("place_duration_s",.5,30.),("release_wait_s",0.,5.),("auto_floor_timeout_s",1.,600.)):
            value=getattr(args,name)
            if not np.isfinite(value) or not lo<=value<=hi:raise ValueError(f"{name}需在[{lo},{hi}]")
        if args.standoff_m is not None:
            if not .03<=args.standoff_m<=.25:raise ValueError("standoff-m需在0.03～0.25米")
            options["elevator"]["standoff_m"]=args.standoff_m
        place=grasp.placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg)
        # This entry is the printed-panel task, not an unmeasured physical stroke.
        if options["elevator"]["panel_type"]!="printed":raise ValueError("当前串联入口用于打印面板")
        options["gripper"]["initialize"]=True
        args.stage="contact";args.sequence_mode=True
        if args.simulate:
            point=geometry.offset_target_base([.4,-.1,.4],options["elevator"]["target_offset_base_m"])
            poses=app.simulated_button_poses(point,np.eye(4),options['elevator'],options['tool']['tip_offset_in_end_m'])
            settings=options['elevator']
            motion_steps=['前置点','同键三帧后直接接触并撤回'] if settings['refinement']['enabled'] else ['接近/纸面/撤回']
            print(json.dumps(dict(simulation_only=True,floor=args.floor,note="合成几何计划；不代表已抓到物体或实机可达/无碰撞",
                steps=["G抓取成功","返回准备位","搬运到放置点","松爪","夹爪模式切换","七轴抬头",f"稳定识别{args.floor}",*motion_steps,"回零并自动退出"],
                place_pose=place.tolist(),synthetic_button_poses={k:v.tolist() for k,v in poses.items()},
                standoff_m=settings['standoff_m'],relocalize_frames=app.RELOCALIZE_FRAMES if settings['refinement']['enabled'] else 0,
                position_only_ik=settings.get('button_orientation','level')=='free',
                move_duration_s=settings['move_duration_s'],dwell_s=settings['dwell_s'],
                target_offset_base_m=settings["target_offset_base_m"],refinement=settings['refinement']),ensure_ascii=False,indent=2))
            return 0
        def before_head(robot,camera,worker,window):
            grasp.grasp_then_place(robot,camera,worker,window,cfg,args)
        with app.terminal_output(args.verbose):
            return app.run_elevator(cfg,options,args,before_head=before_head)
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        return 2


if __name__=="__main__":
    mp.freeze_support()
    raise SystemExit(main())
