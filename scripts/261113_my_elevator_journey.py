"""G抓放/N跳过 → C开始外呼识别 → U/D外呼 → C确认进轿厢 → 输入楼层 → C开始LCD识别 → 到站回零。"""
from pathlib import Path
import argparse
import copy
import json
import multiprocessing as mp
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from elevator import app, grasp, motion
from elevator.journey_261113 import Journey, SearchState, load_settings, effective_call_offset, relative_head_target


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/default.yaml')
    p.add_argument('--workflow-config', type=Path, default=ROOT/'config/261105_my_elevator.yaml')
    p.add_argument('--journey-config', type=Path, default=ROOT/'config/261113_my_journey.yaml')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='默认：相机识别，不初始化机器人')
    mode.add_argument('--simulate', action='store_true', help='无设备的合成状态/几何计划')
    p.add_argument('--device')
    p.add_argument('--ocr-device', choices=('cuda','cpu','auto'))
    p.add_argument('--verbose',action='store_true',help='输出详细识别、轨迹和反馈日志')
    p.add_argument('--press-stage', choices=('approach','contact'), default='contact',
                   help='contact 执行触压，approach 到前置点后结束按键动作')
    p.add_argument('--standoff-m', type=float, help='按钮估计表面前的停靠距离，默认读取 elevator.standoff_m')
    p.add_argument('--place-xyz-m', type=float, nargs=3,
                   help='临时覆盖workflow-config中的grasp_place.xyz_m；base坐标系end_link XYZ，单位米')
    p.add_argument('--place-rpy-deg', type=float, nargs=3)
    p.add_argument('--place-duration-s', type=float, default=3.)
    p.add_argument('--release-wait-s', type=float, default=.5)
    p.add_argument('--floor-input-delay-s', type=float, default=.4,
                   help='楼层输入停顿后自动提交，默认0.4秒；无需回车，支持负数/两位数')
    p.add_argument('--button-orientation',choices=('free','level'),
                   help='覆盖 elevator.button_orientation。free 只约束尖端 XYZ，level 使用完整位姿')
    return p.parse_args(argv)


def simulate(cfg, options, settings, args):
    state = SearchState(settings['search_timeout_s'], retry_plans=settings['search_retry_plans'])
    view_joints = {'CALL':options.get('initial_poses',{}).get('call_view_deg',options['startup_view']['joints_deg'][:6]),
                   'FLOOR':options['startup_view']['joints_deg'][:6],
                   'DISPLAY':settings['display_view_deg']}
    limits = motion.configured_arm_position_limits(cfg)
    trials = {}
    for phase,initial in view_joints.items():
        joints=np.degrees(motion.checked_head_goal(initial,limits))
        state.enter(phase, 0.)
        deltas = []
        for _ in range(state.RETRY_COUNT + 1):
            now=state.deadline
            result = state.tick(now, False)
            goal = None if result == 'HOME' else relative_head_target(joints, result, limits, last_goal_deg=joints)
            if goal is None:
                deltas.append('HOME')
                break
            j2, j3 = (goal[1:3] - joints[1:3]).round(6).tolist()
            deltas.append(dict(J2=j2, J3=j3))
            joints=goal
            state.retry_complete(now)
        trials[phase] = deltas
    elevator = options['elevator']
    refinement = elevator['refinement']
    press_sequence = ('前置点/同键三帧/可选接触及撤回' if refinement['enabled']
                      else '一次定位后接近/可选接触及撤回')
    poses = app.simulated_button_poses(np.array([.4,-.1,.4]),np.eye(4),elevator,options['tool']['tip_offset_in_end_m'])
    print(json.dumps(dict(simulation_only=True, devices_created=False,
        note='合成流程/几何计划，非实景识别、IK可达性、碰撞或按钮触发验证。',
        stages=['G原抓放或N跳过','七轴抬头闭合','等待C开始外呼识别','稳定两键U/D',press_sequence,'恢复抬头',
                'C人工进轿厢站稳','稳定数字面板并输入楼层（无需回车）',press_sequence,
                '恢复'+json.dumps(settings['display_view_deg'],separators=(',',':')),'等待C开始LCD识别',
                '稳定当前LCD等于所选楼层','SDK回零并自动退出'],
        search_timeout_s={phase.lower():seconds for phase,seconds in state.timeouts.items()}, search_failure_traces=trials,
        robot_control=options['robot_control'],
        grasp_ready_pose=grasp.ready_settings(cfg),
        view_joints_deg=view_joints, arm_joint_limits_deg=np.degrees(limits).tolist(),
        place_pose=grasp.placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg).tolist(),
        synthetic_press_poses={k:v.tolist() for k,v in poses.items()},
        call_target_offset_base_m=effective_call_offset(settings,options).tolist(),
        floor_target_offset_base_m=options['elevator']['target_offset_base_m'],
        floor_input_delay_s=args.floor_input_delay_s,
        floor_cache_age_s=(None if options['elevator'].get('floor_cache_until_update',False)
                           else options['elevator']['stable_snapshot_age_s']),
        floor_cache_until_update=options['elevator'].get('floor_cache_until_update',False),
        button_orientation=args.button_orientation,position_only_ik=args.button_orientation=='free',
        standoff_face_panel=refinement['enabled'] and refinement['standoff_face_panel'],
        skip_key='N', skip_transitions={'WAIT_CALL':'BOARD (C confirms boarding)',
            'CALL':'BOARD (C confirms boarding)','FLOOR':'WAIT_DISPLAY (C starts LCD)',
            'WAIT_DISPLAY':'DONE (SDK home and exit)',
            'DISPLAY':'DONE (SDK home and exit)'},
        standoff_m=elevator['standoff_m'],relocalize_frames=app.RELOCALIZE_FRAMES if refinement['enabled'] else 0,
        move_duration_s=elevator['move_duration_s'],dwell_s=elevator['dwell_s'],
        shared_recognizer=True,press_stage=args.press_stage,refinement=refinement),ensure_ascii=False,indent=2))
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = copy.deepcopy(app.read_yaml(args.config))
        options = app.load_workflow(args.workflow_config)
        if args.button_orientation is None:
            args.button_orientation = options['elevator'].get('button_orientation','free')
        cfg = app.apply_initial_poses(cfg,options)
        if args.place_xyz_m is None:
            args.place_xyz_m = list(options['grasp_place']['xyz_m'])
        # User confirmed physical buttons and reuse of the proven contact
        # routine. Preserve the existing stroke; never invent extra travel.
        options['elevator']['panel_type'] = 'physical'
        settings = load_settings(args.journey_config,options)
        effective_call_offset(settings, options)
        args.device = str(cfg.get('yolo',{}).get('device','0')) if args.device is None else args.device
        cfg.setdefault('yolo',{})['device'] = args.device
        if args.ocr_device is not None: options['ocr']['device'] = args.ocr_device
        if args.standoff_m is not None:
            if not .03 <= args.standoff_m <= .25: raise ValueError('standoff-m需0.03..0.25米')
            options['elevator']['standoff_m'] = args.standoff_m
        for name, lo, hi in (('place_duration_s',.5,30.),('release_wait_s',0.,5.),('floor_input_delay_s',0.,2.)):
            value = getattr(args,name)
            if not np.isfinite(value) or not lo <= value <= hi: raise ValueError(f'invalid {name}')
        grasp.placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg)
        if args.simulate: return simulate(cfg,options,settings,args)
        with app.terminal_output(args.verbose):
            return Journey(cfg,options,settings,args).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
