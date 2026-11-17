"""G抓放/N跳过 -> 抬头等待C -> U/D外呼 -> C进轿厢站稳 -> 输入楼层 -> LCD到站回零；N跳过当前电梯阶段。"""
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
from elevator import app, grasp, geometry
from elevator.journey_261113 import Journey, SearchState, load_settings, effective_call_offset


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
    p.add_argument('--press-stage', choices=('approach','contact'), default='contact',
                   help='默认复用既有contact触压；approach仅作分段接近验证')
    p.add_argument('--standoff-m', type=float)
    p.add_argument('--place-xyz-m', type=float, nargs=3,
                   help='临时覆盖workflow-config中的grasp_place.xyz_m；base坐标系end_link XYZ，单位米')
    p.add_argument('--place-rpy-deg', type=float, nargs=3)
    p.add_argument('--place-duration-s', type=float, default=3.)
    p.add_argument('--release-wait-s', type=float, default=.5)
    p.add_argument('--floor-input-delay-s', type=float, default=.4,
                   help='楼层输入停顿后自动提交，默认0.4秒；无需回车，支持负数/两位数')
    p.add_argument('--button-orientation',choices=('free','level'),default='free',
                   help='按键默认仅约束尖端XYZ；level恢复原roll/pitch=0流程，不改变夹取姿态')
    return p.parse_args(argv)


def simulate(cfg, options, settings, args):
    state = SearchState(settings['search_timeout_s'])
    trials = {}
    for phase in ('CALL','FLOOR','DISPLAY'):
        state.enter(phase, 0.)
        deltas = []
        for k in range(3):
            result = state.tick((k+1)*state.timeout_s, False)
            deltas.append(result)
            if result != 'HOME': state.retry_complete((k+1)*state.timeout_s)
        trials[phase] = deltas
    poses = geometry.button_poses([.4,-.1,.4], np.eye(4), options['tool']['tip_offset_in_end_m'],
                                  options['elevator']['standoff_m'], options['elevator']['press_travel_m'])
    print(json.dumps(dict(simulation_only=True, devices_created=False,
        note='合成流程/几何计划，非实景识别、IK可达性、碰撞或按钮触发验证。',
        stages=['G原抓放或N跳过','七轴抬头闭合','等待C开始外呼识别','稳定两键U/D','接近/可选接触/撤回','恢复抬头',
                'C人工进轿厢站稳','稳定数字面板并输入楼层（无需回车）','接近/可选接触/撤回',
                '恢复'+json.dumps(settings['display_view_deg'],separators=(',',':')),'稳定当前LCD等于所选楼层','SDK回零/失能/断开'],
        search_timeout_s=state.timeout_s, search_failure_traces=trials,
        grasp_ready_pose=grasp.ready_settings(cfg),
        view_joints_deg={'CALL':options.get('initial_poses',{}).get('call_view_deg',options['startup_view']['joints_deg'][:6]),
                         'FLOOR':options['startup_view']['joints_deg'][:6],
                         'DISPLAY':settings['display_view_deg']},
        place_pose=grasp.placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg).tolist(),
        synthetic_press_poses={k:v.tolist() for k,v in poses.items()},
        call_target_offset_base_m=effective_call_offset(settings,options).tolist(),
        floor_target_offset_base_m=options['elevator']['target_offset_base_m'],
        floor_input_delay_s=args.floor_input_delay_s,
        floor_cache_age_s=(None if options['elevator'].get('floor_cache_until_update',False)
                           else options['elevator']['stable_snapshot_age_s']),
        floor_cache_until_update=options['elevator'].get('floor_cache_until_update',False),
        button_orientation=args.button_orientation,position_only_ik=args.button_orientation=='free',
        skip_key='N', skip_transitions={'WAIT_CALL':'BOARD (C confirms boarding)',
            'CALL':'BOARD (C confirms boarding)','FLOOR':'DISPLAY (retain/input arrival target)',
            'DISPLAY':'DONE (SDK home and disconnect)'},
        shared_recognizer=True, press_stage=args.press_stage),ensure_ascii=False,indent=2))
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = copy.deepcopy(app.read_yaml(args.config))
        options = app.load_workflow(args.workflow_config)
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
            if not .03 <= args.standoff_m <= .1: raise ValueError('standoff-m需0.03..0.10米')
            options['elevator']['standoff_m'] = args.standoff_m
        for name, lo, hi in (('place_duration_s',.5,30.),('release_wait_s',0.,5.),('floor_input_delay_s',0.,2.)):
            value = getattr(args,name)
            if not np.isfinite(value) or not lo <= value <= hi: raise ValueError(f'invalid {name}')
        grasp.placement_pose(cfg,args.place_xyz_m,args.place_rpy_deg)
        if args.simulate: return simulate(cfg,options,settings,args)
        return Journey(cfg,options,settings,args).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
