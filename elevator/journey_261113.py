"""One camera/robot/model session for grasp -> call -> floor -> arrival.

Existing algorithms and SDK actions are imported, not replaced. No device is
created on import. Walking/boarding is an operator handoff, not navigation.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import select
import sys
import termios
import time
import tty
from types import SimpleNamespace

import cv2
import numpy as np

from drivers.camera.base import CameraFrameError
from . import app, detail, geometry, grasp, motion, terminal_output, terminal_verbose

ROOT = Path(__file__).resolve().parents[1]


class SearchState:
    PHASES = ('CALL', 'FLOOR', 'DISPLAY')
    RETRY_COUNT = 2
    DEFAULT_RETRY_DELTA_DEG = (-15., -25.)

    def __init__(self, timeout_s=10., *, retry_plans=None):
        self.retry_plans = (dict.fromkeys(self.PHASES, (self.DEFAULT_RETRY_DELTA_DEG,) * self.RETRY_COUNT)
                            if retry_plans is None else retry_plans)
        phase_keys={phase.lower() for phase in self.PHASES}
        values=timeout_s if isinstance(timeout_s,dict) else dict.fromkeys(phase_keys,timeout_s)
        if set(values)!=phase_keys:
            raise ValueError('search_timeout_s 需要 call、floor、display 三个值')
        self.timeouts={}
        for phase in self.PHASES:
            raw=values[phase.lower()]
            message=f'search_timeout_s.{phase.lower()} 需要 1～60 秒'
            try:
                seconds=float(raw)
            except (TypeError,ValueError) as exc:
                raise ValueError(message) from exc
            if isinstance(raw,bool) or not 1.<=seconds<=60.:
                raise ValueError(message)
            self.timeouts[phase]=seconds
        self.phase, self.retry_index, self.deadline = 'GRASP', 0, None
        self.started_at = None
        self.waiting_retry = False

    @property
    def timeout_s(self):
        return self.timeouts.get(self.phase,self.timeouts['CALL'])

    def enter(self, phase, now):
        if phase not in (*self.PHASES, 'GRASP', 'WAIT_CALL', 'BOARD', 'WAIT_DISPLAY', 'DONE', 'TIMEOUT'):
            raise ValueError('unknown phase')
        self.phase, self.retry_index = phase, 0
        self.waiting_retry = False
        self.started_at = now if phase in self.PHASES else None
        self.deadline = now+self.timeout_s if phase in self.PHASES else None

    def retry_complete(self, now):
        self.waiting_retry = False
        self.started_at = now
        self.deadline = now+self.timeout_s

    def tick(self, now, ready):
        if self.phase not in self.PHASES or self.waiting_retry:
            return None
        if ready:
            self.deadline = None
            return None
        if self.deadline is None:
            self.deadline = now+self.timeout_s
        if now < self.deadline:
            return None
        if self.retry_index == self.RETRY_COUNT:
            self.phase, self.deadline = 'TIMEOUT', None
            return 'HOME'
        delta = self.retry_plans[self.phase][self.retry_index]
        self.retry_index += 1
        self.deadline, self.waiting_retry = None, True
        return delta


def relative_head_target(joints_deg, delta_deg, limits, *, last_goal_deg=None):
    """Apply a J2/J3 step within URDF limits, holding the other axes at feedback."""
    q = np.asarray(joints_deg, float)
    delta = np.asarray(delta_deg, float)
    if q.shape != (6,) or delta.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(delta).all():
        raise ValueError('抬头重试需要六轴有效反馈和 J2/J3 两个有限角度增量')

    lower, upper = np.degrees(limits[1:3]).T
    current = q[1:3]
    proposed = np.clip(current + delta, lower, upper)
    advances = np.sign(delta) * (proposed - current) > 1e-9
    new_view = advances.copy()
    if last_goal_deg is not None:
        boundary = np.where(delta < 0, lower, upper)
        at_last_boundary = (proposed == boundary) & np.isclose(last_goal_deg[1:3], boundary, rtol=0., atol=1e-9)
        new_view &= ~at_last_boundary  # Lagging feedback does not create another search at the same limit.
    if not new_view.any():
        return None

    goal = q.copy()
    goal[1:3] = np.where(advances, proposed, current)
    return np.degrees(motion.checked_head_goal(goal, limits))


def shared_display_ocr(detector):
    from .floor_display_261113 import DisplayOCR
    base = detector.ocr.retry_backend
    if base is None:
        raise RuntimeError('数字OCR没有可复用的识别引擎')
    reader = DisplayOCR.__new__(DisplayOCR)
    reader.engine, reader.input_type = base.engine, base.input_type
    reader.info, reader.calls = dict(base.backend_info), 0
    return reader


def _worker(connection, requests, cfg, settings, journey):
    try:
        began=time.perf_counter()
        from .panel_261112 import FastPanelPerception
        from .call_panel_261112 import CallPanelDetector
        from .floor_display_261113 import FloorDisplayReader
        panel = FastPanelPerception(cfg, argparse.Namespace(**settings))
        call = CallPanelDetector(required=3)
        display = FloorDisplayReader(shared_display_ocr(panel), min_confidence=journey['display_min_confidence'])
        initialized=time.perf_counter()
        panel.warmup(cfg['camera'].get('color_height', 720), cfg['camera'].get('color_width', 1280))
        with terminal_output(settings.get('verbose',True)):
            app.report_vision_ready(panel.ocr.backend_info,began,initialized)
        connection.send(('ready',))
        previous = None
        while True:
            request = requests.get()
            if request is None:
                break
            kind, frame_id, at, color, depth, K, generation, focus_px = request
            if previous != (kind, generation):
                panel.request_refresh(); call.reset(); display.reset()
                previous = (kind, generation)
            if kind == 'grasp':
                from utils.ordinary_grasp import estimate_grasps
                opts = panel.yolo_options
                results = panel.model.predict(color, verbose=False, device=opts.get('device', 'cpu'),
                                              conf=opts.get('conf', .25), iou=opts.get('iou', .45))
                result = estimate_grasps(results, depth, K,
                    depth_quantile=float(cfg.get('grasp_pipeline', {}).get('grasp', {}).get('depth_quantile', .75)))
            elif kind == 'panel':
                result = panel.process(color, captured_at=at, focus_px=focus_px)
            elif kind == 'call':
                result = call.process(color, captured_at=at, generation=generation)
            elif kind == 'display':
                result = display.process(color, captured_at=at, generation=generation)
            else:
                raise ValueError('unknown journey inference task')
            connection.send(('result', frame_id, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(('error', f'{type(exc).__name__}: {exc}'))
        except OSError:
            pass
    finally:
        connection.close()


class JourneyWorker(app.PerceptionWorker):
    KINDS=('grasp','panel','call','display')

    def __init__(self, cfg, settings, journey):
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.requests = ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_worker, args=(child, self.requests, cfg, settings, journey), daemon=True)
        self.process.start(); child.close()
        self.pending, self.submitted_at, self.timeout = None, 0., 20.


def retry_angle_pair(raw, key):
    message = f'{key} 需要两个有限角度，单位度'
    try:
        angles = np.asarray(raw, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if angles.shape != (2,) or not np.isfinite(angles).all() or any(isinstance(angle, bool) for angle in raw):
        raise ValueError(message)
    return tuple(angles.tolist())


def load_settings(path, options=None):
    value = app.read_yaml(path)
    workflow = options or {}
    value['search_timeout_s']=workflow.get('search_timeout_s',value.get('search_timeout_s',10.))
    retry_key = 'search_retry_delta_deg'
    legacy_keys = {phase: f'{phase.lower()}_j3_retry_deg' for phase in SearchState.PHASES}
    if retry_key not in workflow and any(key in value for key in legacy_keys.values()):
        # Older workflow/journey backup pairs retain their J3-only schedules.
        value['search_retry_plans'] = {
            phase: tuple((0., step) for step in retry_angle_pair(value.get(key), key))
            for phase, key in legacy_keys.items()
        }
    else:
        step = retry_angle_pair(workflow.get(retry_key, SearchState.DEFAULT_RETRY_DELTA_DEG), retry_key)
        value['search_retry_plans'] = dict.fromkeys(SearchState.PHASES, (step,) * SearchState.RETRY_COUNT)
    SearchState(value['search_timeout_s'], retry_plans=value['search_retry_plans'])
    for name, lo, hi in (('settle_s', .1, 2.), ('max_settle_s', 1., 15.),
                         ('arrival_timeout_s', 10., 3600.), ('max_observation_age_s', .1, 2.),
                         ('display_min_confidence', .5, 1.)):
        number = float(value[name])
        if not math.isfinite(number) or not lo <= number <= hi:
            raise ValueError(f'invalid {name}')
        value[name] = number
    # Current workflows have one pose source. Older custom journey files can
    # still supply their display pose when no shared initial_poses is present.
    initial=workflow.get('initial_poses',{})
    head = np.asarray(initial.get('display_view_deg',value.get('display_view_deg')), float)
    if head.shape != (6,) or not np.isfinite(head).all():
        raise ValueError('invalid display pose; configure initial_poses.display_view_deg in workflow-config')
    value['display_view_deg']=head.tolist()
    if value.get('call_target_offset_base_m') is not None:
        geometry.offset_target_base(None, value['call_target_offset_base_m'])
    return value


def effective_call_offset(settings, options):
    value = settings.get('call_target_offset_base_m')
    if value is None:
        value = options['elevator']['target_offset_base_m']
    geometry.offset_target_base(None, value)
    return np.asarray(value, float).copy()


def display_floor_id(value):
    if value is None:
        return None
    text = str(value)
    if text.startswith('B') and text[1:].isdigit():
        text = '-'+text[1:]
    try:
        return geometry.normalize_floor_id(text)
    except ValueError:
        return None


class Journey:
    """Blocking robot actions, responsive cancellation; single inference task."""
    KINDS = {'GRASP': 'grasp', 'CALL': 'call', 'FLOOR': 'panel', 'DISPLAY': 'display'}
    SKIP_PHASES = ('WAIT_CALL', 'CALL', 'FLOOR', 'WAIT_DISPLAY', 'DISPLAY')

    def __init__(self, cfg, options, settings, args):
        self.cfg, self.options, self.settings, self.args = app.apply_initial_poses(cfg,options), options, settings, args
        initial=options.get('initial_poses',{})
        self.views={'CALL':list(initial.get('call_view_deg',options['startup_view']['joints_deg'][:6])),
                    'FLOOR':list(initial.get('floor_view_deg',options['startup_view']['joints_deg'][:6])),
                    'DISPLAY':list(initial.get('display_view_deg',settings['display_view_deg']))}
        self.state = SearchState(settings['search_timeout_s'], retry_plans=settings.get('search_retry_plans'))
        self.robot = self.camera = self.worker = None
        self.window = 'Elevator journey | G/N -> C -> U/D -> C -> floor/N -> C -> LCD'
        self.display = self.last = self.last_snapshot = None
        self.call_cache = None
        self.floor_cache = {}
        self.grasp_view = None
        self.rendered_result = self.rendered_panel = None
        self.generation = self.frame_id = self.failures = 0
        self.recognition_logged_generation = None
        self.tracker = app._tracker(options)
        self.request = None; self.typed = ''; self.floor = None
        self.floor_input_due = None
        self.floor_input_delay_s = float(getattr(args, 'floor_input_delay_s', .4))
        self.terminal_state = None
        self.want_grasp = False; self.grasp_after = None
        self.home = False; self.arrival_deadline = None
        self.quit_home_requested = False
        self.skip_requested = False
        self.skip_in_progress = False
        self.skip_retreat = None
        self.skip_start_pose = None
        self.skip_position_only = False
        self.button_orientation = getattr(args,'button_orientation','level')
        self.preview_q = np.array(self.views['CALL'], float)
        self.preview_limits = None

    def log(self, message, *, verbose_only=False):
        printer=detail if verbose_only else print
        printer(f'[Journey:{self.state.phase}] {message}', flush=True)

    def cancelled(self):
        if self.display is not None:
            cv2.imshow(self.window, self.display)
        key = cv2.waitKey(1) & 255
        try:
            visible = cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if not visible or key in (27, 32):
            self.quit_home_requested = False
            self.skip_requested = False
            return True
        if key in (ord('q'), ord('Q')):
            self.quit_home_requested = True
            self.skip_requested = False
            return True
        value = self.read_terminal()
        if value is not None and value.lower() in ('q', 'stop'):
            self.quit_home_requested = value.lower() == 'q'
            self.skip_requested = False
            return True
        if ((key in (ord('n'), ord('N')) or value is not None and value.lower() == 'n')
                and self.state.phase in self.SKIP_PHASES and not self.skip_in_progress):
            self.skip_requested = True
            self.quit_home_requested = False
            return True
        return False

    def stage_action(self, operation, *args):
        """Finish SDK cancellation before carrying out an explicitly requested skip."""
        try:
            return operation(*args)
        except motion.MotionCancelled:
            if not self.skip_requested:raise
            self.skip_stage()

    def enter_board(self):
        self.head(self.views['FLOOR'])
        self.restore_terminal()
        self.arrival_deadline = None
        self.state.enter('BOARD', time.perf_counter())
        text='到轿厢面板前站稳后按 C。'
        if self.args.press_stage=='approach':text='人工呼叫电梯。'+text
        self.log(text)

    def enter_display(self, floor):
        self.floor = floor
        self.arrival_deadline = None
        self.state.enter('WAIT_DISPLAY', time.perf_counter())
        self.head(self.views['DISPLAY'])
        self.enable_floor_terminal()
        self.log('按 C 开始识别当前楼层，N 跳过观察并回零退出。')

    def begin_display(self):
        self.invalidate(); self.settle()
        self.state.enter('DISPLAY', time.perf_counter())
        if self.floor is None:
            self.log('开始LCD楼层识别；请输入到站目标楼层（仅用于显示比较），无需回车；N回零退出。')
        else:
            self.restore_terminal()
            self.arrival_deadline = time.perf_counter()+self.settings['arrival_timeout_s']
            self.log(f'等待独立稳定的当前楼层={self.floor}；N跳过识别并回零退出。')

    def skip_stage(self):
        phase = self.state.phase
        if phase not in self.SKIP_PHASES or self.skip_in_progress:return False
        target = self.floor
        if phase == 'FLOOR':
            raw = self.typed or (self.request['value'] if self.request is not None else target)
            try:target = geometry.normalize_floor_id(raw) if raw is not None else None
            except ValueError:target = None
        self.skip_requested = False
        self.skip_in_progress = True
        try:
            # A failed stop propagates as a control error, before any new goal
            # or home request. SDK failures never count as successful skips.
            if self.robot is not None:self.robot.stop()
            retreat, self.skip_retreat = self.skip_retreat, None
            start_pose, self.skip_start_pose = self.skip_start_pose, None
            position_only, self.skip_position_only = self.skip_position_only, False
            if self.robot is not None and retreat is not None and start_pose is not None:
                actual = self.robot.pose()
                angle = np.arccos(np.clip((np.trace(start_pose[:3,:3].T@actual[:3,:3])-1)/2,-1,1))
                if np.linalg.norm(actual[:3,3]-start_pose[:3,3]) <= .002 and angle <= np.radians(.5):
                    retreat = None  # N before the first SDK move must not move toward the button.
            self.invalidate()
            self.arrival_deadline = None
            if self.robot is not None and retreat is not None:
                self.log('N已停止按钮动作，先回撤到该动作原定前置点，再切换阶段。')
                if position_only:
                    offset=np.asarray(self.options['tool']['tip_offset_in_end_m'])
                    self.robot.move_position(retreat[:3,3]+retreat[:3,:3]@offset,
                        self.options['elevator']['retreat_duration_s'],self.cancelled,tip_offset=offset)
                else:self.robot.move(retreat, self.options['elevator']['retreat_duration_s'], self.cancelled)
            if phase in ('WAIT_CALL', 'CALL'):
                self.log('N跳过外呼，恢复抬头后进入人工进轿厢交接。')
                self.enter_board()
            elif phase == 'FLOOR':
                self.log('N跳过楼层按键，转到 LCD 观察姿态。')
                self.enter_display(target)
            else:
                self.restore_terminal()
                self.home = True
                self.state.enter('DONE', time.perf_counter())
                self.log('N跳过最后的LCD识别阶段，自动回零退出。' if self.robot is not None else 'N跳过最后的LCD识别阶段，退出预览。')
        finally:
            self.skip_in_progress = False
        return True

    def enable_floor_terminal(self):
        """Read floor digits and the LCD handoff key without Enter."""
        if self.terminal_state is not None or not sys.stdin.isatty():
            return
        try:
            fd = sys.stdin.fileno()
            settings = termios.tcgetattr(fd)
            tty.setcbreak(fd, termios.TCSANOW)
        except (AttributeError, OSError, ValueError, TypeError, termios.error):
            return  # OpenCV window keys still work with redirected stdin.
        self.terminal_state = fd, settings

    def restore_terminal(self):
        if self.terminal_state is not None:
            fd, settings = self.terminal_state
            termios.tcsetattr(fd, termios.TCSANOW, settings)
            self.terminal_state = None

    def read_terminal(self):
        if self.terminal_state is not None:
            fd = self.terminal_state[0]
            if not select.select([fd], [], [], 0)[0]:return None
            value = os.read(fd, 1).decode('ascii', errors='ignore')
            return 'stop' if value in (' ', '\x1b', 's', 'S') else value
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
        return None

    def invalidate(self):
        self.generation += 1
        self.last = self.last_snapshot = self.request = None
        self.call_cache = None
        self.floor_cache.clear()
        self.typed = ''; self.floor_input_due = None
        self.rendered_result = self.rendered_panel = None
        self.tracker.reset()
        if self.camera is not None:
            self.camera.warm_up(3)

    def snapshot(self, *, with_arm_feedback=False):
        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            self.failures += 1
            if self.failures >= 15:
                raise CameraFrameError('连续15次RGB-D获取失败')
            return None
        self.failures = 0; self.frame_id += 1
        feedback = None
        if self.robot is None:
            pose = np.eye(4)
        elif with_arm_feedback:
            pose,feedback = self.robot.pose_with_feedback()
        else:
            pose = self.robot.pose()
        return geometry.Snapshot(self.frame_id, time.perf_counter(), color.copy(), depth.copy(),
                                 self.K, self.D, pose, self.generation,arm_feedback=feedback)

    def settle(self):
        began = time.perf_counter()
        limit = self.settings['max_settle_s']
        until = began+limit
        since = anchor = None
        peak_m = peak_deg = 0.
        frames = 0
        while time.perf_counter() < until:
            snap = self.snapshot()
            if snap is not None:
                if snap.captured_at>=until:break
                frames += 1
                self.display = snap.color.copy()
                if anchor is None:
                    anchor, since = snap.T_base_end, snap.captured_at
                delta,angle = geometry.pose_difference(anchor,snap.T_base_end)
                peak_m,peak_deg=max(peak_m,delta),max(peak_deg,np.degrees(angle))
                if delta > .002 or angle > np.radians(.5):
                    anchor, since = snap.T_base_end, snap.captured_at
                if snap.captured_at-since >= self.settings['settle_s']:
                    self.log(f'[Settle] wait_s={snap.captured_at-began:.2f} '
                             f'peak_window_mm={peak_m*1000:.1f} peak_window_deg={peak_deg:.2f}',verbose_only=True)
                    return
            else:since=anchor=None
            if self.cancelled():
                raise motion.MotionCancelled('用户停止')
        reason = '六轴反馈未停稳' if frames else '无可用RGB-D'
        raise TimeoutError(f'{reason}（{limit:g}s）；反馈窗口最大位姿变化 '
                           f'{peak_m*1000:.1f} mm / {peak_deg:.2f}°；未开始{self.state.timeout_s:g}秒识别窗口')

    def arm_view_limits(self):
        if self.robot is not None:
            return self.robot.arm_position_limits()
        if self.preview_limits is None:
            self.preview_limits = motion.configured_arm_position_limits(self.cfg)
        return self.preview_limits

    def head(self, joints):
        joints = np.asarray(joints[:6],float)
        if self.robot is not None:
            values = joints.tolist()+[self.options['gripper']['closed_angle_deg']]
            self.robot.head_up(values, self.options['startup_view']['duration_s'], self.cancelled,
                               closed_tolerance=self.options['gripper']['closed_tolerance_rad'],
                               wait_for_arrival=self.options['startup_view']['wait_for_arrival'])
        else:
            motion.checked_head_goal(joints,self.arm_view_limits())
        self.preview_q = joints.copy()
        self.invalidate(); self.settle()

    def begin_call(self):
        if self.robot is not None:
            self.robot.switch_gripper_to_posvel(self.cancelled)
        self.head(self.views['CALL'])
        # No inference kind or search deadline until the operator confirms.
        self.state.enter('WAIT_CALL', time.perf_counter())
        self.log('按 C 搜寻外呼面板，Q 回零退出。')

    def fresh(self, now):
        return (self.last is not None and self.last_snapshot is not None
                and self.last_snapshot.generation == self.generation
                and 0 <= now-self.last_snapshot.captured_at <= self.settings['max_observation_age_s'])

    def ready(self, now):
        if self.state.phase == 'CALL':
            return (self.fresh(now) and self.last.state == 'STABLE' and len(self.last.pair) == 2
                    or self.call_memory(now) is not None)
        if self.state.phase == 'FLOOR':
            return (self.fresh(now) and self.tracker.panel.stable and self.last.panel_bbox is not None and not self.last.errors
                    or bool(self.floor_memory(now)))
        if not self.fresh(now):
            return False
        if self.state.phase == 'DISPLAY':
            # A confirmed non-target floor is valid: wait, do not move J3.
            return self.last.state == 'STABLE' and self.last.current_floor is not None
        return False

    def call_memory(self, now):
        """A bounded complete stable RGB-D/pose bundle; not a timeless target."""
        if self.call_cache is None:
            return None
        snapshot, result = self.call_cache
        age_limit = self.options['elevator'].get('stable_snapshot_age_s', 2.)
        if (age_limit <= 0 or snapshot.generation != self.generation
                or not 0 <= now-snapshot.captured_at <= age_limit):
            self.call_cache = None
            return None
        return snapshot, result

    def resolve_call(self, current):
        """Reuse stable UP/DOWN without another detector pass after the key."""
        latest = self.fresh(current.captured_at) and self.last.state == 'STABLE' and len(self.last.pair) == 2
        after_key = latest and self.last_snapshot.captured_at > self.request['at']
        if after_key:
            snapshot, result = self.last_snapshot, self.last
            age_limit = self.settings['max_observation_age_s']
            source = 'NEW_RESULT'
        else:
            saved = self.call_memory(current.captured_at)
            if saved is None:
                raise ValueError('等待稳定外呼观测；近期缓存不存在或已过期')
            snapshot, result = saved
            age_limit = self.options['elevator'].get('stable_snapshot_age_s', 2.)
            source = 'STABLE_CALL_CACHE'
        if self.fresh(current.captured_at):
            if self.last.state in ('EXTRA_BUTTONS', 'AMBIGUOUS', 'CLUTTERED', 'INVALID_FRAME', 'INVALID_TIMESTAMP'):
                raise ValueError('当前外呼面板有歧义或无效，不能复用缓存')
            for old, new in zip(result.pair, self.last.pair):
                if (math.hypot(old.x-new.x, old.y-new.y) > max(3., .15*old.radius)
                        or not .9 <= new.radius/old.radius <= 1.1):
                    raise ValueError('当前外呼位置已改变，等待重新稳定')
        # Check BOTH keys, not just the selected one. These cheap local checks
        # reject moved/covered panels while tolerating a detector-only dropout.
        for ring in result.pair:
            geometry.check_cached_observation(snapshot, current, SimpleNamespace(ring=ring), age_limit)
        index = 0 if self.request['value'] == 'U' else 1
        return snapshot, SimpleNamespace(ring=result.pair[index]), source

    def floor_cache_age_limit(self):
        if self.options['elevator'].get('floor_cache_until_update',False):return None
        return self.options['elevator'].get('stable_snapshot_age_s',2.)

    def floor_memory(self, now):
        """One latest confirmed RGB-D bundle per floor in this motion epoch."""
        age_limit = self.floor_cache_age_limit()
        self.floor_cache = {label:saved for label,saved in self.floor_cache.items()
            if saved[0].generation == self.generation and math.isfinite(now-saved[0].captured_at)
            and 0 <= now-saved[0].captured_at
            and (age_limit is None or age_limit > 0 and now-saved[0].captured_at <= age_limit)}
        return self.floor_cache

    def remember_floors(self, snapshot, result, now):
        self.floor_memory(now)
        if result.errors or result.panel_bbox is None or not self.tracker.panel.stable:return
        labels = {b.text for b in result.buttons if b.text is not None}
        associations = tuple(self.tracker.remembered_associations(result, now=now))
        labels.update(a.label for a in associations)
        # Validate label-memory associations once per frame, then reuse them
        # while selecting individual labels from the same result.
        evidence_tracker = SimpleNamespace(panel=self.tracker.panel,
            remembered_associations=lambda result,now=None: associations)
        if geometry.position_floor_indices(result, '1'):labels.add('1')
        updates = {}
        for label in labels:
            try:
                label = geometry.normalize_floor_id(label)
                button = geometry.select_floor(result, evidence_tracker, label, self.options['ocr']['min_confidence'], now)
                self.check_floor_conflicts(label,button,now)
            except ValueError:continue
            updates[label] = snapshot, result, button
        if self.floor_cache_age_limit() is None:
            # A later UNIQUE confirmed reading at the same physical rim
            # replaces the old identity. Apply a whole frame atomically so a
            # two-label swap cannot accidentally delete the other new record.
            for label, saved in tuple(self.floor_cache.items()):
                old = saved[2].ring
                for new_label, (_, _, button) in updates.items():
                    ring=button.ring
                    same=(math.hypot(ring.x-old.x,ring.y-old.y)<max(6.,.5*old.radius)
                          and .75<=ring.radius/old.radius<=1.33)
                    if label!=new_label and same:
                        del self.floor_cache[label]
                        break
        self.floor_cache.update(updates)

    def check_floor_conflicts(self, label, button, now):
        if not self.fresh(now):return
        if self.last.errors:raise ValueError('当前面板识别异常，等待恢复后再复用缓存')
        metadata = getattr(self.last, 'tracking', {})
        inferred = geometry.position_floor_indices(self.last, '1')
        matches = set()
        for index, other in enumerate(self.last.buttons):
            readings = list(getattr(other, 'variants', []))
            if other.text is not None and other.confidence is not None:
                readings.append((other.text, other.confidence))
            proof = metadata.get('region_evidence', {}).get(str(index), {})
            cached = metadata.get('display_label_cache', {}).get(str(index), {})
            if (self.floor_cache_age_limit() is None and other.status!='STABLE' and index not in inferred):
                # Flickering OCR candidates must not veto an already confirmed
                # unchanged button. A new stable reading can replace it above.
                readings=[]
            if proof.get('source') == 'PIXEL_MATCHED_CACHE' and proof.get('pixel_match') and cached:
                readings.append((cached['text'], cached['confidence']))
            same = math.hypot(other.ring.x-button.ring.x, other.ring.y-button.ring.y) < max(6., .5*button.ring.radius)
            labels = {'1'} if index in inferred else set()
            for text, score in readings:
                if not math.isfinite(score) or score < self.options['ocr']['min_confidence']:continue
                try:current_label = geometry.normalize_floor_id(text)
                except ValueError:
                    if same and len(str(text).strip()) >= 3 and str(text).strip().isalpha():
                        raise ValueError('当前圆环出现功能文字，不能使用旧楼层缓存')
                    continue
                labels.add(current_label)
            for current_label in labels:
                if current_label == label:matches.add(index)
                if (same and current_label != label) or (not same and current_label == label):
                    raise ValueError('当前数字与缓存位置冲突，等待重新稳定')
        if len(matches) > 1:raise ValueError('当前楼层对应多个圆环，不能使用缓存')

    def resolve_floor(self, current):
        label = self.request['value']
        saved = self.floor_memory(current.captured_at).get(label)
        snapshot = self.last_snapshot
        after_key = self.fresh(current.captured_at) and snapshot.captured_at > self.request['at']
        if after_key and self.tracker.panel.stable:
            try:
                button = geometry.select_floor(self.last, self.tracker, label, self.options['ocr']['min_confidence'], current.captured_at)
            except ValueError:
                pass
            else:
                self.check_floor_conflicts(label, button, current.captured_at)
                geometry.check_cached_observation(snapshot, current, button, self.settings['max_observation_age_s'])
                source = ('CURRENT_RIM_POSITION_INFERENCE' if geometry.position_floor_indices(self.last, label)
                          else 'CURRENT_RIM_LABEL_MEMORY' if button.text is None else 'NEW_RESULT')
                return snapshot, button, source
        if saved is None:
            raise ValueError(f'楼层{label}暂无有效稳定缓存，等待可靠识别')
        snapshot, result, button = saved
        self.check_floor_conflicts(label, button, current.captured_at)
        age_limit=self.floor_cache_age_limit()
        geometry.check_cached_observation(snapshot, current, button, age_limit)
        prefix='LATCHED' if age_limit is None else 'STABLE'
        return snapshot, button, prefix+('_FLOOR_POSITION_CACHE' if geometry.position_floor_indices(result,label) else '_FLOOR_CACHE')

    def submit_floor(self, now):
        arrival_target = self.state.phase == 'DISPLAY' and self.floor is None
        if (self.state.phase != 'FLOOR' and not arrival_target) or not self.typed or self.typed == '-':return
        typed, self.typed = self.typed, ''
        self.floor_input_due = None
        try:
            value = geometry.normalize_floor_id(typed)
            if arrival_target:
                self.floor = value
                self.arrival_deadline = now+self.settings['arrival_timeout_s']
                self.restore_terminal()
                self.log(f'到站目标设为{value}；仅比较LCD显示，N回零退出。')
                return
            if self.fresh(now):self.remember_floors(self.last_snapshot, self.last, now)
            available = self.floor_memory(now)
            if value not in available:
                if not self.fresh(now) or not self.tracker.panel.stable:
                    raise ValueError('请先等待数字面板稳定')
                geometry.select_floor(self.last, self.tracker, value, self.options['ocr']['min_confidence'], now)
        except ValueError as exc:
            message=str(exc).rstrip('。；')
            self.log(message+('。' if '请等待或重新输入' in message else '；请等待或重新输入。')); return
        self.request = dict(value=value, at=now)
        self.log(f'已选择 {value} 楼。')

    def commit_floor_input(self, now):
        if self.floor_input_due is not None and now >= self.floor_input_due:
            self.submit_floor(now)

    def press(self, current):
        if self.request is None or not self.ready(current.captured_at):
            return False
        settings = self.options['elevator']
        surface = None
        try:
            if self.state.phase == 'CALL':
                snap, button, source = self.resolve_call(current)
                compensation = effective_call_offset(self.settings, self.options)
                compensation_source = ('elevator.target_offset_base_m (shared)' if self.settings.get('call_target_offset_base_m') is None
                                       else 'journey.call_target_offset_base_m (override)')
            else:
                snap, button, source = self.resolve_floor(current)
                compensation = self.options['elevator']['target_offset_base_m']
                compensation_source = 'elevator.target_offset_base_m'
            if self.robot is not None and settings.get('refinement',{}).get('enabled',False):
                surface = geometry.measure_button_surface(snap,button,self.hand_eye,self.cfg,
                                                          vertical_panel=settings['refinement']['vertical_panel'])
                camera_point,base = surface.point_camera,surface.point_base
            else:
                camera_point,base = geometry.point_from_button(snap,button,self.hand_eye,self.cfg)
            point = geometry.offset_target_base(base, compensation)
        except ValueError as exc:
            reported=self.request.get('reported')
            if reported is None or terminal_verbose() and current.captured_at-reported>=1.:
                self.log(str(exc)); self.request['reported'] = current.captured_at
            return False
        if terminal_verbose():
            self.log(f"target={self.request['value']} source={source} image_age_s={current.captured_at-snap.captured_at:.3f} "
                     f"rim={button.ring.center} camera_m={camera_point.tolist()} "
                     f"raw_base_m={None if base is None else base.tolist()} offset_source={compensation_source} "
                     f"offset_base_m={np.asarray(compensation).tolist()} corrected_base_m={None if point is None else point.tolist()}")
        else:
            label={'U':'上行','D':'下行'}.get(self.request['value'],self.request['value']+'楼')
            self.log(f'已定位{label}按键'+('，开始接近。' if self.robot is not None else '。'))
        if self.robot is not None:
            start_pose = self.robot.pose()
            position_only = self.button_orientation=='free'
            offset = self.options['tool']['tip_offset_in_end_m']
            self.skip_start_pose = start_pose.copy()
            self.skip_position_only = position_only
            try:
                if surface is not None:
                    selected = self.request['value']
                    kind = self.KINDS[self.state.phase]
                    def capture():
                        snapshot = self.snapshot(with_arm_feedback=True)
                        if snapshot is not None:self.display = snapshot.color.copy()
                        return snapshot
                    def refresh():
                        self.generation += 1
                        self.last = self.last_snapshot = None
                        self.tracker.reset()
                        return app.reacquire_button(capture,self.worker,kind,selected,self.options,
                            self.hand_eye,self.cfg,self.cancelled,generation=self.generation,
                            settle_s=self.settings['settle_s'],hint_point_base=surface.point_base)
                    app.execute_refined_button(self.robot,surface,settings,offset,self.args.press_stage,refresh,
                        self.cancelled,position_only=position_only,target_offset=compensation,
                        remember_retreat=lambda pose:setattr(self,'skip_retreat',pose.copy()))
                else:
                    level = start_pose if position_only else geometry.level_pose(start_pose,offset)
                    poses = geometry.button_poses(point,level,offset,settings['standoff_m'],settings['press_travel_m'],
                        forward_base=geometry.horizontal_approach_direction(level) if position_only else None)
                    self.skip_retreat = poses['retreat']
                    app.execute_button(self.robot,point,level,settings,offset,self.args.press_stage,self.cancelled,
                                       **({'position_only':True} if position_only else {}))
            except motion.MotionCancelled:
                if not self.skip_requested:
                    self.skip_retreat = self.skip_start_pose = None
                    self.skip_position_only = False
                raise
            except BaseException:
                self.skip_retreat = self.skip_start_pose = None
                self.skip_position_only = False
                raise
            else:
                self.skip_retreat = self.skip_start_pose = None
                self.skip_position_only = False
        phase, selected = self.state.phase, self.request['value']
        self.request = None
        if phase == 'CALL':
            self.enter_board()
        else:
            self.enter_display(selected)
        return True

    def pick(self, current):
        if not self.want_grasp or not self.fresh(current.captured_at) or self.last_snapshot.captured_at <= self.grasp_after:
            return
        from utils.ordinary_grasp import select_best_grasp
        from utils.camera_utils import compose_cam_to_base_transform
        from utils.transforms import transform_grasp_pose_to_base
        best = select_best_grasp(self.last)
        self.want_grasp = False
        if best is None:
            self.log('未找到有效目标，G重试或N跳过。'); return
        snap = self.last_snapshot
        if self.robot is not None:
            T = compose_cam_to_base_transform(snap.T_base_end, self.hand_eye, self.cfg)
            pre = self.cfg.get('grasp_pipeline', {}).get('grasp', {}).get('pregrasp_offset_m', .08)
            target, before = transform_grasp_pose_to_base(best.position, best.tcp_rotation, T, pre)
            success = grasp.pick_and_place(self.robot, target, before, self.cfg, self.place,
                                          self.args.place_duration_s, self.args.release_wait_s, self.cancelled)
            self.invalidate()
            if not success:
                self.log('原驱动未确认抓取成功，G重试或N跳过。'); return
        else:
            self.log('仅预览抓取目标，未执行抓放。')
        self.begin_call()

    def key(self, key, now):
        text = str(key).upper()
        phase = self.state.phase
        if text == 'N' and phase in self.SKIP_PHASES:
            self.skip_stage(); return
        if text == 'R' and phase == 'FLOOR':
            self.invalidate(); self.settle()
            self.state.enter('FLOOR',time.perf_counter())
            self.log('已清空楼层缓存；在当前姿态重新识别，未下发机械臂移动。')
            return
        if phase == 'GRASP':
            if text == 'N':
                self.want_grasp = False; self.begin_call()
            elif text == 'G' and not self.want_grasp:
                self.invalidate(); self.want_grasp = True; self.grasp_after = time.perf_counter()
        elif phase == 'WAIT_CALL' and text == 'C':
            # A fresh stationary view is required after any manual relocation.
            # Keep the head posture; start the search timer only after settling.
            self.invalidate(); self.settle()
            self.state.enter('CALL', time.perf_counter())
            self.log('搜寻外呼面板。U 上行，D 下行，N 跳过。')
        elif phase == 'BOARD' and text == 'C':
            # CALL already restored the floor-button view. Boarding must
            # invalidate all visual evidence without repeating the same move.
            self.invalidate(); self.settle()
            self.state.enter('FLOOR', time.perf_counter())
            self.enable_floor_terminal()
            self.log(f'输入楼层，停顿 {self.floor_input_delay_s:g} 秒自动提交。N 跳过。')
        elif phase == 'WAIT_DISPLAY' and text == 'C':
            self.begin_display()
        elif phase == 'CALL' and text in ('U', 'D'):
            if self.ready(now):
                if self.fresh(now) and self.last.state == 'STABLE' and len(self.last.pair) == 2:
                    self.call_cache = self.last_snapshot, self.last
                self.request = dict(value=text, at=now)
            else:
                self.log('请先等待两键面板稳定。')
        elif phase == 'FLOOR' or phase == 'DISPLAY' and self.floor is None:
            if text in ('ENTER', '\n', '\r'):
                self.submit_floor(now)  # Optional shortcut, no longer required.
            elif text == 'BACKSPACE':
                self.typed = self.typed[:-1]
                self.request = None
                self.floor_input_due = now+self.floor_input_delay_s if self.typed not in ('', '-') else None
            elif len(text) == 1 and text in '-0123456789':
                self.request = None
                proposed = self.typed+text
                if not re.fullmatch(r'-?[0-9]{0,2}', proposed):
                    self.typed = ''; self.floor_input_due = None
                    self.log('楼层需为-99到99；输入已清空，请重新输入。'); return
                self.typed = proposed
                self.floor_input_due = now+self.floor_input_delay_s if proposed != '-' else None

    def receive(self, now):
        received = self.worker.receive()
        if received is None:
            return
        snap, result = received
        if snap.generation != self.generation or not 0 <= now-snap.captured_at <= self.settings['max_observation_age_s']:
            return
        self.last_snapshot, self.last = snap, result
        confirmed = None
        if self.state.phase == 'CALL':
            if result.state == 'STABLE' and len(result.pair) == 2:
                self.call_cache = snap, result
                confirmed = '外呼上下键'
            elif result.state in ('EXTRA_BUTTONS', 'AMBIGUOUS', 'CLUTTERED', 'INVALID_FRAME', 'INVALID_TIMESTAMP'):
                self.call_cache = None
        elif self.state.phase == 'FLOOR':
            self.tracker.update(result, self.generation, camera_pose=snap.T_base_end, now=now)
            self.remember_floors(snap, result, now)
            if self.floor_cache:
                confirmed = f'首批楼层 {sorted(self.floor_cache)}'
        if (confirmed is not None and self.state.started_at is not None
                and self.recognition_logged_generation != self.generation):
            self.log(f'{confirmed}已确认，用时 {now-self.state.started_at:.2f} 秒。')
            self.recognition_logged_generation = self.generation

    def draw(self, current):
        from . import perception
        if self.fresh(current.captured_at) and self.state.phase in ('CALL', 'DISPLAY', 'FLOOR'):
            if self.rendered_result is not self.last:
                if self.state.phase == 'CALL':
                    from .call_panel_261112 import draw_result
                    self.rendered_panel = draw_result(self.last_snapshot.color, self.last)
                elif self.state.phase == 'DISPLAY':
                    from .floor_display_261113 import draw_result
                    self.rendered_panel = draw_result(self.last_snapshot.color, self.last)
                else:
                    self.rendered_panel = perception.draw_result(self.last_snapshot.color, self.last)
                self.rendered_result = self.last
            image = self.rendered_panel.copy()
        else:
            image = current.color.copy()
            if self.state.phase == 'GRASP' and self.last is not None:
                from utils.ordinary_grasp import select_best_grasp
                if self.grasp_view is None:
                    self.grasp_view = grasp.original_main()
                image = self.grasp_view._render_display(image, self.last, select_best_grasp(self.last), 'G=grasp/place N=skip')
        remaining = '-' if self.state.deadline is None else f'{max(0,self.state.deadline-current.captured_at):.1f}s'
        # Standalone visualizers have a "NO ROBOT" footer. Replace it with
        # this application's actual execution mode when reusing their images.
        cv2.rectangle(image, (0, image.shape[0]-86), (image.shape[1], image.shape[0]), (0,0,0), -1)
        controls = {
            'WAIT_CALL': 'C=start UP/DOWN detection N=skip call | no search timeout',
            'BOARD': 'C=boarded and stationary; continue to floor buttons',
            'WAIT_DISPLAY': f'C=start LCD detection N=home/exit | target={self.floor}',
        }.get(self.state.phase, f'G=grasp N=skip U/D=call | input={self.typed} target={self.floor}')
        if self.state.phase == 'CALL':
            saved = self.call_memory(current.captured_at)
            age = '-' if saved is None else f'{current.captured_at-saved[0].captured_at:.2f}s'
            controls = f'U=UP D=DOWN N=skip call | stable cache age={age}'
        elif self.state.phase == 'FLOOR':
            remaining_input = '-' if self.floor_input_due is None else f'{max(0,self.floor_input_due-current.captured_at):.1f}s'
            mode='until update' if self.floor_cache_age_limit() is None else 'timed'
            controls = f'floor digits=auto N=skip press R=clear | input={self.typed} submit={remaining_input} memory={sorted(self.floor_memory(current.captured_at))} ({mode})'
        elif self.state.phase == 'DISPLAY':
            controls = (f'LCD arrival target digits=auto | input={self.typed} N=home/exit' if self.floor is None
                        else f'LCD arrival target={self.floor} | N=skip LCD and home/exit')
        lines = [f"JOURNEY {self.state.phase} | retry={self.state.retry_index}/{self.state.RETRY_COUNT} "
                 f"search={remaining} | {'EXECUTE' if self.robot else 'PREVIEW'}",
                 controls,
                 f'press={self.args.press_stage} | Q=home/exit SPACE/ESC=stop']
        for i, line in enumerate(lines):
            xy = (8, image.shape[0]-68+20*i)
            cv2.putText(image, line, xy, cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 3)
            cv2.putText(image, line, xy, cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        self.display = image
        cv2.imshow(self.window, image)

    def run(self):
        from drivers.camera import make_camera
        from utils.camera_utils import load_hand_eye
        ocr = self.options['ocr']
        infer = dict(device=self.args.device, roi=None, max_side=1200, geometry_only=False, ocr_threads=2,
                     verbose=terminal_verbose(),
                     ocr_min_score=ocr['min_confidence'], ocr_device=ocr['device'], ocr_preprocessing=ocr['preprocessing'],
                     ocr_gpu_index=ocr['gpu_index'], ocr_model_dir=ROOT/ocr['model_dir'], workflow_evidence=True,
                     ocr_buttons_per_update=ocr.get('buttons_per_update',4), ocr_refresh_s=ocr.get('refresh_s',2.))
        try:
            began=time.perf_counter()
            self.log('初始化相机和识别。')
            self.worker = JourneyWorker(self.cfg, infer, self.settings)
            camera_began=time.perf_counter()
            self.camera = make_camera(self.cfg); self.camera.open()
            camera_ready=time.perf_counter()
            self.log(f'[Startup] camera_open_s={camera_ready-camera_began:.2f}',verbose_only=True)
            self.K, self.D = np.asarray(self.camera.K).copy(), np.asarray(self.camera.D).copy()
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            def show_startup(color):
                self.display=color.copy()
                cv2.putText(self.display, 'Loading shared YOLO/OCR | Q=exit', (8,32),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
            self.worker.wait_ready(on_wait=app.startup_preview(self.camera,show_startup,self.cancelled))
            self.log(f'[Startup] vision_remaining_s={time.perf_counter()-camera_ready:.2f} '
                     f'vision_and_camera_s={time.perf_counter()-began:.2f}',verbose_only=True)
            self.hand_eye = None
            self.place = grasp.placement_pose(self.cfg, self.args.place_xyz_m, self.args.place_rpy_deg)
            if self.args.execute:
                self.hand_eye, mode = load_hand_eye(ROOT, str(self.cfg['camera']['type']).lower())
                if self.hand_eye is None or mode != 'eye_in_hand' or np.asarray(self.hand_eye).shape != (4,4) or not np.isfinite(self.hand_eye).all():
                    raise RuntimeError('需要有效eye_in_hand手眼标定')
                robot_cfg=app.robot_runtime_config(self.cfg,self.options)
                self.robot = motion.Robot(robot_cfg); self.robot.start(gripper_posvel=False)
                self.log('移动到夹取预备位。')
                pose_began=time.perf_counter()
                self.grasp_view = grasp.enter_ready_pose(self.robot,self.cfg,self.cancelled)
                pose_reached=time.perf_counter()
                self.invalidate(); self.settle()
                self.log(f'[Startup] ready_move_s={pose_reached-pose_began:.2f} '
                         f'ready_settle_s={time.perf_counter()-pose_reached:.2f} '
                         f'total_ready_s={time.perf_counter()-began:.2f}',verbose_only=True)
            self.log('G抓放，N跳过。')
            while True:
                if self.state.phase == 'DONE':return 0
                current = self.snapshot()
                if current is None:
                    if self.cancelled():
                        if self.skip_requested:self.skip_stage()
                        else:raise motion.MotionCancelled('用户停止')
                    continue
                now = current.captured_at
                self.receive(now)
                queued_request=self.request
                # Consume N/Q before dispatching a queued button press or head retry.
                self.draw(current)
                key = cv2.waitKey(1) & 255
                if key in (27,32) or cv2.getWindowProperty(self.window,cv2.WND_PROP_VISIBLE) < 1:
                    self.quit_home_requested = False
                    raise motion.MotionCancelled('用户停止')
                if key in (ord('q'),ord('Q')):
                    self.quit_home_requested = True
                    raise motion.MotionCancelled('Q回零结束任务')
                if key in (10,13): self.stage_action(self.key,'ENTER',time.perf_counter())
                elif key in (8,127): self.stage_action(self.key,'BACKSPACE',time.perf_counter())
                elif key != 255: self.stage_action(self.key,chr(key),time.perf_counter())
                if self.state.phase == 'DONE':return 0
                if current.generation != self.generation: continue
                value = self.read_terminal()
                if value is not None:
                    if value.lower() in ('q','stop'):
                        self.quit_home_requested = value.lower() == 'q'
                        raise motion.MotionCancelled('终端退出')
                    if self.state.phase == 'FLOOR' or self.state.phase == 'DISPLAY' and self.floor is None:
                        if value in ('\n','\r'):value='ENTER'
                        elif value in ('\b','\x7f'):value='BACKSPACE'
                        elif len(value) > 1:
                            self.typed = value; value='ENTER'
                    self.stage_action(self.key,value,time.perf_counter())
                self.commit_floor_input(time.perf_counter())
                if self.state.phase == 'DONE':return 0
                if current.generation != self.generation: continue
                if self.state.phase == 'GRASP':
                    self.pick(current)
                    if current.generation != self.generation: continue
                if queued_request is not None and self.request is queued_request:
                    if self.stage_action(self.press, current): continue
                    if self.request is None:continue
                    if now-self.request['at'] > self.options['elevator']['floor_wait_s']:
                        self.log('本次请求定位超时，请重新选择；未使用旧三维目标。'); self.request = None
                if self.state.phase == 'DISPLAY' and self.ready(now):
                    if self.floor is not None and display_floor_id(self.last.current_floor) == self.floor:
                        self.log(f'LCD稳定显示目标楼层 {self.floor}，任务结束后自动回零退出；未判定停车或开门。')
                        self.home = True; self.state.enter('DONE', now); return 0
                if self.arrival_deadline is not None and now >= self.arrival_deadline:
                    self.log('等待到站超时，结束任务并回零退出。'); self.home = True; return 2
                delta = self.state.tick(now, self.ready(now))
                if delta == 'HOME':
                    self.log('三轮均无可靠面板，结束任务并回零退出。'); self.home = True; return 2
                if delta is not None:
                    joints = (np.degrees(self.robot._arm_feedback()[0]) if self.robot is not None else self.preview_q)
                    goal = relative_head_target(joints, delta, self.arm_view_limits(), last_goal_deg=self.preview_q)
                    if goal is None:
                        self.log('J2/J3 已无剩余抬头行程，结束搜寻并回零。' if self.robot is not None
                                 else 'J2/J3 已无剩余抬头行程，结束预览。')
                        self.home = True; self.state.enter('TIMEOUT', now); return 2
                    step = goal[1:3] - joints[1:3]
                    limited = '，本次按限位缩短' if not np.allclose(step, delta, rtol=0., atol=1e-8) else ''
                    self.log(f'{self.state.timeout_s:g} 秒未找到，再抬头 J2 {step[0]:+.3g}°、J3 {step[1]:+.3g}°'
                             f'（{self.state.retry_index}/{self.state.RETRY_COUNT}{limited}）。')
                    self.log(f'目标六轴 deg={goal.tolist()}',verbose_only=True)
                    phase = self.state.phase
                    self.stage_action(self.head, goal)
                    if self.state.phase == phase:self.state.retry_complete(time.perf_counter())
                    continue
                kind = self.KINDS.get(self.state.phase)
                if kind is not None and self.worker.pending is None:
                    self.worker.submit(current,kind=kind)
        except (motion.TrajectoryPlanningError,motion.TrajectorySendError,motion.HeadJointLimitError,app.ButtonRefinementError,
                app.PerceptionError,TimeoutError,CameraFrameError) as exc:
            self.home = True
            self.request = None
            self.skip_requested = False
            self.skip_retreat = self.skip_start_pose = None
            self.skip_position_only = False
            self.log('任务中止，尝试回零。')
            raise
        except motion.MotionCancelled:
            self.home = self.quit_home_requested
            if self.home:
                self.log('Q结束任务：停止当前指令，SDK回零后退出。' if self.robot is not None
                         else 'Q退出：未连接机器人，不发送回零动作。')
            else:
                self.log('已停止当前动作，退出任务。')
            return 0 if self.quit_home_requested else 130
        finally:
            app.finish_elevator(self.robot,self.camera,self.worker,self.home,restore_terminal=self.restore_terminal)
