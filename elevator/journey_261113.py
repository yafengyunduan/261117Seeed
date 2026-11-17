"""One camera/robot/model session for grasp -> call -> floor -> arrival.

Existing algorithms and SDK actions are imported, not replaced. No device is
created on import. Walking/boarding is an operator handoff, not navigation.
"""
from __future__ import annotations

import argparse
import copy
import json
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

from . import app, geometry, grasp, motion

ROOT = Path(__file__).resolve().parents[1]


class SearchState:
    PLANS = {'CALL': (-15., -15.), 'FLOOR': (-15., -15.), 'DISPLAY': (-15., -10.)}

    def __init__(self, timeout_s=10.):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('search timeout must be positive')
        self.timeout_s = timeout_s
        self.phase, self.retry_index, self.deadline = 'GRASP', 0, None
        self.waiting_retry = False

    def enter(self, phase, now):
        if phase not in (*self.PLANS, 'GRASP', 'WAIT_CALL', 'BOARD', 'DONE', 'TIMEOUT'):
            raise ValueError('unknown phase')
        self.phase, self.retry_index = phase, 0
        self.waiting_retry = False
        self.deadline = now+self.timeout_s if phase in self.PLANS else None

    def retry_complete(self, now):
        self.waiting_retry = False
        self.deadline = now+self.timeout_s

    def tick(self, now, ready):
        if self.phase not in self.PLANS or self.waiting_retry:
            return None
        if ready:
            self.deadline = None
            return None
        if self.deadline is None:
            self.deadline = now+self.timeout_s
        if now < self.deadline:
            return None
        if self.retry_index == 2:
            self.phase, self.deadline = 'TIMEOUT', None
            return 'HOME'
        delta = self.PLANS[self.phase][self.retry_index]
        self.retry_index += 1
        self.deadline, self.waiting_retry = None, True
        return delta


def relative_j3_target(robot, delta, j3_bounds=(-3.14, 0.)):
    q = np.asarray(robot.arm.get_state()[0][:6], float)
    if q.shape != (6,) or not np.isfinite(q).all() or not math.isfinite(delta):
        raise ValueError('invalid joint feedback/delta')
    goal = np.degrees(q)
    goal[2] += delta
    if not j3_bounds[0] <= np.radians(goal[2]) <= j3_bounds[1]:
        raise ValueError(f'J3 target outside URDF limits: {goal[2]:.3f}deg')
    return goal


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
        from .panel_261112 import FastPanelPerception
        from .call_panel_261112 import CallPanelDetector
        from .floor_display_261113 import FloorDisplayReader
        panel = FastPanelPerception(cfg, argparse.Namespace(**settings))
        call = CallPanelDetector(required=3)
        display = FloorDisplayReader(shared_display_ocr(panel), min_confidence=journey['display_min_confidence'])
        panel.warmup(cfg['camera'].get('color_height', 720), cfg['camera'].get('color_width', 1280))
        print('[Journey OCR shared] '+json.dumps(panel.ocr.backend_info, ensure_ascii=False), flush=True)
        connection.send(('ready',))
        previous = None
        while True:
            request = requests.get()
            if request is None:
                break
            kind, frame_id, at, color, depth, K, generation = request
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
                result = panel.process(color, captured_at=at)
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
    def __init__(self, cfg, settings, journey):
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.requests = ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_worker, args=(child, self.requests, cfg, settings, journey), daemon=True)
        self.process.start(); child.close()
        self.pending, self.submitted_at, self.timeout = None, 0., 20.

    def submit(self, snapshot, kind='panel'):
        if kind not in ('grasp', 'panel', 'call', 'display'):
            raise ValueError('unknown inference task')
        if self.pending is None:
            self.requests.put_nowait((kind, snapshot.frame_id, snapshot.captured_at, snapshot.color,
                                     snapshot.depth_mm if kind == 'grasp' else None,
                                     snapshot.K if kind == 'grasp' else None, snapshot.generation))
            self.pending, self.submitted_at = snapshot, time.perf_counter()


def load_settings(path, options=None):
    value = app.read_yaml(path)
    for name, lo, hi in (('search_timeout_s', 1., 60.), ('settle_s', .1, 2.), ('max_settle_s', 1., 15.),
                         ('arrival_timeout_s', 10., 3600.), ('max_observation_age_s', .1, 2.),
                         ('display_min_confidence', .5, 1.)):
        number = float(value[name])
        if not math.isfinite(number) or not lo <= number <= hi:
            raise ValueError(f'invalid {name}')
        value[name] = number
    # The requested retry schedules are explicit, not accumulated across stages.
    for key, plan in (('call_j3_retry_deg', [-15, -15]), ('floor_j3_retry_deg', [-15, -15]),
                      ('display_j3_retry_deg', [-15, -10])):
        if value[key] != plan:
            raise ValueError(f'{key} expected {plan}')
    # Current workflows have one pose source. Older custom journey files can
    # still supply their display pose when no shared initial_poses is present.
    initial=(options or {}).get('initial_poses',{})
    head = np.asarray(initial.get('display_view_deg',value.get('display_view_deg')), float)
    if head.shape != (6,) or not np.isfinite(head).all() or not -179.9 <= head[2] <= 0:
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

    def __init__(self, cfg, options, settings, args):
        self.cfg, self.options, self.settings, self.args = app.apply_initial_poses(cfg,options), options, settings, args
        initial=options.get('initial_poses',{})
        self.views={'CALL':list(initial.get('call_view_deg',options['startup_view']['joints_deg'][:6])),
                    'FLOOR':list(initial.get('floor_view_deg',options['startup_view']['joints_deg'][:6])),
                    'DISPLAY':list(initial.get('display_view_deg',settings['display_view_deg']))}
        self.state = SearchState(settings['search_timeout_s'])
        self.robot = self.camera = self.worker = None
        self.window = 'Elevator journey | G/N -> C -> U/D -> C -> floor -> arrival'
        self.display = self.last = self.last_snapshot = None
        self.call_cache = None
        self.floor_cache = {}
        self.grasp_view = None
        self.rendered_result = self.rendered_panel = None
        self.generation = self.frame_id = self.failures = 0
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

    def log(self, message):
        print(f'[Journey:{self.state.phase}] {message}', flush=True)

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
                and self.state.phase in ('WAIT_CALL', 'CALL', 'FLOOR', 'DISPLAY') and not self.skip_in_progress):
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
        self.log('等待人工进入轿厢、停在面板前并站稳；按 C 继续。接近模式需人工处理实际呼叫。')

    def enter_display(self, floor):
        self.floor = floor
        self.head(self.views['DISPLAY'])
        self.state.enter('DISPLAY', time.perf_counter())
        if floor is None:
            self.arrival_deadline = None
            self.enable_floor_terminal()
            self.log('开始LCD楼层识别；请输入到站目标楼层（仅用于显示比较），无需回车；N回零退出。')
        else:
            self.restore_terminal()
            self.arrival_deadline = time.perf_counter()+self.settings['arrival_timeout_s']
            self.log(f'等待独立稳定的当前楼层={floor}；N跳过识别并回零退出。')

    def skip_stage(self):
        phase = self.state.phase
        if phase not in ('WAIT_CALL', 'CALL', 'FLOOR', 'DISPLAY') or self.skip_in_progress:return False
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
                self.log('N跳过楼层按键，转入当前楼层显示识别。')
                self.enter_display(target)
            else:
                self.restore_terminal()
                self.home = True
                self.state.enter('DONE', time.perf_counter())
                self.log('N跳过最后的LCD识别阶段，回零退出。' if self.robot is not None else 'N跳过最后的LCD识别阶段，退出预览。')
        finally:
            self.skip_in_progress = False
        return True

    def enable_floor_terminal(self):
        """Only the floor stage uses per-key terminal input; restore on leaving."""
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

    def snapshot(self):
        color, depth = self.camera.get_frame()
        if color is None or depth is None:
            self.failures += 1
            if self.failures >= 15:
                raise RuntimeError('连续15次RGB-D获取失败')
            return None
        self.failures = 0; self.frame_id += 1
        pose = np.eye(4) if self.robot is None else self.robot.pose()
        return geometry.Snapshot(self.frame_id, time.perf_counter(), color.copy(), depth.copy(),
                                 self.K, self.D, pose, self.generation)

    def settle(self):
        until = time.perf_counter()+self.settings['max_settle_s']
        since = anchor = None
        while time.perf_counter() < until:
            snap = self.snapshot()
            if snap is not None:
                self.display = snap.color.copy()
                if anchor is None:
                    anchor, since = snap.T_base_end, snap.captured_at
                angle = np.arccos(np.clip((np.trace(anchor[:3, :3].T@snap.T_base_end[:3, :3])-1)/2, -1, 1))
                if np.linalg.norm(anchor[:3, 3]-snap.T_base_end[:3, 3]) > .002 or angle > np.radians(.5):
                    anchor, since = snap.T_base_end, snap.captured_at
                if snap.captured_at-since >= self.settings['settle_s']:
                    return
            if self.cancelled():
                raise motion.MotionCancelled('用户停止')
        raise TimeoutError(f'六轴反馈未静止；未开始{self.state.timeout_s:g}秒识别窗口')

    def head(self, joints):
        self.preview_q = np.asarray(joints[:6], float).copy()
        if self.robot is not None:
            values = list(joints[:6])+[self.options['gripper']['closed_angle_deg']]
            self.robot.head_up(values, self.options['startup_view']['duration_s'], self.cancelled,
                               closed_tolerance=self.options['gripper']['closed_tolerance_rad'],
                               wait_for_arrival=self.options['startup_view']['wait_for_arrival'])
        self.invalidate(); self.settle()

    def begin_call(self):
        if self.robot is not None:
            self.robot.switch_gripper_to_posvel(self.cancelled)
        self.head(self.views['CALL'])
        # No inference kind or search deadline until the operator confirms.
        self.state.enter('WAIT_CALL', time.perf_counter())
        self.log('已恢复抬头姿态；按 C 开始识别上行/下行外呼面板。等待期间不计搜索超时；Q回零退出。')

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
        self.log(f'已选择楼层{value}；自动提交，核对已确认缓存及当前场景后执行；无需当前圆环为绿色。')

    def commit_floor_input(self, now):
        if self.floor_input_due is not None and now >= self.floor_input_due:
            self.submit_floor(now)

    def press(self, current):
        if self.request is None or not self.ready(current.captured_at):
            return False
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
            camera_point, base = geometry.point_from_button(snap, button, self.hand_eye, self.cfg)
            point = geometry.offset_target_base(base, compensation)
        except ValueError as exc:
            if current.captured_at-self.request.get('reported', -math.inf) >= 1.:
                self.log(str(exc)); self.request['reported'] = current.captured_at
            return False
        self.log(f"target={self.request['value']} source={source} image_age_s={current.captured_at-snap.captured_at:.3f} "
                 f"rim={button.ring.center} camera_m={camera_point.tolist()} "
                 f"raw_base_m={None if base is None else base.tolist()} offset_source={compensation_source} "
                 f"offset_base_m={np.asarray(compensation).tolist()} corrected_base_m={None if point is None else point.tolist()}")
        if self.robot is not None:
            start_pose = self.robot.pose()
            position_only = self.button_orientation=='free'
            level = start_pose if position_only else geometry.level_pose(start_pose, self.options['tool']['tip_offset_in_end_m'])
            poses = geometry.button_poses(point, level, self.options['tool']['tip_offset_in_end_m'],
                self.options['elevator']['standoff_m'], self.options['elevator']['press_travel_m'],
                forward_base=geometry.horizontal_approach_direction(level) if position_only else None)
            self.skip_retreat = poses['retreat']
            self.skip_start_pose = start_pose.copy()
            self.skip_position_only = position_only
            try:
                app.execute_button(self.robot, point, level, self.options['elevator'],
                                   self.options['tool']['tip_offset_in_end_m'], self.args.press_stage, self.cancelled,
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
        if text == 'N' and phase in ('WAIT_CALL', 'CALL', 'FLOOR', 'DISPLAY'):
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
            self.log('开始外呼面板识别；稳定后 U=上行，D=下行；机身须站稳。')
        elif phase == 'BOARD' and text == 'C':
            # CALL already restored the floor-button view. Boarding must
            # invalidate all visual evidence without repeating the same move.
            self.invalidate(); self.settle()
            self.state.enter('FLOOR', time.perf_counter())
            self.enable_floor_terminal()
            self.log(f'在窗口或终端输入楼层，无需回车；停顿{self.floor_input_delay_s:g}秒自动提交，支持负数及两位数。')
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
        if self.state.phase == 'CALL':
            if result.state == 'STABLE' and len(result.pair) == 2:
                self.call_cache = snap, result
            elif result.state in ('EXTRA_BUTTONS', 'AMBIGUOUS', 'CLUTTERED', 'INVALID_FRAME', 'INVALID_TIMESTAMP'):
                self.call_cache = None
        if self.state.phase == 'FLOOR':
            self.tracker.update(result, self.generation, camera_pose=snap.T_base_end, now=now)
            self.remember_floors(snap, result, now)

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
        }.get(self.state.phase, f'G=grasp N=skip U/D=call | input={self.typed} target={self.floor}')
        if self.state.phase == 'CALL':
            saved = self.call_memory(current.captured_at)
            age = '-' if saved is None else f'{current.captured_at-saved[0].captured_at:.2f}s'
            controls = f'U=UP D=DOWN N=skip call | stable cache age={age}'
        elif self.state.phase == 'FLOOR':
            remaining_input = '-' if self.floor_input_due is None else f'{max(0,self.floor_input_due-current.captured_at):.1f}s'
            mode='until update' if self.floor_cache_age_limit() is None else 'timed'
            controls = f'floor digits=auto N=skip R=clear | input={self.typed} submit={remaining_input} memory={sorted(self.floor_memory(current.captured_at))} ({mode})'
        elif self.state.phase == 'DISPLAY':
            controls = (f'LCD arrival target digits=auto | input={self.typed} N=home/exit' if self.floor is None
                        else f'LCD arrival target={self.floor} | N=skip LCD and home/exit')
        lines = [f"JOURNEY {self.state.phase} | retry={self.state.retry_index}/2 search={remaining} | {'EXECUTE' if self.robot else 'PREVIEW'}",
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
                     ocr_min_score=ocr['min_confidence'], ocr_device=ocr['device'], ocr_preprocessing=ocr['preprocessing'],
                     ocr_gpu_index=ocr['gpu_index'], ocr_model_dir=ROOT/ocr['model_dir'], workflow_evidence=True,
                     ocr_buttons_per_update=ocr.get('buttons_per_update',4), ocr_refresh_s=ocr.get('refresh_s',2.))
        try:
            self.log('初始化共享YOLO/CUDA OCR，完成后才开始阶段计时。')
            self.worker = JourneyWorker(self.cfg, infer, self.settings)
            self.camera = make_camera(self.cfg); self.camera.open()
            self.K, self.D = np.asarray(self.camera.K).copy(), np.asarray(self.camera.D).copy()
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            def initializing():
                snap = self.snapshot()
                if snap is not None:
                    self.display = snap.color.copy()
                    cv2.putText(self.display, 'Loading shared YOLO/OCR | Q=exit', (8,32),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
                if self.cancelled():
                    raise motion.MotionCancelled('初始化取消')
            self.worker.wait_ready(on_wait=initializing)
            self.hand_eye = None
            self.place = grasp.placement_pose(self.cfg, self.args.place_xyz_m, self.args.place_rpy_deg)
            if self.args.execute:
                self.hand_eye, mode = load_hand_eye(ROOT, str(self.cfg['camera']['type']).lower())
                if self.hand_eye is None or mode != 'eye_in_hand' or np.asarray(self.hand_eye).shape != (4,4) or not np.isfinite(self.hand_eye).all():
                    raise RuntimeError('需要有效eye_in_hand手眼标定')
                cfg = copy.deepcopy(self.cfg); cfg['trajectory'] = self.options['trajectory']
                self.robot = motion.Robot(cfg); self.robot.start(gripper_posvel=False)
                self.log('进入原夹取预备姿态，完成后等待G抓放或N跳过。')
                module, controller = grasp.original_motion_adapter(self.robot, self.cancelled)
                module._move_ready(controller, grasp.ready_settings(self.cfg))
                self.invalidate(); self.settle()
            self.log(f'G抓取后放置，N跳过；机械狗须站稳。{self.state.timeout_s:g}秒计时不含模型初始化和动作。')
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
                if self.state.phase == 'GRASP':
                    self.pick(current)
                    if current.generation != self.generation: continue
                if self.request is not None:
                    if self.stage_action(self.press, current): continue
                    if self.request is None:continue
                    if now-self.request['at'] > self.options['elevator']['floor_wait_s']:
                        self.log('本次请求定位超时，请重新选择；未使用旧三维目标。'); self.request = None
                if self.state.phase == 'DISPLAY' and self.ready(now):
                    if self.floor is not None and display_floor_id(self.last.current_floor) == self.floor:
                        self.log(f'LCD稳定显示目标楼层 {self.floor}，按任务条件回零退出；未判定停车或开门。')
                        self.home = True; self.state.enter('DONE', now); return 0
                if self.arrival_deadline is not None and now >= self.arrival_deadline:
                    self.log('等待到站超时，回零退出。'); self.home = True; return 2
                delta = self.state.tick(now, self.ready(now))
                if delta == 'HOME':
                    self.log('三轮均无可靠面板，回零退出。'); self.home = True; return 2
                if delta is not None:
                    if self.robot is not None:
                        goal = relative_j3_target(self.robot, delta)
                    else:
                        goal = self.preview_q.copy(); goal[2] += delta
                    self.log(f'{self.state.timeout_s:g}秒无稳定面板，相对当前J3 {delta:g}deg，目标六轴={goal.tolist()}')
                    phase = self.state.phase
                    self.stage_action(self.head, goal)
                    if self.state.phase == phase:self.state.retry_complete(time.perf_counter())
                    continue
                self.draw(current)
                key = cv2.waitKey(1) & 255
                if key in (27,32) or cv2.getWindowProperty(self.window,cv2.WND_PROP_VISIBLE) < 1:
                    self.quit_home_requested = False
                    raise motion.MotionCancelled('用户停止')
                if key in (ord('q'),ord('Q')):
                    self.quit_home_requested = True
                    raise motion.MotionCancelled('Q回零退出')
                if key in (10,13): self.stage_action(self.key,'ENTER',time.perf_counter())
                elif key in (8,127): self.stage_action(self.key,'BACKSPACE',time.perf_counter())
                elif key != 255: self.stage_action(self.key,chr(key),time.perf_counter())
                if self.state.phase == 'DONE':return 0
                value = self.read_terminal()
                if value is not None:
                    if value.lower() in ('q','stop'):
                        self.quit_home_requested = value.lower() == 'q'
                        raise motion.MotionCancelled('终端退出')
                    if self.state.phase == 'FLOOR' or self.state.phase == 'DISPLAY' and self.floor is None:
                        if value in ('\n','\r'):self.key('ENTER',time.perf_counter())
                        elif value in ('\b','\x7f'):self.key('BACKSPACE',time.perf_counter())
                        elif len(value) > 1:
                            self.typed = value; self.key('ENTER',time.perf_counter())
                        else:self.key(value,time.perf_counter())
                    else: self.stage_action(self.key,value,time.perf_counter())
                self.commit_floor_input(time.perf_counter())
                if self.state.phase == 'DONE':return 0
                if current.generation != self.generation: continue
                kind = self.KINDS.get(self.state.phase)
                if kind is not None and self.worker.pending is None:
                    self.worker.submit(current,kind=kind)
        except motion.TrajectoryPlanningError:
            # The adapter has already stopped/held successfully; the failed
            # plan never started a sender. Joint-space safe_home does not
            # repeat IK for the rejected Cartesian target.
            self.home = True
            self.request = None
            self.skip_requested = False
            self.skip_retreat = self.skip_start_pose = None
            self.skip_position_only = False
            self.log('IK/轨迹规划失败且本次轨迹未发送；先调用SDK safe_home回零，再失能断开。')
            raise
        except motion.MotionCancelled:
            self.home = self.quit_home_requested
            if self.home:
                self.log('Q正常退出：先停止当前指令，再SDK回零并断开。' if self.robot is not None
                         else 'Q退出：未连接机器人，不发送回零动作。')
            else:
                self.log('已停止；不追加自动回零动作。')
            return 0 if self.quit_home_requested else 130
        finally:
            active = sys.exc_info()[0] is not None
            errors = []
            try:self.restore_terminal()
            except BaseException as exc:errors.append(f'terminal: {exc}')
            if self.robot is not None:
                try: self.robot.close(home=self.home)
                except BaseException as exc: errors.append(f'robot: {exc}')
            if self.camera is not None:
                try: self.camera.close()
                except BaseException as exc: errors.append(f'camera: {exc}')
            if self.worker is not None:
                try: self.worker.close()
                except BaseException as exc: errors.append(f'worker: {exc}')
            cv2.destroyAllWindows()
            if errors:
                print('[Cleanup] '+'; '.join(errors),file=sys.stderr,flush=True)
                if not active: raise RuntimeError('; '.join(errors))
