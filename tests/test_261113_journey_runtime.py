"""Journey.run integration checks with fake camera/worker/robot and virtual time."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import os
import pty
import termios
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from elevator import journey_261113 as j
from elevator.perception import Ring, ButtonObservation


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class FakeCamera:
    K = np.array([[100., 0., 80.], [0., 100., 60.], [0., 0., 1.]])
    D = np.zeros(5)
    def __init__(self, clock, events): self.clock, self.events = clock, events
    def open(self): self.events.append(('camera_open',))
    def warm_up(self, count): self.clock.now += count*.1
    def get_frame(self):
        self.clock.now += .1
        return np.full((120, 160, 3), 70, np.uint8), np.full((120, 160), 300, np.uint16)
    def close(self): self.events.append(('camera_close',))


class FakeRobot:
    def __init__(self, events, fail_cleanup=False):
        self.events, self.fail_cleanup = events, fail_cleanup
        self.q = np.zeros(6)
        self.T = np.eye(4); self.T[:3, 3] = [.25, 0., .3]
        self.arm = NS(get_state=lambda: (np.r_[self.q, 0.], np.zeros(7), np.zeros(7)))
    def start(self, **kwargs): self.events.append(('robot_start', kwargs))
    def pose(self): return self.T.copy()
    def arm_position_limits(self):
        return np.array([[-2.8,2.8],[-3.14,0.],[-3.14,0.],[-1.87,1.57],[-1.57,1.57],[-3.14,3.14]])
    def _arm_feedback(self): return self.q.copy(),np.zeros(6)
    def switch_gripper_to_posvel(self, cancelled): self.events.append(('switch_gripper',))
    def head_up(self, values, duration, cancelled, **kwargs):
        self.q = np.radians(values[:6]); self.events.append(('head', list(values)))
        self.T = np.eye(4); self.T[:3, 3] = [.25, 0., .3]
    def move(self, target, duration, cancelled, *, settle=None):
        self.T = np.asarray(target).copy(); self.events.append(('move', self.T[:3, 3].tolist()))
    def move_ready_in_stages(self, target, duration, joint4_duration, cancelled):
        self.T=np.asarray(target).copy()
        self.events.append(('staged_ready',self.T[:3,3].tolist(),duration,joint4_duration))
    def move_position(self,target,duration,cancelled,*,tip_offset=(0,0,0)):
        from utils.transforms import pose6d_to_mat4
        self.events.append(('position_move',np.asarray(target).tolist()))
        T=pose6d_to_mat4(0.,0.,0.,.05,-.6,.1)
        T[:3,3]=np.asarray(target)-T[:3,:3]@np.asarray(tip_offset)
        self.move(T,duration,cancelled)
        return T
    def stop(self): self.events.append(('stop',))
    def finish_task(self, *, home):
        self.events.append(('robot_finish', home))
        if self.fail_cleanup: raise RuntimeError('synthetic home/hold failure')


class FakeTracker:
    def __init__(self): self.panel = NS(stable=False)
    def reset(self): self.panel.stable = False
    def update(self, result, *args, **kwargs): self.panel.stable = result.panel_bbox is not None
    def remembered_associations(self, *args, **kwargs): return ()


class FakeWorker:
    def __init__(self, owner, events, scenario):
        self.owner, self.events, self.scenario = owner, events, scenario
        self.pending = None; self.kind = None; self.display_count = 0; self.panel_count = 0
        self.injected = set()
    def wait_ready(self, **kwargs): self.events.append(('worker_ready',))
    def submit(self, snapshot, kind='panel'):
        self.pending, self.kind = snapshot, kind
        self.events.append(('submit', kind, snapshot.generation, snapshot.frame_id))
    def receive(self):
        if self.pending is None: return None
        if self.scenario=='stale' and self.kind not in self.injected:
            self.injected.add(self.kind)
            old=replace(self.pending,generation=self.pending.generation-1)
            # Deliberately use a wrong result structure. The generation guard
            # must discard it before any stage-specific attribute access.
            wrong=[] if self.kind=='call' else NS(state='STABLE',pair=())
            self.events.append(('stale_received',self.kind,old.generation))
            return old,wrong
        snap, kind = self.pending, self.kind; self.pending = None
        if kind == 'call':
            missing=self.scenario in ('timeout','joint_limits')
            result = NS(state='NO_PAIR' if missing else 'STABLE',
                        pair=() if missing else (Ring(80.,40.,12.,1.),Ring(80.,80.,12.,1.)))
        elif kind == 'panel':
            self.panel_count+=1
            button = ButtonObservation(Ring(80.,60.,12.,1.),text='3',confidence=.99,
                                       text_bbox=(74,50,86,65),status='STABLE')
            if self.scenario=='latched_floor6':
                button.text='6' if self.panel_count==1 else None
                button.confidence=.99 if self.panel_count==1 else None
                button.status='STABLE' if self.panel_count==1 else 'NO_TEXT'
                if self.panel_count==1:self.events.append(('floor6_confirmed',snap.captured_at))
            result = NS(panel_bbox=(50,20,110,100),errors=[],buttons=[button],tracking={},captured_at=snap.captured_at)
        elif kind == 'display':
            self.display_count += 1
            value = '6' if self.scenario=='non_target' and self.display_count<=120 else '3'
            if self.scenario=='latched_floor6':value='6'
            if self.scenario=='history':
                result = NS(state='UNSTABLE',current_floor=None,observed_floor='3',last_confirmed_floor='3')
            else:
                result = NS(state='STABLE',current_floor=value,observed_floor=value,last_confirmed_floor='3')
        else: result = []
        self.events.append(('received',kind,snap.generation,snap.frame_id))
        return snap,result
    def close(self): self.events.append(('worker_close',))


class JourneyRuntimeTests(unittest.TestCase):
    def settings(self):
        options = dict(trajectory={'sample_interval_s':.1},
            ocr=dict(min_confidence=.80,device='cuda',preprocessing='bright_dark',gpu_index=0,
                     model_dir='models/261104_my_rapidocr_torch'),temporal={'required':3},
            startup_view=dict(joints_deg=[0,0,-10,-20,0,0],duration_s=2.,wait_for_arrival=False),
            gripper=dict(closed_angle_deg=0.,closed_tolerance_rad=.01),
            tool={'tip_offset_in_end_m':[0,0,0]},
            elevator=dict(target_offset_base_m=[0,0,0],standoff_m=.1,press_travel_m=0.,
                          panel_type='physical',move_duration_s=.1,contact_duration_s=.1,retreat_duration_s=.1,dwell_s=0.,floor_wait_s=5.))
        settings = dict(search_timeout_s=10.,settle_s=.2,max_settle_s=1.,arrival_timeout_s=120.,
            max_observation_age_s=.8,display_min_confidence=.8,display_view_deg=[0,0,-25,-20,0,0],
            call_target_offset_base_m=[0,0,0])
        args=NS(execute=True,press_stage='contact',device='cpu',place_xyz_m=[.25,-.38,.35],
                place_rpy_deg=None,place_duration_s=3.,release_wait_s=.5)
        cfg={'camera':{'type':'realsense_d405'},'calibration':{},
             'robot':{'ready_pose':dict(x=.3,y=0.,z=.3,roll=0.,pitch=.7,yaw=0.,duration=3.)}}
        return cfg,options,settings,args

    def flow(self, scenario='complete', fail_cleanup=False, execute=True, button_orientation=None, initial_poses=None):
        clock, events = Clock(), []
        cfg, options, settings, args = self.settings()
        if initial_poses is not None:options['initial_poses']=initial_poses
        if scenario=='joint_limits':options['initial_poses']={'call_view_deg':[0,-170,-160,-20,0,0]}
        if scenario=='wait_display':settings['arrival_timeout_s']=10.
        if scenario=='latched_floor6':options['elevator']['floor_cache_until_update']=True
        if scenario in ('button_confirmation_timeout','standoff_send_error'):
            options['elevator'].update(standoff_m=.20,refinement=dict(
                enabled=True,vertical_panel=True,max_correction_m=.03,reacquire_timeout_s=1.))
        args.execute=execute
        if button_orientation is not None:args.button_orientation=button_orientation
        with ExitStack() as stack:
            stack.enter_context(patch.object(j.app,'_tracker',side_effect=lambda options:FakeTracker()))
            owner=j.Journey(cfg,options,settings,args)
            camera=FakeCamera(clock,events); robot=FakeRobot(events,fail_cleanup); worker=FakeWorker(owner,events,scenario)
            stack.enter_context(patch('drivers.camera.make_camera',return_value=camera))
            stack.enter_context(patch('utils.camera_utils.load_hand_eye',return_value=(np.eye(4),'eye_in_hand')))
            stack.enter_context(patch.object(j.motion,'Robot',return_value=robot))
            stack.enter_context(patch.object(j,'JourneyWorker',return_value=worker))
            stack.enter_context(patch.object(j.time,'perf_counter',side_effect=clock))
            terminal_stream=None
            if scenario.startswith('terminal_input'):
                terminal_master,terminal_slave=pty.openpty()
                terminal_stream=os.fdopen(terminal_slave,'r',buffering=1)
                terminal_original=termios.tcgetattr(terminal_stream.fileno())
                stack.callback(os.close,terminal_master)
                stack.callback(terminal_stream.close)
                stack.enter_context(patch.object(j.sys,'stdin',terminal_stream))
                if scenario=='terminal_input_error':
                    move=robot.move
                    def fail_floor(target,duration,cancelled):
                        if owner.state.phase=='FLOOR':raise RuntimeError('synthetic floor command failure')
                        move(target,duration,cancelled)
                    robot.move=fail_floor
            else:
                stack.enter_context(patch.object(j.sys,'stdin',NS(isatty=lambda:False)))
            stack.enter_context(patch.object(j.cv2,'namedWindow'))
            stack.enter_context(patch.object(j.cv2,'imshow'))
            stack.enter_context(patch.object(j.cv2,'destroyAllWindows'))
            stack.enter_context(patch.object(j.cv2,'getWindowProperty',return_value=1))
            if scenario in ('button_confirmation_timeout','standoff_send_error'):
                def plane(snapshot,button,*args,**kwargs):
                    normal=np.array([-1.,0.,0.])
                    return j.geometry.ButtonSurface(snapshot,button,np.array([0.,0.,.30]),
                        np.array([.48,0.,.34]),normal,normal,.0005,110)
                stack.enter_context(patch.object(j.geometry,'measure_button_surface',side_effect=plane))
                stack.enter_context(patch.object(j.app,'reacquire_button',
                    side_effect=j.app.ButtonRefinementError('同键重新定位超时')))
                if scenario=='standoff_send_error':
                    def stop_sender(*args,**kwargs):
                        robot.stop()
                        raise j.motion.TrajectorySendError('synthetic standoff sender failure')
                    robot.move=stop_sender
            sent=set()
            moving=[False]
            interactive = [False]
            call_wait_started = [None]
            display_wait_started = [None]
            def draw(owner, current):
                owner.display=current.color
                events.append(('draw',owner.state.phase))
                if owner.state.phase=='GRASP':interactive[0]=True
            stack.enter_context(patch.object(j.Journey,'draw',draw))
            def adapter(actual_robot, cancelled):
                def ready(controller, ready_cfg):
                    if cancelled():raise j.motion.MotionCancelled('synthetic startup cancel')
                    if scenario=='ready_failure':raise RuntimeError('synthetic ready failure')
                    if scenario=='ik_failure_ready':
                        robot.stop();events.append(('planning_failure','GRASP_READY'))
                        raise j.motion.TrajectoryPlanningError(np.eye(4),3.)
                    events.append(('ready',dict(ready_cfg)))
                    from utils.transforms import pose6d_to_mat4
                    actual_robot.T=pose6d_to_mat4(ready_cfg['x'],ready_cfg['y'],ready_cfg['z'],
                                               ready_cfg['roll'],ready_cfg['pitch'],ready_cfg.get('yaw',0.))
                return NS(_move_ready=ready),object()
            stack.enter_context(patch.object(j.grasp,'original_motion_adapter',side_effect=adapter))
            def key(delay=1):
                if scenario=='quit': return ord('q')
                if moving[0]:return ord('n')
                if owner.skip_in_progress:return -1
                phase=owner.state.phase
                if phase=='GRASP' and interactive[0] and 'N' not in sent:
                    sent.add('N'); events.append(('key','N')); return ord('n')
                if phase=='WAIT_CALL' and 'CALL_C' not in sent:
                    if scenario in ('skip_call','skip_all','skip_stop_error') and 'SKIP_CALL' not in sent:
                        sent.add('SKIP_CALL');return ord('n')
                    if call_wait_started[0] is None:
                        call_wait_started[0]=clock.now
                        events.append(('call_wait_start',clock.now))
                    if scenario in ('wait_call','wait_call_q') and clock.now-call_wait_started[0]<35.:
                        return -1
                    if scenario=='wait_call_q': return ord('q')
                    sent.add('CALL_C'); events.append(('key','CALL_C',clock.now)); return ord('c')
                if phase=='CALL' and scenario=='skip_active_call' and 'SKIP_CALL' not in sent:
                    sent.add('SKIP_CALL');return ord('n')
                if phase=='CALL' and owner.ready(clock.now) and 'U' not in sent:
                    sent.add('U'); return ord('u')
                if phase=='BOARD' and 'C' not in sent:
                    sent.add('C'); return ord('c')
                if phase=='FLOOR' and scenario in ('skip_floor','skip_all') and 'SKIP_FLOOR' not in sent:
                    sent.add('SKIP_FLOOR');return ord('n')
                if phase=='FLOOR' and scenario=='skip_floor_selected' and '3' in sent and 'SKIP_FLOOR' not in sent:
                    sent.add('SKIP_FLOOR');return ord('n')
                if phase=='FLOOR' and scenario=='skip_floor_pending' and owner.request is not None:
                    events.append(('key','SKIP_PENDING_FLOOR'));return ord('n')
                if phase=='FLOOR' and owner.ready(clock.now):
                    if scenario=='latched_floor6':
                        saved=owner.floor_memory(clock.now).get('6')
                        if saved is None or clock.now-saved[0].captured_at<30.:return -1
                        if '6' not in sent:
                            sent.add('6');events.append(('key','FLOOR6',clock.now));return ord('6')
                        return -1
                    if terminal_stream is not None and 'TERMINAL' not in sent:
                        sent.add('TERMINAL')
                        value=(b'q' if scenario=='terminal_input_quit' else
                               b'n' if scenario=='terminal_input_skip_floor' else b'3')
                        os.write(terminal_master,value)
                        events.append(('terminal_key',value.decode()))
                        return -1
                    if terminal_stream is not None:return -1
                    if '3' not in sent: sent.add('3'); return ord('3')
                if phase=='WAIT_DISPLAY' and ('draw','WAIT_DISPLAY') in events and 'DISPLAY_C' not in sent:
                    if scenario in ('skip_all','skip_display','terminal_input_skip_floor') and 'SKIP_DISPLAY' not in sent:
                        sent.add('SKIP_DISPLAY');return ord('n')
                    if scenario=='wait_display':
                        if display_wait_started[0] is None:
                            display_wait_started[0]=clock.now
                            events.append(('display_wait_start',clock.now))
                        if clock.now-display_wait_started[0]<35.:return -1
                    sent.add('DISPLAY_C');events.append(('key','DISPLAY_C',clock.now))
                    if terminal_stream is not None:
                        os.write(terminal_master,b'c')
                        events.append(('terminal_key','c'))
                        return -1
                    return ord('c')
                if phase=='DISPLAY':
                    if scenario=='skip_floor' and owner.floor is None and 'ARRIVAL_DIGIT' not in sent:
                        sent.add('ARRIVAL_DIGIT');return ord('3')
                return -1
            if scenario=='skip_stop_error':
                def failed_stop():raise RuntimeError('synthetic skip stop failure')
                robot.stop=failed_stop
            if scenario in ('motion_skip_call','motion_skip_floor'):
                original_move=robot.move
                def cancelled_move(target,duration,cancelled):
                    phase='CALL' if scenario=='motion_skip_call' else 'FLOOR'
                    if owner.state.phase==phase and 'MOTION_SKIP' not in sent:
                        sent.add('MOTION_SKIP');moving[0]=True
                        robot.T[0,3]+=.01  # Actual movement before cancellation, not a pre-command N.
                        events.append(('motion_skip',phase))
                        try:
                            if cancelled():
                                robot.stop();raise j.motion.MotionCancelled('synthetic N motion skip')
                        finally:moving[0]=False
                    original_move(target,duration,cancelled)
                robot.move=cancelled_move
            if scenario.startswith('ik_failure') and scenario!='ik_failure_ready':
                original_move=robot.move
                def planning_failure(target,duration,cancelled):
                    phase='FLOOR' if scenario=='ik_failure_floor' else 'CALL'
                    if owner.state.phase==phase:
                        events.append(('planning_failure',phase))
                        if scenario=='ik_failure_feedback':raise RuntimeError('synthetic feedback unavailable')
                        if scenario=='ik_failure_stop':raise RuntimeError('synthetic stop failure')
                        robot.stop()
                        raise j.motion.TrajectoryPlanningError(target,duration)
                    original_move(target,duration,cancelled)
                robot.move=planning_failure
            stack.enter_context(patch.object(j.cv2,'waitKey',side_effect=key))
            stack.enter_context(redirect_stdout(io.StringIO())); stack.enter_context(redirect_stderr(io.StringIO()))
            try: code=owner.run()
            except Exception as exc:
                if fail_cleanup or scenario.startswith('ik_failure') or scenario in (
                        'ready_failure','terminal_input_error','skip_stop_error','button_confirmation_timeout',
                        'standoff_send_error'):
                    if terminal_stream is not None:
                        events.append(('terminal_restored',termios.tcgetattr(terminal_stream.fileno())==terminal_original))
                    return exc,events,owner,worker
                raise
            if terminal_stream is not None:
                events.append(('terminal_restored',termios.tcgetattr(terminal_stream.fileno())==terminal_original))
            return code,events,owner,worker

    def test_floor6_once_confirmed_then_unread_for_30_seconds_still_finishes_button_and_arrival(self):
        code,events,owner,worker=self.flow('latched_floor6')
        self.assertEqual(code,0);self.assertEqual(owner.floor,'6');self.assertEqual(owner.state.phase,'DONE')
        confirmed=next(e[1] for e in events if e[0]=='floor6_confirmed')
        chosen=next(e[2] for e in events if e[:2]==('key','FLOOR6'))
        self.assertGreaterEqual(chosen-confirmed,30.)
        self.assertGreater(worker.panel_count,100)
        self.assertEqual(len([e for e in events if e[0]=='move']),6)
        self.assertEqual([e for e in events if e[0]=='robot_finish'],[('robot_finish',True)])

    def test_four_independent_initial_poses_drive_the_real_journey_orchestration(self):
        initial=dict(grasp_ready=dict(x=.27,y=.01,z=.31,roll=0.,pitch=.6,yaw=.1,duration=2.5,
                                     joint4_stage_duration_s=1.25),
                     call_view_deg=[0,0,-20,-20,0,0],floor_view_deg=[0,0,-30,-20,0,0],
                     display_view_deg=[0,0,-45,-20,0,0])
        code,events,owner,_=self.flow(initial_poses=initial)
        self.assertEqual(code,0)
        self.assertFalse(any(e[0]=='ready' for e in events))
        self.assertEqual([e for e in events if e[0]=='staged_ready'],[('staged_ready',[.27,.01,.31],2.5,1.25)])
        self.assertLess(events.index(('staged_ready',[.27,.01,.31],2.5,1.25)),events.index(('key','N')))
        self.assertEqual([e[1] for e in events if e[0]=='head'],
                         [initial[key]+[0.] for key in ('call_view_deg','floor_view_deg','display_view_deg')])
        self.assertEqual(owner.state.phase,'DONE')
        self.assertEqual([e for e in events if e[0]=='robot_finish'],[('robot_finish',True)])

    def test_n_c_call_u_board_c_floor3_arrival3_reuses_one_session_and_homes(self):
        code,events,owner,worker=self.flow()
        self.assertEqual(code,0); self.assertEqual(owner.state.phase,'DONE'); self.assertEqual(owner.floor,'3')
        self.assertEqual([e for e in events if e[0]=='robot_finish'],[('robot_finish',True)])
        self.assertEqual(len([e for e in events if e[0]=='robot_start']),1)
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10,-10,-25])
        self.assertEqual(len([e for e in events if e[0]=='move']),6)
        self.assertIn(('camera_close',),events); self.assertIn(('worker_close',),events)
        for kind in ('call','panel','display'): self.assertTrue(any(e[:2]==('submit',kind) for e in events))

    def test_call_wait_over_three_search_windows_does_not_infer_or_retry_before_c(self):
        code,events,owner,worker=self.flow('wait_call')
        self.assertEqual(code,0)
        wait_at=next(i for i,e in enumerate(events) if e[0]=='call_wait_start')
        c_at=next(i for i,e in enumerate(events) if e[:2]==('key','CALL_C'))
        self.assertGreaterEqual(events[c_at][2]-events[wait_at][1],35.)
        self.assertFalse(any(e[0] in ('submit','head','move') for e in events[wait_at:c_at]))
        self.assertGreater(next(i for i,e in enumerate(events) if e[:2]==('submit','call')),c_at)
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10,-10,-25])

    def test_terminal_floor_digit_without_newline_completes_chain_and_restores_tty(self):
        code,events,owner,worker=self.flow('terminal_input')
        self.assertEqual(code,0);self.assertEqual(owner.floor,'3')
        self.assertIn(('terminal_key','3'),events)
        self.assertIn(('terminal_key','c'),events)
        self.assertIn(('terminal_restored',True),events)
        self.assertIn(('robot_finish',True),events)
        self.assertIsNone(owner.terminal_state)

    def test_terminal_q_during_floor_wait_homes_and_restores_tty(self):
        code,events,owner,worker=self.flow('terminal_input_quit')
        self.assertEqual(code,0);self.assertIsNone(owner.floor)
        self.assertIn(('terminal_key','q'),events)
        self.assertIn(('terminal_restored',True),events)
        self.assertIn(('robot_finish',True),events)

    def test_terminal_mode_restores_even_when_floor_motion_raises(self):
        error,events,owner,worker=self.flow('terminal_input_error')
        self.assertIsInstance(error,RuntimeError)
        self.assertIn('synthetic floor command failure',str(error))
        self.assertIn(('terminal_restored',True),events)
        self.assertIn(('robot_finish',False),events)
        self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)

    def test_skip_call_keeps_boarding_confirmation_then_completes_floor_and_display(self):
        for scenario in ('skip_call','skip_active_call'):
            with self.subTest(scenario=scenario):
                code,events,owner,worker=self.flow(scenario)
                self.assertEqual(code,0);self.assertEqual(owner.floor,'3')
                self.assertIn(('draw','BOARD'),events)
                self.assertEqual(len([e for e in events if e[0]=='move']),3)
                self.assertIn(('robot_finish',True),events)

    def test_skip_floor_does_only_call_contact_and_accepts_arrival_target_without_pressing(self):
        for scenario in ('skip_floor','skip_floor_selected','skip_floor_pending'):
            with self.subTest(scenario=scenario):
                code,events,owner,worker=self.flow(scenario)
                self.assertEqual(code,0);self.assertEqual(owner.floor,'3')
                self.assertEqual(len([e for e in events if e[0]=='move']),3)
                self.assertIn(('draw','WAIT_DISPLAY'),events)
                self.assertIn(('draw','DISPLAY'),events)
                self.assertIn(('robot_finish',True),events)
                if scenario=='skip_floor_pending':self.assertIn(('key','SKIP_PENDING_FLOOR'),events)

    def test_wait_display_has_no_inference_or_timeout_until_c(self):
        code,events,owner,worker=self.flow('wait_display')
        self.assertEqual(code,0)
        start=next(i for i,e in enumerate(events) if e[0]=='display_wait_start')
        confirmed=next(i for i,e in enumerate(events) if e[:2]==('key','DISPLAY_C'))
        self.assertGreaterEqual(events[confirmed][2]-events[start][1],35.)
        self.assertFalse(any(e[0] in ('submit','head','move') for e in events[start:confirmed]))
        self.assertGreater(next(i for i,e in enumerate(events) if e[:2]==('submit','display')),confirmed)
        self.assertIn(('robot_finish',True),events)

    def test_skip_last_display_stage_homes_without_running_lcd_inference(self):
        code,events,owner,worker=self.flow('skip_display')
        self.assertEqual(code,0);self.assertEqual(owner.state.phase,'DONE')
        self.assertFalse(any(e[:2]==('submit','display') for e in events))
        self.assertIn(('robot_finish',True),events)

    def test_skip_all_elevator_stages_performs_no_button_contact_and_homes(self):
        code,events,owner,worker=self.flow('skip_all')
        self.assertEqual(code,0);self.assertEqual(owner.state.phase,'DONE')
        self.assertFalse(any(e[0]=='move' for e in events))
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10,-10,-25])
        self.assertIn(('robot_finish',True),events)
        self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)
        self.assertEqual(owner.floor_cache,{});self.assertIsNone(owner.call_cache)

    def test_skips_in_preview_never_initialize_or_control_robot(self):
        code,events,owner,worker=self.flow('skip_all',execute=False)
        self.assertEqual(code,0);self.assertEqual(owner.state.phase,'DONE')
        self.assertFalse(any(e[0] in ('robot_start','head','move','stop','robot_finish') for e in events))

    def test_n_during_call_or_floor_motion_stops_and_advances_to_next_stage(self):
        for scenario,phase in (('motion_skip_call','CALL'),('motion_skip_floor','FLOOR')):
            with self.subTest(scenario=scenario):
                code,events,owner,worker=self.flow(scenario)
                self.assertEqual(code,0);self.assertEqual(owner.floor,'3')
                self.assertIn(('motion_skip',phase),events)
                self.assertEqual(len([e for e in events if e[0]=='move']),4)
                self.assertIn(('stop',),events);self.assertIn(('robot_finish',True),events)

    def test_terminal_n_skips_floor_then_last_stage_restores_tty_and_homes(self):
        code,events,owner,worker=self.flow('terminal_input_skip_floor')
        self.assertEqual(code,0);self.assertIsNone(owner.floor)
        self.assertIn(('terminal_key','n'),events);self.assertIn(('terminal_restored',True),events)
        self.assertEqual(len([e for e in events if e[0]=='move']),3)
        self.assertIn(('robot_finish',True),events)

    def test_failed_skip_stop_aborts_without_new_posture_or_home(self):
        error,events,owner,worker=self.flow('skip_stop_error')
        self.assertIsInstance(error,RuntimeError)
        self.assertIn('synthetic skip stop failure',str(error))
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10])
        self.assertIn(('robot_finish',False),events)
        self.assertFalse(any(e[0]=='move' for e in events))

    def test_planning_rejection_in_call_floor_or_ready_requests_home_and_keeps_control(self):
        for scenario,phase in (('ik_failure_call','CALL'),('ik_failure_floor','FLOOR'),('ik_failure_ready','GRASP_READY')):
            with self.subTest(scenario=scenario):
                error,events,owner,worker=self.flow(scenario)
                self.assertIsInstance(error,j.motion.TrajectoryPlanningError)
                self.assertFalse(error.trajectory_sent);self.assertTrue(owner.home)
                self.assertIn(('planning_failure',phase),events)
                self.assertGreater(events.index(('robot_finish',True)),events.index(('planning_failure',phase)))
                self.assertIsNone(owner.request);self.assertIsNone(owner.skip_retreat)
                self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)

    def test_stop_or_feedback_fault_does_not_request_automatic_home(self):
        for scenario in ('ik_failure_stop','ik_failure_feedback'):
            with self.subTest(scenario=scenario):
                error,events,owner,worker=self.flow(scenario)
                self.assertIsInstance(error,RuntimeError)
                self.assertNotIsInstance(error,j.motion.TrajectoryPlanningError)
                self.assertIn(('robot_finish',False),events);self.assertFalse(owner.home)

    def test_post_standoff_confirmation_timeout_homes_without_attempting_contact(self):
        error,events,owner,_=self.flow('button_confirmation_timeout')
        self.assertIsInstance(error,j.app.ButtonRefinementError)
        self.assertTrue(owner.home)
        self.assertEqual(len([event for event in events if event[0]=='move']),1)
        self.assertIn(('robot_finish',True),events)
        self.assertLess(events.index(('worker_close',)),events.index(('robot_finish',True)))
        self.assertIsNone(owner.request)

    def test_standoff_sender_error_stops_then_requests_home_without_contact(self):
        error,events,owner,_=self.flow('standoff_send_error')
        self.assertIsInstance(error,j.motion.TrajectorySendError)
        self.assertTrue(owner.home)
        self.assertLess(events.index(('stop',)),events.index(('robot_finish',True)))
        self.assertFalse(any(event[0]=='move' for event in events))

    def test_failed_home_cleanup_preserves_planning_error_and_closes_other_resources(self):
        error,events,owner,worker=self.flow('ik_failure_call',fail_cleanup=True)
        self.assertIsInstance(error,j.motion.TrajectoryPlanningError)
        self.assertIn(('robot_finish',True),events)
        self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)

    def test_free_orientation_uses_position_moves_for_call_and_floor_and_homes(self):
        with patch.object(j.geometry,'point_from_button',wraps=j.geometry.point_from_button) as solve, \
                patch.object(j.app,'reacquire_button',side_effect=AssertionError('second localization')):
            code,events,owner,worker=self.flow(button_orientation='free')
        self.assertEqual(code,0);self.assertEqual(owner.floor,'3')
        self.assertEqual(solve.call_count,2)
        self.assertEqual(len([e for e in events if e[0]=='position_move']),6)
        self.assertIn(('robot_finish',True),events)

    def test_free_position_skip_retreat_and_planning_error_keep_previous_recovery(self):
        code,events,owner,worker=self.flow('motion_skip_floor',button_orientation='free')
        self.assertEqual(code,0)
        self.assertEqual(len([e for e in events if e[0]=='move']),4)
        self.assertGreaterEqual(len([e for e in events if e[0]=='position_move']),5)
        error,events,owner,worker=self.flow('ik_failure_call',button_orientation='free')
        self.assertIsInstance(error,j.motion.TrajectoryPlanningError)
        self.assertIn(('robot_finish',True),events)

    def test_q_during_call_wait_homes_without_starting_call_recognition(self):
        code,events,owner,worker=self.flow('wait_call_q')
        self.assertEqual(code,0)
        self.assertEqual(owner.state.phase,'WAIT_CALL')
        self.assertIsNone(owner.state.deadline)
        self.assertFalse(any(e[:2]==('submit','call') for e in events))
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10])
        self.assertIn(('robot_finish',True),events)

    def test_original_ready_pose_runs_once_after_start_before_input_and_inference(self):
        code,events,owner,worker=self.flow()
        self.assertEqual(code,0)
        ready=[e for e in events if e[0]=='ready']
        self.assertEqual(ready,[('ready',owner.cfg['robot']['ready_pose'])])
        ready_at=events.index(ready[0])
        self.assertGreater(ready_at,next(i for i,e in enumerate(events) if e[0]=='robot_start'))
        self.assertLess(ready_at,events.index(('draw','GRASP')))
        self.assertLess(ready_at,events.index(('key','N')))
        self.assertLess(ready_at,next(i for i,e in enumerate(events) if e[0]=='submit'))

    def test_ready_failure_stops_before_grasp_or_head_and_still_releases_resources(self):
        error,events,owner,worker=self.flow('ready_failure')
        self.assertIsInstance(error,RuntimeError)
        self.assertIn('synthetic ready failure',str(error))
        self.assertFalse(any(e[0] in ('head','move','submit') for e in events))
        self.assertIn(('robot_finish',False),events)
        self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)

    def test_preview_never_calls_ready_or_any_robot_action(self):
        code,events,owner,worker=self.flow(execute=False)
        self.assertEqual(code,0)
        self.assertIsNone(owner.robot)
        self.assertFalse(any(e[0] in ('ready','robot_start','head','move','switch_gripper','robot_finish') for e in events))
        self.assertIn(('camera_close',),events);self.assertIn(('worker_close',),events)

    def test_three_search_windows_timeout_home_and_no_button_action(self):
        code,events,owner,worker=self.flow('timeout')
        self.assertEqual(code,2); self.assertEqual(owner.state.phase,'TIMEOUT')
        heads=[e[1][:6] for e in events if e[0]=='head']
        np.testing.assert_allclose(heads,[[0,0,-10,-20,0,0],[0,-15,-35,-20,0,0],[0,-30,-60,-20,0,0]])
        self.assertFalse(any(e[0]=='move' for e in events))
        self.assertIn(('robot_finish',True),events)

    def test_both_axes_at_the_limit_run_one_shortened_search_then_home(self):
        code,events,owner,worker=self.flow('joint_limits')
        self.assertEqual(code,2);self.assertEqual(owner.state.phase,'TIMEOUT')
        heads=[e[1][:6] for e in events if e[0]=='head']
        limit=np.degrees(-3.14)
        np.testing.assert_allclose(heads,[[0,-170,-160,-20,0,0],[0,limit,limit,-20,0,0]])
        self.assertFalse(any(e[0]=='move' for e in events))
        self.assertIn(('robot_finish',True),events)

    def test_late_results_from_old_generation_cannot_cross_stage_boundaries(self):
        code,events,owner,worker=self.flow('stale')
        self.assertEqual(code,0); self.assertEqual(owner.floor,'3')
        self.assertEqual({e[1] for e in events if e[0]=='stale_received'},{'call','panel','display'})
        self.assertEqual(len([e for e in events if e[0]=='move']),6)
        self.assertIn(('robot_finish',True),events)

    def test_q_requests_home_and_exits_normally(self):
        code,events,owner,worker=self.flow('quit')
        self.assertEqual(code,0); self.assertIn(('robot_finish',True),events)
        self.assertFalse(any(e[0] in ('head','move') for e in events))
        self.assertIn(('camera_close',),events); self.assertIn(('worker_close',),events)

    def test_non_target_current_floor_waits_more_than_ten_seconds_without_head_retry(self):
        code,events,owner,worker=self.flow('non_target')
        self.assertEqual(code,0); self.assertGreater(worker.display_count,120)
        self.assertEqual([e[1][2] for e in events if e[0]=='head'],[-10,-10,-25])

    def test_observed_or_historical_target_does_not_complete_arrival(self):
        code,events,owner,worker=self.flow('history')
        self.assertEqual(code,2); self.assertEqual(owner.state.phase,'TIMEOUT')
        heads=[e[1][1:3] for e in events if e[0]=='head']
        np.testing.assert_allclose(heads,[[0,-10],[0,-10],[0,-25],[-15,-50],[-30,-75]])
        self.assertIn(('robot_finish',True),events)

    def test_cleanup_failure_propagates_and_other_resources_still_close(self):
        error,events,owner,worker=self.flow(fail_cleanup=True)
        self.assertIsInstance(error,RuntimeError); self.assertIn('synthetic home/hold failure',str(error))
        self.assertIn(('robot_finish',True),events)
        self.assertIn(('camera_close',),events); self.assertIn(('worker_close',),events)


if __name__=='__main__': unittest.main()
