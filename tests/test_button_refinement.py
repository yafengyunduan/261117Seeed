"""Offline RGB-D and robot feedback fixtures for the shared button action."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from elevator import app, geometry, journey_261113, motion, perception
from utils.transforms import pose6d_to_mat4


ROOT = Path(__file__).resolve().parents[1]
K = np.array([[315.,0.,160.],[0.,315.,120.],[0.,0.,1.]])
HAND_EYE = np.array([[0.,0.,1.,-.045],[-1.,0.,0.,0.],[0.,-1.,0.,.025],[0.,0.,0.,1.]])
START = np.eye(4)
START[:3,3] = [.25,0.,.30]
BUTTON = np.array([.50,0.,.32])
NORMAL = np.array([-1.,.30,0.])
NORMAL /= np.linalg.norm(NORMAL)
Y_GRID,X_GRID = np.indices((240,320))
RAYS = np.stack(((X_GRID-K[0,2])/K[0,0],(Y_GRID-K[1,2])/K[1,1],np.ones_like(X_GRID)),axis=-1)


def make_frame(pose=START, point=BUTTON, *, frame=1, timestamp=20., generation=0, label='3', protrusion=0., normal=NORMAL):
    camera = pose@HAND_EYE
    ray_normal = camera[:3,:3].T @ normal
    plane_offset = normal @ (point-camera[:3,3])
    depths = plane_offset / (RAYS @ ray_normal)
    selected = point+protrusion*normal
    def rim(center):
        xyz = camera[:3,:3].T @ (center-camera[:3,3])
        pixel = (K@xyz)[:2]/xyz[2]
        return perception.Ring(float(pixel[0]),float(pixel[1]),float(K[0,0]*.013/xyz[2]),1.)
    ring = rim(selected)
    if protrusion:
        mask = (X_GRID-ring.x)**2+(Y_GRID-ring.y)**2 <= ring.radius**2
        depths[mask] = (plane_offset+protrusion)/(RAYS[mask]@ray_normal)
    depth = np.round(depths*1000.).astype(np.uint16)
    color = np.full((240,320,3),50,np.uint8)
    cv2.circle(color,ring.center,round(ring.radius),(225,225,225),2)
    cv2.putText(color,'3' if label in ('U','D') else label,(round(ring.x-ring.radius*.4),round(ring.y+ring.radius*.4)),
                cv2.FONT_HERSHEY_SIMPLEX,ring.radius*.055,(230,230,230),2)
    if label in ('U','D'):
        other = rim(point+np.array([0.,0.,-.035 if label=='U' else .035]))
        pair = (ring,other) if label=='U' else (other,ring)
        result = SimpleNamespace(state='STABLE',pair=pair)
        button = SimpleNamespace(ring=ring)
    else:
        button = perception.ButtonObservation(ring,label,.99,status='STABLE')
        result = perception.FrameResult(timestamp,color.shape[:2],(0,0,320,240),'YOLOE',[button])
    snapshot = geometry.Snapshot(frame,timestamp,color,depth,K,np.zeros(5),pose.copy(),generation)
    return snapshot, result, button


class FeedbackRobot:
    def __init__(self, error_on_move=None):
        self.actual = START.copy()
        self.moves = []
        self.error_on_move = error_on_move
        self.stop = Mock()
        self.start = Mock()
        self.close = Mock()
        self.finish_task = Mock()
        self.head_up = Mock()
        self.arrival_policies = []
        self.position_moves = []

    def pose(self):
        return self.actual.copy()

    def pose_with_feedback(self):
        return self.pose(),motion.ArmFeedback(np.zeros(6),np.zeros(6),np.zeros(6))

    def move(self, target, duration, cancelled, *, settle=None):
        if cancelled():
            self.stop()
            raise motion.MotionCancelled('test cancelled')
        self.moves.append(target.copy())
        self.arrival_policies.append(settle)
        self.actual = target.copy()
        if len(self.moves)==self.error_on_move:
            self.actual[:3,3] += .009*NORMAL

    def move_position(self, tip, duration, cancelled, *, tip_offset, settle=None):
        self.position_moves.append(tip.copy())
        orientation = self.actual.copy()
        target = geometry.pose_for_tip(orientation,tip,tip_offset)
        self.move(target,duration,cancelled,settle=settle)
        return target


class FakeWorker:
    def __init__(self, results):
        self.results = results
        self.pending = None
        self.submissions = []
        self.focus_hints = []

    def wait_ready(self,**kwargs):
        pass

    def submit(self, snapshot, kind='panel', *, focus_px=None):
        self.pending = (snapshot,self.results[snapshot.frame_id])
        self.submissions.append((kind,snapshot.generation,snapshot.frame_id))
        self.focus_hints.append(focus_px)

    def receive(self):
        value,self.pending = self.pending,None
        return value

    def close(self):
        pass


class ButtonRefinementTests(unittest.TestCase):
    def setUp(self):
        self.options = app.load_workflow(ROOT/'config/261105_my_elevator.yaml')
        self.settings = self.options['elevator']
        self.settings.update(dwell_s=0.,target_offset_base_m=[.004,-.003,.002],button_orientation='level')
        self.settings['refinement'].update(enabled=True,standoff_face_panel=True,reacquire_timeout_s=1.6)
        self.quiet = redirect_stdout(io.StringIO())
        self.output = self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__,None,None,None)

    def measured(self, pose=START, point=BUTTON, **kwargs):
        snapshot,_,button = make_frame(pose,point,**kwargs)
        return geometry.measure_button_surface(snapshot,button,HAND_EYE,{})

    def test_tilted_plane_uses_outer_depth_but_keeps_protruding_button_face(self):
        normal=np.array([-1.,.30,.15]);normal/=np.linalg.norm(normal)
        snapshot,_,button=make_frame(protrusion=.006,normal=normal)
        snapshot.depth_mm[::3,::2] = 0
        snapshot.depth_mm[1::7,::2] += 21
        measured=geometry.measure_button_surface(snapshot,button,HAND_EYE,{})
        np.testing.assert_allclose(measured.outward_base,normal,atol=.012)
        np.testing.assert_allclose(measured.point_base,BUTTON+.006*normal,atol=.001)
        self.assertLess(measured.plane_rms_m,.001)

    def test_vertical_panel_standoff_is_horizontal_in_base_even_if_camera_pitches(self):
        tilted=pose6d_to_mat4(*START[:3,3],0.,-.28,0.)
        snapshot,_,button=make_frame(tilted)
        measured=geometry.measure_button_surface(snapshot,button,HAND_EYE,{},vertical_panel=True)
        np.testing.assert_allclose(measured.point_base,BUTTON,atol=.001)
        np.testing.assert_allclose(measured.outward_base,NORMAL,atol=.012)
        self.assertEqual(measured.outward_base[2],0.)
        offset=np.array([.012,0.,-.008])
        poses=app.refined_button_poses(measured.point_base,measured.outward_base,tilted,self.settings,offset)
        tip=lambda name:geometry.tip_position(poses[name],offset)
        np.testing.assert_allclose(tip('approach')-tip('surface'),self.settings['standoff_m']*NORMAL,atol=.002)
        self.assertAlmostEqual(tip('approach')[2],tip('surface')[2])
        np.testing.assert_allclose(poses['approach'][:3,0],-measured.outward_base,atol=1e-8)
        np.testing.assert_allclose(poses['approach'][:3,2],[0,0,1],atol=1e-8)

    def test_floor_priority_projection_uses_current_view_and_can_fall_back(self):
        moved=pose6d_to_mat4(.30,-.04,.32,0.,0.,-.12)
        snap,_,button=make_frame(moved)
        np.testing.assert_allclose(geometry.project_button_hint(snap,BUTTON,HAND_EYE,{}),
                                   [button.ring.x,button.ring.y],atol=.001)
        self.assertIsNone(geometry.project_button_hint(snap,[-.5,0.,.32],HAND_EYE,{}))

    def test_vertical_prior_does_not_hide_a_large_fitted_elevation(self):
        normal=np.array([-1.,.1,.6]);normal/=np.linalg.norm(normal)
        snapshot,_,button=make_frame(normal=normal)
        with self.assertRaisesRegex(ValueError,'偏离 base 水平面'):
            geometry.measure_button_surface(snapshot,button,HAND_EYE,{},vertical_panel=True)
        measured=geometry.measure_button_surface(snapshot,button,HAND_EYE,{},vertical_panel=False)
        np.testing.assert_allclose(measured.outward_base,normal,atol=.012)

    def test_no_valid_panel_samples_is_not_a_horizontal_fallback(self):
        snapshot,_,button = make_frame()
        pixel_x,pixel_y = button.ring.center
        bad=np.zeros_like(snapshot.depth_mm)
        bad[pixel_y-3:pixel_y+4,pixel_x-3:pixel_x+4]=snapshot.depth_mm[pixel_y-3:pixel_y+4,pixel_x-3:pixel_x+4]
        with self.assertRaisesRegex(ValueError,'周边有效深度不足'):
            geometry.measure_button_surface(replace(snapshot,depth_mm=bad),button,HAND_EYE,{})

    def test_fresh_digit_and_new_pixel_cache_can_support_three_frames_without_two_votes(self):
        _,result,button=make_frame(frame=9,timestamp=21.,generation=1)
        button.status='UNSTABLE'
        self.assertIs(app._match_reobserved_button(result,'3','panel',.7),button)
        button.text=button.confidence=None
        button.status='NO_TEXT'
        evidence=dict(source='PIXEL_MATCHED_CACHE',pixel_match=True)
        result.tracking.update(workflow_evidence=True,region_evidence={'0':evidence},
                               display_label_cache={'0':{'text':'3','confidence':.96}})
        self.assertIs(app._match_reobserved_button(result,'3','panel',.7),button)
        evidence['pixel_match']=False
        with self.assertRaisesRegex(ValueError,'未读到楼层3'):
            app._match_reobserved_button(result,'3','panel',.7)
        evidence.update(pixel_match=True,source='CURRENT_OCR')
        button.text,button.confidence='8',.99
        with self.assertRaisesRegex(ValueError,'未读到楼层3'):
            app._match_reobserved_button(result,'3','panel',.7)
        button.text='3'
        result.buttons.append(replace(button))
        result.tracking['region_evidence']['1']=evidence
        with self.assertRaisesRegex(ValueError,'多个圆环'):
            app._match_reobserved_button(result,'3','panel',.7)

    def test_call_accepts_only_the_present_pair_and_orders_up_down(self):
        _,result,_=make_frame(frame=2,timestamp=21.,generation=1,label='U')
        result.state='UNSTABLE'
        self.assertIs(app._match_reobserved_button(result,'U','call',.7).ring,result.pair[0])
        self.assertIs(app._match_reobserved_button(result,'D','call',.7).ring,result.pair[1])
        result.state='AMBIGUOUS'
        with self.assertRaisesRegex(ValueError,'重新确认外呼'):
            app._match_reobserved_button(result,'U','call',.7)

    def test_pre_motion_result_cannot_be_used_even_if_refresh_returns_it(self):
        robot=FeedbackRobot()
        old=self.measured()
        with self.assertRaisesRegex(app.ButtonRefinementError,'运动之前'):
            app.execute_refined_button(robot,old,self.settings,[0,0,0],'contact',lambda *args:old)
        self.assertEqual(len(robot.moves),1)

    def journey_action(self, label, *, unread=False, missed_sample=False, approach=False, error_on_move=None):
        cfg={'robot':{},'camera':{'type':'synthetic'}}
        journey_settings=journey_261113.load_settings(ROOT/'config/261113_my_journey.yaml',self.options)
        args=SimpleNamespace(press_stage='approach' if approach else 'contact',button_orientation='free')
        flow=journey_261113.Journey(cfg,self.options,journey_settings,args)
        phase='CALL' if label in ('U','D') else 'FLOOR'
        flow.state.enter(phase,20.)
        flow.robot=robot=FeedbackRobot(error_on_move)
        flow.hand_eye=HAND_EYE
        flow.head=Mock()
        flow.restore_terminal=Mock()
        flow.request=dict(value=label,at=20.)
        initial,result,button=make_frame(frame=1,timestamp=20.05,label=label)
        if phase=='FLOOR':
            initial=replace(initial,captured_at=10.)
            flow.floor_cache[label]=(initial,result,button)
        flow.last_snapshot,flow.last=initial,result
        results={}
        flow.worker=worker=FakeWorker(results)
        worker.pending=(initial,object())
        clock=[20.10]
        frames=[]
        def capture(*, with_arm_feedback=False):
            clock[0]+=.10
            observed=BUTTON+np.array([0.,.006+.001*len(worker.submissions),-.002])
            snapshot,detected,button=make_frame(robot.pose(),observed,frame=len(frames)+3,timestamp=clock[0],
                                              generation=flow.generation,label=label)
            if with_arm_feedback:
                snapshot=replace(snapshot,arm_feedback=robot.pose_with_feedback()[1])
            # Keep valid depth only at the selected rim. A second plane fit
            # would time out here even though the new button point is usable.
            u,v=button.ring.center
            depth=np.zeros_like(snapshot.depth_mm)
            depth[v-3:v+4,u-3:u+4]=snapshot.depth_mm[v-3:v+4,u-3:u+4]
            if missed_sample and len(worker.submissions)==1:depth.fill(0)
            snapshot=replace(snapshot,depth_mm=depth)
            if unread:
                detected=SimpleNamespace(state='NO_PAIR',pair=()) if phase=='CALL' else perception.FrameResult(clock[0],(240,320))
            frames.append(snapshot)
            results[snapshot.frame_id]=detected
            return snapshot
        flow.snapshot=capture
        with patch.object(app.time,'perf_counter',side_effect=lambda:clock[0]):
            flow.cancelled=lambda:False
            current=replace(initial,frame_id=2,captured_at=clock[0])
            if unread or error_on_move is not None:
                with self.assertRaises(app.ButtonRefinementError):flow.press(current)
            else:self.assertTrue(flow.press(current))
        return flow,robot,worker

    def test_up_down_and_cached_floor_press_after_exactly_three_fresh_frames_at_standoff(self):
        for label in ('U','D','3'):
            with self.subTest(label=label):
                flow,robot,worker=self.journey_action(label)
                self.assertEqual(len(robot.moves),3)
                self.assertEqual(robot.arrival_policies,[
                    motion.SettledArrival('approach',.15,stationary_vlim_rad_s=.05,lock_after_settle=True),None,
                    motion.SettledArrival('retreat',.15,True)])
                self.assertEqual(len(worker.submissions),3)
                self.assertEqual({generation for _,generation,_ in worker.submissions},{1})
                correction=np.asarray(self.settings['target_offset_base_m'])
                np.testing.assert_allclose(robot.moves[0][:3,3],
                                           BUTTON+correction+self.settings['standoff_m']*NORMAL,atol=.002)
                np.testing.assert_allclose(robot.moves[0][:3,0],-NORMAL,atol=.012)
                np.testing.assert_allclose(robot.moves[0][:3,2],[0,0,1],atol=1e-8)
                self.assertEqual(len(robot.position_moves),2)
                self.assertTrue(all((hint is not None)==(label=='3') for hint in worker.focus_hints))
                np.testing.assert_allclose(robot.moves[1][:3,3],BUTTON+correction+[0.,.008,-.002],atol=.001)
                np.testing.assert_allclose(robot.moves[2],robot.moves[0],atol=.002)
                self.assertEqual(flow.state.phase,'BOARD' if label in ('U','D') else 'WAIT_DISPLAY')
        self.assertRegex(self.output.getvalue(),r'stable=3/3 .*rim_shift_px=[1-9][0-9]*\.[0-9] '
                                                r'window_end_span_mm=0\.0 window_end_turn_deg=0\.00')
        self.assertIn('"largest_joint": null',self.output.getvalue())

    def test_persistent_pose_sway_blocks_new_ocr_until_relocalization_deadline(self):
        snap,_,_=make_frame(generation=1)
        worker=FakeWorker({})
        clock=[20.]
        frame=[0]
        def capture():
            clock[0]+=.1;frame[0]+=1
            pose=START.copy();pose[1,3]+=.004*(frame[0]%2)
            return replace(snap,frame_id=frame[0],captured_at=clock[0],T_base_end=pose)
        with patch.object(app.time,'perf_counter',side_effect=lambda:clock[0]), \
                self.assertRaisesRegex(app.ButtonRefinementError,'1.6s.*末端还未停稳.*4.0 mm'):
            app.reacquire_button(capture,worker,'panel','3',self.options,HAND_EYE,{},lambda:False,generation=1)
        self.assertEqual(worker.submissions,[])
        self.assertLessEqual(clock[0]-20.,1.7+1e-9)

    def test_relocalization_detects_reversal_even_close_to_the_first_pose(self):
        snap,result,_=make_frame(generation=1)
        worker=FakeWorker({})
        clock=[20.]
        frame=[0]
        def capture():
            clock[0]+=.1;frame[0]+=1
            pose=START.copy()
            q=np.zeros(6)
            if frame[0]>1:
                direction=(-1)**frame[0]
                pose[2,3]+=.0015*direction
                q[3]=np.radians(-.4*direction)
            feedback=motion.ArmFeedback(q,np.zeros(6),np.zeros(6),-1.4+.3*np.sign(q[3]))
            worker.results[frame[0]]=result
            return replace(snap,frame_id=frame[0],captured_at=clock[0],T_base_end=pose,arm_feedback=feedback)
        with patch.object(app.time,'perf_counter',side_effect=lambda:clock[0]), \
                self.assertRaisesRegex(app.ButtonRefinementError,'末端还未停稳.*3.0 mm'):
            app.reacquire_button(capture,worker,'panel','3',self.options,HAND_EYE,{},lambda:False,generation=1)
        self.assertEqual(worker.submissions,[])
        prefix='[StandoffFeedback] '
        log=next(line for line in self.output.getvalue().splitlines() if line.startswith(prefix))
        row=json.loads(log[len(prefix):])
        self.assertEqual(row['status'],'TIMEOUT')
        self.assertEqual(row['largest_joint'],'J4')
        self.assertEqual(row['joint_range_deg'],[0.,0.,0.,.8,0.,0.])
        self.assertEqual(row['hold_target_range_deg'],[0.]*6)
        self.assertEqual(row['fk_axis_range_mm'],[0.,0.,3.])
        self.assertEqual(row['joint_delta_low_to_high_z_deg'][3],-.8)
        self.assertEqual(row['joint4_error_range_deg'],[-.4,.4])
        self.assertEqual(row['joint4_torque_range_nm'],[-1.7,-1.1])

    def test_missing_arm_feedback_times_out_without_using_zero_or_old_pose(self):
        clock=[20.]
        worker=FakeWorker({})
        def capture():
            clock[0]+=.1
            raise motion.ArmFeedbackError('joint4')
        with patch.object(app.time,'perf_counter',side_effect=lambda:clock[0]), \
                self.assertRaisesRegex(app.ButtonRefinementError,'等待有效的六轴反馈.*joint4'):
            app.reacquire_button(capture,worker,'panel','3',self.options,HAND_EYE,{},lambda:False,generation=1)
        self.assertEqual(worker.submissions,[])
        self.assertRegex(self.output.getvalue(),r'"samples": 0, "invalid_samples": [1-9][0-9]*')

    def test_no_contact_if_three_fresh_frames_never_confirm_identity(self):
        for label in ('U','3'):
            with self.subTest(label=label):
                flow,robot,_=self.journey_action(label,unread=True)
                self.assertEqual(len(robot.moves),1)
                flow.head.assert_not_called()

    def test_invalid_depth_resets_the_three_frame_streak(self):
        _,robot,worker=self.journey_action('3',missed_sample=True)
        self.assertEqual(len(robot.moves),3)
        self.assertEqual(len(worker.submissions),5)
        self.assertEqual(self.output.getvalue().count('stable=1/3'),2)

    def test_cancellation_keeps_the_standoff_without_contact(self):
        robot=FeedbackRobot()
        def cancel():
            raise motion.MotionCancelled('用户停止重新定位')
        with self.assertRaisesRegex(motion.MotionCancelled,'重新定位'):
            app.execute_refined_button(robot,self.measured(),self.settings,[0,0,0],'contact',cancel)
        self.assertEqual(len(robot.moves),1)
        robot.stop.assert_called_once()

    def test_approach_stage_completes_visual_refinement_without_contact(self):
        self.settings['refinement']['standoff_brake_s']=.7
        _,robot,_=self.journey_action('3',approach=True)
        self.assertEqual(robot.arrival_policies,[motion.SettledArrival('approach',.15,
            brake_tail_s=.7,stationary_vlim_rad_s=.05,lock_after_settle=True)])
        self.assertEqual(len(robot.moves),1)
        self.assertNotIn('[ContactCheck]',self.output.getvalue())

    def test_concise_approach_reports_j4_and_result_without_per_frame_logs(self):
        with app.terminal_output(False):
            _,robot,worker=self.journey_action('3',approach=True)
        self.assertEqual(len(worker.submissions),3)
        self.assertEqual(len(robot.moves),1)
        log=self.output.getvalue()
        self.assertIn('[StandoffFeedback] STABLE 3，J4=0.000°',log)
        self.assertIn('完成精定位验证',log)
        self.assertNotIn('[Relocalize]',log)
        self.assertNotIn('camera_m=',log)

    def test_contact_feedback_out_of_tolerance_retracts_without_any_extra_press(self):
        self.settings['press_travel_m']=.002
        _,robot,_=self.journey_action('U',error_on_move=2)
        self.assertEqual(len(robot.moves),3)
        self.assertIn('GEOMETRY_OUT_OF_TOLERANCE',self.output.getvalue())
        self.assertNotIn('target=press',self.output.getvalue())

    def test_contact_dwell_is_after_final_press_and_before_retreat(self):
        self.settings.update(dwell_s=.4,press_travel_m=.003)
        robot=FeedbackRobot()
        clock=[0.]
        send_at=[]
        def sleep(seconds):
            clock[0]+=seconds
        move=robot.move
        def send(*args,**kwargs):
            send_at.append(clock[0])
            return move(*args,**kwargs)
        def refresh():
            return self.measured(robot.pose(),frame=2,timestamp=21.,generation=1)
        with patch.object(robot,'move',side_effect=send), \
                patch.object(app.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(app.time,'sleep',side_effect=sleep):
            poses=app.execute_refined_button(robot,self.measured(),self.settings,[0,0,0],'contact',refresh)
        np.testing.assert_allclose(send_at,[0.,0.,0.,.4])
        np.testing.assert_allclose(robot.moves[-2],poses['press'])
        np.testing.assert_allclose(robot.moves[-1],poses['retreat'])

    def standalone_action(self, unread=False):
        robot=FeedbackRobot();clock=[20.];result_by_frame={}
        worker=FakeWorker(result_by_frame)
        camera=SimpleNamespace(K=K,D=np.zeros(5),open=Mock(),close=Mock(),warm_up=Mock())
        sequence=[0]
        def get_frame():
            clock[0]+=.10
            sequence[0]+=1
            self.assertLess(sequence[0],65)
            snap,result,_=make_frame(robot.pose(),frame=sequence[0],timestamp=clock[0])
            if unread and robot.moves:result.buttons.clear()
            result_by_frame[sequence[0]]=result
            return snap.color,snap.depth_mm
        camera.get_frame=get_frame
        cfg={'robot':{},'camera':{'type':'synthetic'}}
        args=SimpleNamespace(execute=True,stage='contact',device='cpu',floor='3')
        with ExitStack() as stack:
            for target, value in (('drivers.camera.make_camera',camera),('elevator.app.PerceptionWorker',worker),
                                  ('elevator.motion.Robot',robot),('utils.camera_utils.load_hand_eye',(HAND_EYE,'eye_in_hand')),
                                  ('cv2.getWindowProperty',1),('cv2.namedWindow',None),('cv2.imshow',None),('cv2.destroyAllWindows',None)):
                stack.enter_context(patch(target,return_value=value))
            stack.enter_context(patch.object(app.sys.stdin,'isatty',return_value=False))
            stack.enter_context(patch.object(app.time,'perf_counter',side_effect=lambda:clock[0]))
            stack.enter_context(patch('cv2.waitKey',side_effect=lambda _: ord('q') if len(robot.moves)==3 else -1))
            if unread:
                stack.enter_context(redirect_stderr(io.StringIO()))
                with self.assertRaisesRegex(app.ButtonRefinementError,'未读到楼层3'):
                    app.run_elevator(cfg,self.options,args)
            else:self.assertEqual(app.run_elevator(cfg,self.options,args),0)
        camera.close.assert_called_once()
        robot.finish_task.assert_called_once_with(home=True)
        robot.close.assert_not_called()
        return robot,worker

    def test_standalone_floor_entry_runs_the_same_staged_executor(self):
        robot,worker=self.standalone_action()
        self.assertEqual(len(robot.moves),3)
        order=list(dict.fromkeys(generation for _,generation,_ in worker.submissions))
        self.assertEqual(order,[1,2,3])  # Initial search, one refinement, post-retreat preview.
        frame_ids=[frame for _,_,frame in worker.submissions]
        self.assertEqual(frame_ids,sorted(set(frame_ids)))
        self.assertIn('physical_contact=UNVERIFIED',self.output.getvalue())

    def test_standalone_relocalization_timeout_requests_home_instead_of_disabling(self):
        robot,_=self.standalone_action(unread=True)
        self.assertEqual(len(robot.moves),1)
        self.assertNotIn('[ContactCheck]',self.output.getvalue())


if __name__=='__main__':
    unittest.main()
