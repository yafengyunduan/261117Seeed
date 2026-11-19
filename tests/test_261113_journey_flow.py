"""Method-level journey regression with fake IO and no device initialization."""
import contextlib
import io
import os
from pathlib import Path
import pty
import termios
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from elevator import app, geometry, journey_261113 as journey, perception


ROOT = Path(__file__).resolve().parents[1]


class JourneyMethodTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'camera': {'type': 'realsense_d405'}, 'robot': {}}
        self.options = app.load_workflow(ROOT/'config/261105_my_elevator.yaml')
        self.options['elevator']['refinement']['enabled'] = False
        self.settings = journey.load_settings(ROOT/'config/261113_my_journey.yaml',self.options)
        self.args = NS(place_xyz_m=[.25, -.38, .35], place_rpy_deg=None,
                       place_duration_s=3., release_wait_s=.5, press_stage='contact')
        self.flow = journey.Journey(self.cfg, self.options, self.settings, self.args)
        self.pose = np.eye(4); self.pose[:3, 3] = [.26, 0., .2]
        self.flow.robot = NS(pose=Mock(side_effect=lambda: self.pose.copy()), head_up=Mock(),
                             switch_gripper_to_posvel=Mock(), move=Mock(), stop=Mock(), close=Mock())
        self.flow.camera = NS(warm_up=Mock())
        self.flow.worker = NS(receive=Mock(return_value=None))
        self.flow.settle = Mock()  # No camera frames, UI or sleeping in these tests.
        self.flow.hand_eye = np.eye(4)
        self.flow.place = np.eye(4)
        self.flow.tracker = NS(panel=NS(stable=True), reset=Mock(), update=Mock(),
                               remembered_associations=Mock(return_value=()))
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__(); self.addCleanup(self.stdout.__exit__, None, None, None)

    def snapshot(self, at=100.05, *, generation=None, depth=400, pose=None):
        return geometry.Snapshot(
            1, at, np.zeros((80, 100, 3), np.uint8), np.full((80, 100), depth, np.uint16),
            np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]), np.zeros(5),
            self.pose.copy() if pose is None else pose.copy(),
            self.flow.generation if generation is None else generation)

    def call_result(self, *, stable=True):
        return NS(state='STABLE' if stable else 'UNSTABLE',
                  pair=(perception.Ring(40., 30., 12., 1.), perception.Ring(60., 55., 12., 1.)))

    def panel_result(self, labels=('3',)):
        buttons = [perception.ButtonObservation(perception.Ring(40.+20*i, 40., 10., 1.),
                    label, .99, (38+20*i, 38, 42+20*i, 42), status='STABLE')
                   for i, label in enumerate(labels)]
        return perception.FrameResult(100.05, (80, 100), (5, 5, 95, 75), 'GEOMETRY', buttons)

    def evidence(self, phase='CALL', result=None, at=100.05):
        self.flow.state.enter(phase, 100.)
        self.flow.last_snapshot = self.snapshot(at)
        self.flow.last = self.call_result() if result is None else result

    def test_n_skips_grasp_and_place_then_switches_and_heads_once(self):
        self.flow.want_grasp = True
        with patch.object(journey.grasp, 'pick_and_place', side_effect=AssertionError('must skip')) as pick:
            self.flow.key('N', 100.)
        pick.assert_not_called()
        self.assertFalse(self.flow.want_grasp)
        self.assertEqual(self.flow.state.phase, 'WAIT_CALL')
        self.assertIsNone(self.flow.state.deadline)
        self.flow.robot.switch_gripper_to_posvel.assert_called_once()
        self.flow.robot.head_up.assert_called_once()
        target = self.flow.robot.head_up.call_args.args[0]
        np.testing.assert_allclose(target[:6], self.flow.views['CALL'])
        self.assertEqual(target[6], self.options['gripper']['closed_angle_deg'])
        self.assertEqual(self.flow.generation, 1)

    def test_successful_grasp_and_place_restore_head_then_wait_for_c(self):
        self.flow.want_grasp, self.flow.grasp_after = True, 100.
        self.flow.last_snapshot, self.flow.last = self.snapshot(100.1), ['synthetic grasp']
        best = NS(position=np.array([0., 0., .4]), tcp_rotation=np.eye(3))
        with patch('utils.ordinary_grasp.select_best_grasp', return_value=best), \
                patch('utils.camera_utils.compose_cam_to_base_transform', return_value=np.eye(4)), \
                patch('utils.transforms.transform_grasp_pose_to_base', return_value=((0.,)*6, (0.,)*6)), \
                patch.object(journey.grasp, 'pick_and_place', return_value=True) as pick:
            self.flow.pick(self.snapshot(100.2))
        pick.assert_called_once()
        self.flow.robot.head_up.assert_called_once()
        self.assertEqual(self.flow.state.phase, 'WAIT_CALL')
        self.assertIsNone(self.flow.state.deadline)
        self.assertIsNone(self.flow.last)
        self.assertIsNone(self.flow.last_snapshot)
        self.assertIsNone(self.flow.KINDS.get(self.flow.state.phase))

    def test_wait_call_requires_c_and_starts_full_search_after_fresh_settle(self):
        self.flow.begin_call()
        generation = self.flow.generation
        for key in ('G', 'U', 'D', '3', 'ENTER'):
            self.flow.key(key, 1000.)
            self.assertEqual(self.flow.state.phase, 'WAIT_CALL')
            self.assertIsNone(self.flow.state.tick(1000., ready=False))
        self.flow.last, self.flow.last_snapshot = object(), self.snapshot(1000.)
        self.flow.request = dict(value='U', at=1000.)
        def settled():
            self.assertEqual(self.flow.state.phase, 'WAIT_CALL')
            self.assertIsNone(self.flow.last)
            self.assertIsNone(self.flow.last_snapshot)
            self.assertIsNone(self.flow.request)
        self.flow.settle.side_effect = settled
        with patch.object(journey.time, 'perf_counter', return_value=1001.):
            self.flow.key('c', 1000.)
        self.assertEqual(self.flow.state.phase, 'CALL')
        self.assertEqual(self.flow.state.deadline, 1001.+self.settings['search_timeout_s']['call'])
        self.assertEqual(self.flow.generation, generation+1)
        self.flow.robot.head_up.assert_called_once()  # C does not repeat the move.

    def test_g_requests_fresh_grasp_and_failed_grasp_stays_in_grasp_phase(self):
        move_ready = Mock()
        with patch.object(journey.grasp, 'original_motion_adapter', return_value=(NS(_move_ready=move_ready), object())), \
                patch.object(journey.time, 'perf_counter', return_value=100.):
            self.flow.key('G', 100.)
        move_ready.assert_not_called()  # Startup already entered the original ready pose.
        self.assertTrue(self.flow.want_grasp)
        self.assertEqual(self.flow.grasp_after, 100.)
        self.flow.last_snapshot = self.snapshot(100.1)
        self.flow.last = ['synthetic grasp']
        best = NS(position=np.array([0., 0., .4]), tcp_rotation=np.eye(3))
        with patch('utils.ordinary_grasp.select_best_grasp', return_value=best), \
                patch('utils.camera_utils.compose_cam_to_base_transform', return_value=np.eye(4)), \
                patch('utils.transforms.transform_grasp_pose_to_base', return_value=((0.,)*6, (0.,)*6)), \
                patch.object(journey.grasp, 'pick_and_place', return_value=False) as pick:
            self.flow.pick(self.snapshot(100.2))
        pick.assert_called_once()
        self.assertFalse(self.flow.want_grasp)
        self.assertEqual(self.flow.state.phase, 'GRASP')
        self.flow.robot.switch_gripper_to_posvel.assert_not_called()
        self.flow.robot.head_up.assert_not_called()

    def test_no_grasp_candidate_never_enters_call(self):
        self.flow.want_grasp, self.flow.grasp_after = True, 100.
        self.flow.last_snapshot, self.flow.last = self.snapshot(100.1), []
        with patch('utils.ordinary_grasp.select_best_grasp', return_value=None), \
                patch.object(journey.grasp, 'pick_and_place') as pick:
            self.flow.pick(self.snapshot(100.2))
        pick.assert_not_called()
        self.assertFalse(self.flow.want_grasp)
        self.assertEqual(self.flow.state.phase, 'GRASP')

    def test_u_and_d_require_current_stable_pair_before_queueing_request(self):
        self.evidence(result=self.call_result(stable=False))
        self.flow.key('U', 100.1)
        self.assertIsNone(self.flow.request)
        self.flow.last = self.call_result()
        self.flow.last_snapshot = self.snapshot(99.)  # Expired even though state says STABLE.
        self.flow.key('D', 100.1)
        self.assertIsNone(self.flow.request)
        self.flow.last_snapshot = self.snapshot(100.05)
        self.flow.key('D', 100.1)
        self.assertEqual(self.flow.request, {'value': 'D', 'at': 100.1})

    def test_press_waits_for_a_post_key_result_then_uses_selected_current_rim(self):
        self.flow.options['elevator']['stable_snapshot_age_s'] = 0.  # Explicit legacy mode.
        for command, expected_point in (('U', [.22, -.04, .6]), ('D', [.30, .06, .6])):
            with self.subTest(command=command):
                # Reset only state and robot recording; reuse the same fake IO.
                self.flow.robot.head_up.reset_mock()
                self.evidence(at=100.)
                self.flow.key(command, 100.)
                with patch.object(journey.app, 'execute_button') as execute:
                    self.assertFalse(self.flow.press(self.snapshot(100.1)))
                    execute.assert_not_called()
                    self.flow.last_snapshot = self.snapshot(100.05)
                    self.assertTrue(self.flow.press(self.snapshot(100.15)))
                execute.assert_called_once()
                np.testing.assert_allclose(execute.call_args.args[1],
                    np.asarray(expected_point)+self.options['elevator']['target_offset_base_m'],atol=1e-10)
                self.assertEqual(execute.call_args.args[5], 'contact')
                self.assertEqual(self.flow.state.phase, 'BOARD')
                self.assertIsNone(self.flow.request)
                self.flow.robot.head_up.assert_called_once()

    def remember_call(self, at=100.):
        self.flow.state.enter('CALL', at)
        result=self.call_result()
        self.flow.worker.receive.return_value=(self.snapshot(at),result)
        self.flow.receive(at)
        return self.flow.call_cache

    def test_call_cache_bridges_detector_dropout_and_does_not_need_post_key_inference(self):
        for command in ('U','D'):
            with self.subTest(command=command):
                self.remember_call()
                self.flow.worker.receive.return_value=(self.snapshot(100.3),NS(state='NO_PAIR',pair=()))
                self.flow.receive(100.3)
                self.flow.key(command,100.4)
                self.assertIsNotNone(self.flow.request)
                with patch.object(journey.app,'execute_button') as execute:
                    self.assertTrue(self.flow.press(self.snapshot(100.4)))
                execute.assert_called_once()
                self.assertEqual(self.flow.state.phase,'BOARD')
                self.assertIsNone(self.flow.call_cache)

    def test_call_cache_rejects_changed_pixels_depth_pose_and_expiry(self):
        for change in ('color','depth','pose','time','generation','other_key'):
            with self.subTest(change=change):
                self.remember_call()
                self.flow.last=NS(state='NO_PAIR',pair=())
                self.flow.key('U',100.2)
                current=self.snapshot(100.3)
                if change=='color':current.color[:]=255
                elif change=='other_key':current.color[45:68,48:72]=255
                elif change=='depth':current.depth_mm[:]=410
                elif change=='pose':current.T_base_end[0,3]+=.01
                elif change=='time':current=self.snapshot(103.)
                elif change=='generation':current=self.snapshot(100.3,generation=self.flow.generation+1)
                with patch.object(journey.app,'execute_button') as execute:
                    self.assertFalse(self.flow.press(current))
                execute.assert_not_called()
                self.assertEqual(self.flow.state.phase,'CALL')

    def test_ambiguous_call_result_and_invalidation_clear_saved_pair(self):
        for state in ('EXTRA_BUTTONS','AMBIGUOUS','CLUTTERED'):
            with self.subTest(state=state):
                self.remember_call()
                self.flow.worker.receive.return_value=(self.snapshot(100.2),NS(state=state,pair=()))
                self.flow.receive(100.2)
                self.assertIsNone(self.flow.call_cache)
                self.flow.key('D',100.3)
                self.assertIsNone(self.flow.request)
        self.remember_call()
        self.flow.invalidate()
        self.assertIsNone(self.flow.call_cache)

    def test_missing_depth_does_not_move_or_advance_from_call(self):
        self.evidence()
        self.flow.request = dict(value='U', at=100.)
        self.flow.last_snapshot = self.snapshot(100.05, depth=0)
        with patch.object(journey.app, 'execute_button') as execute:
            self.assertFalse(self.flow.press(self.snapshot(100.15, depth=0)))
        execute.assert_not_called()
        self.flow.robot.move.assert_not_called(); self.flow.robot.head_up.assert_not_called()
        self.assertEqual(self.flow.state.phase, 'CALL')

    def test_pose_change_cannot_use_an_old_rgbd_target(self):
        self.evidence()
        self.flow.request = dict(value='U', at=100.)
        changed = self.pose.copy(); changed[0, 3] += .01
        with patch.object(journey.app, 'execute_button') as execute:
            self.assertFalse(self.flow.press(self.snapshot(100.15, pose=changed)))
        execute.assert_not_called()
        self.assertEqual(self.flow.state.phase, 'CALL')

    def test_board_requires_c_before_floor_selection(self):
        self.flow.state.enter('BOARD', 100.)
        self.flow.last, self.flow.last_snapshot = object(), self.snapshot(100.05)
        self.flow.request = dict(value='U', at=100.)
        for key in ('G', 'N', 'U', 'D', '3', 'ENTER'):
            self.flow.key(key, 100.1)
            self.assertEqual(self.flow.state.phase, 'BOARD')
        self.flow.robot.head_up.assert_not_called()
        self.flow.key('C', 100.2)
        self.assertEqual(self.flow.state.phase, 'FLOOR')
        # The call-button action already restored the head posture. C only
        # clears evidence after boarding and waits for the body/camera to settle.
        self.flow.robot.head_up.assert_not_called()
        self.flow.settle.assert_called_once()
        self.assertEqual(self.flow.generation, 1)
        self.assertIsNone(self.flow.last)
        self.assertIsNone(self.flow.last_snapshot)
        self.assertIsNone(self.flow.request)
        self.flow.camera.warm_up.assert_called_once_with(3)
        self.assertEqual(self.flow.state.retry_index, 0)

    def test_missing_or_duplicated_floor_labels_reject_enter(self):
        for labels, requested in ((('4',), '3'), (('3', '3'), '3')):
            with self.subTest(labels=labels):
                self.evidence('FLOOR', self.panel_result(labels))
                self.flow.request = None
                self.flow.typed = requested
                self.flow.key('ENTER', 100.1)
                self.assertIsNone(self.flow.request)
                self.assertEqual(self.flow.state.phase, 'FLOOR')
                self.assertEqual(self.flow.typed, '')

    def test_floor_input_builds_one_valid_request_and_clears_typed_text(self):
        self.evidence('FLOOR', self.panel_result(('12',)))
        for key in ('1', '3', 'BACKSPACE', '2', 'ENTER'):
            self.flow.key(key, 100.1)
        self.assertEqual(self.flow.request, {'value': '12', 'at': 100.1})
        self.assertEqual(self.flow.typed, '')
        self.flow.robot.move.assert_not_called()

    def save_floor(self,label='3',at=100.05):
        self.flow.state.enter('FLOOR',100.)
        snapshot,result=self.snapshot(at),self.panel_result((label,))
        self.flow.worker.receive.return_value=snapshot,result
        self.flow.receive(at)
        self.assertIn(label,self.flow.floor_cache)
        return snapshot,result

    def test_latched_floor6_survives_long_unstable_or_missing_ocr_and_executes_without_green_ring(self):
        for reading in (None,'6','8'):
            with self.subTest(reading=reading):
                saved,_=self.save_floor('6')
                for index in range(40):
                    now=110.+index*2.
                    result=self.panel_result(('6',))
                    result.buttons[0].text=reading
                    result.buttons[0].confidence=None if reading is None else .99
                    result.buttons[0].status='UNSTABLE'
                    self.flow.worker.receive.return_value=self.snapshot(now),result
                    self.flow.receive(now)
                self.assertIs(self.flow.floor_memory(200.)['6'][0],saved)
                self.flow.key('6',200.);self.flow.commit_floor_input(200.41)
                self.assertEqual(self.flow.request['value'],'6')
                with patch.object(journey.app,'execute_button') as execute:
                    self.assertTrue(self.flow.press(self.snapshot(200.45)))
                execute.assert_called_once()
                expected=np.array([.22,0.,.6])+self.options['elevator']['target_offset_base_m']
                np.testing.assert_allclose(execute.call_args.args[1],expected)
                self.assertEqual(self.flow.floor,'6')

    def test_latched_floor_is_replaced_by_later_unique_stable_identity_at_same_ring(self):
        self.save_floor('6')
        newer=self.snapshot(130.,depth=401)
        self.flow.worker.receive.return_value=newer,self.panel_result(('8',))
        self.flow.receive(130.)
        self.assertNotIn('6',self.flow.floor_memory(180.))
        self.assertIs(self.flow.floor_cache['8'][0],newer)
        self.flow.key('6',180.);self.flow.commit_floor_input(180.41)
        self.assertIsNone(self.flow.request)
        self.flow.robot.move.assert_not_called()

    def test_latched_floor_updates_rgbd_and_applies_multi_label_replacement_atomically(self):
        self.flow.state.enter('FLOOR',100.)
        original=self.snapshot(100.05)
        self.flow.worker.receive.return_value=original,self.panel_result(('6','8'))
        self.flow.receive(100.05)
        newer=self.snapshot(130.,depth=402)
        self.flow.worker.receive.return_value=newer,self.panel_result(('8','6'))
        self.flow.receive(130.)
        memory=self.flow.floor_memory(300.)
        self.assertEqual(set(memory),{'6','8'})
        self.assertIs(memory['6'][0],newer);self.assertIs(memory['8'][0],newer)
        self.assertEqual(memory['6'][2].ring.center,(60,40))
        self.assertEqual(memory['8'][2].ring.center,(40,40))

    def test_latched_mode_never_creates_target_from_only_unstable_or_low_score_text(self):
        self.flow.state.enter('FLOOR',100.)
        for status,confidence in (('UNSTABLE',.99),('STABLE',.2)):
            result=self.panel_result(('6',));result.buttons[0].status=status;result.buttons[0].confidence=confidence
            self.flow.worker.receive.return_value=self.snapshot(100.2),result
            self.flow.receive(100.2)
            self.flow.key('6',100.21);self.flow.commit_floor_input(100.62)
            self.assertEqual(self.flow.floor_cache,{})
            self.assertIsNone(self.flow.request)
        self.flow.robot.move.assert_not_called()

    def test_conflicting_stable_labels_at_one_rim_are_never_latched(self):
        self.flow.state.enter('FLOOR',100.)
        result=self.panel_result(('6','8'))
        result.buttons[1].ring=result.buttons[0].ring
        self.flow.worker.receive.return_value=self.snapshot(100.2),result
        self.flow.receive(100.2)
        self.assertEqual(self.flow.floor_memory(180.),{})
        self.flow.last=None;self.flow.last_snapshot=None
        self.flow.key('6',180.);self.flow.commit_floor_input(180.41)
        self.assertIsNone(self.flow.request)
        self.flow.robot.move.assert_not_called()

    def test_latched_cache_still_blocks_changed_scene_even_after_long_wait(self):
        for change in ('color','depth','pose','generation','intrinsics'):
            with self.subTest(change=change):
                self.save_floor('6')
                self.flow.key('6',200.);self.flow.commit_floor_input(200.41)
                current=self.snapshot(200.45)
                if change=='color':current.color[:]=255
                elif change=='depth':current.depth_mm[:]=410
                elif change=='pose':current.T_base_end[0,3]+=.01
                elif change=='generation':current=self.snapshot(200.45,generation=self.flow.generation+1)
                elif change=='intrinsics':current.K[0,0]*=1.01
                with patch.object(journey.app,'execute_button') as execute:
                    self.assertFalse(self.flow.press(current))
                execute.assert_not_called()

    def test_r_clears_floor_memory_and_pending_selection_without_moving(self):
        self.save_floor('6')
        self.flow.key('6',100.1)
        self.flow.key('R',100.2)
        self.assertEqual(self.flow.floor_cache,{})
        self.assertIsNone(self.flow.request);self.assertEqual(self.flow.typed,'')
        self.assertEqual(self.flow.state.phase,'FLOOR')
        self.flow.robot.move.assert_not_called();self.flow.robot.head_up.assert_not_called()

    def test_floor_cache_executes_after_target_rim_drops_out_without_new_ocr(self):
        saved,_=self.save_floor()
        current=self.panel_result(('4',))
        # The latest detector omits floor 3 while its actual pixels stay unchanged.
        current.buttons[0].ring=perception.Ring(70.,55.,10.,1.)
        self.flow.worker.receive.return_value=self.snapshot(100.2),current
        self.flow.receive(100.2)
        self.flow.key('3',100.21)
        self.flow.commit_floor_input(100.61)
        self.assertEqual(self.flow.request['value'],'3')
        with patch.object(journey.app,'execute_button') as execute:
            self.assertTrue(self.flow.press(self.snapshot(100.65,depth=401)))
        execute.assert_called_once()
        # Freeze the original depth/pose, rather than mixing current depth with old OCR.
        expected=np.array([.22,0.,.6])+self.options['elevator']['target_offset_base_m']
        np.testing.assert_allclose(execute.call_args.args[1],expected)
        self.assertEqual(self.flow.floor,'3')
        self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')
        self.assertEqual(self.flow.floor_cache,{})

    def test_floor_cache_can_bridge_entire_panel_detector_dropout(self):
        self.save_floor()
        self.flow.last=perception.FrameResult(100.2,(80,100))
        self.flow.last_snapshot=self.snapshot(100.2)
        self.flow.tracker.panel.stable=False
        self.flow.key('3',100.21)
        self.flow.commit_floor_input(100.61)
        with patch.object(journey.app,'execute_button') as execute:
            self.assertTrue(self.flow.press(self.snapshot(100.65)))
        execute.assert_called_once()

    def test_floor_cache_refreshes_from_verified_label_memory_without_current_ocr(self):
        self.save_floor()
        snapshot,result=self.snapshot(100.2,depth=401),self.panel_result(('3',))
        button=result.buttons[0]
        button.text=None;button.confidence=None;button.status='NO_TEXT'
        result.tracking.update(workflow_evidence=True,
            region_evidence={'0':dict(track_id=7,source='PIXEL_MATCHED_CACHE',pixel_match=True)},
            display_label_cache={'0':dict(text='3',confidence=.99)})
        self.flow.tracker.remembered_associations.reset_mock()
        self.flow.tracker.remembered_associations.return_value=(NS(label='3',observation_index=0),)
        self.flow.worker.receive.return_value=snapshot,result
        self.flow.receive(100.2)
        self.flow.tracker.remembered_associations.assert_called_once()
        self.assertIs(self.flow.floor_cache['3'][0],snapshot)
        self.assertIsNone(self.flow.floor_cache['3'][2].text)
        self.flow.last=perception.FrameResult(100.3,(80,100))
        self.flow.last_snapshot=self.snapshot(100.3)
        self.flow.tracker.panel.stable=False
        self.flow.key('3',100.3);self.flow.commit_floor_input(100.71)
        with patch.object(journey.app,'execute_button') as execute:
            self.assertTrue(self.flow.press(self.snapshot(100.75,depth=402)))
        expected=np.array([.2199,0.,.601])+self.options['elevator']['target_offset_base_m']
        np.testing.assert_allclose(execute.call_args.args[1],expected)

    def test_floor_cache_rejects_changed_scene_pose_depth_and_confirmed_conflicts(self):
        for change in ('color','depth','pose','generation','conflict','duplicate'):
            with self.subTest(change=change):
                self.save_floor()
                self.flow.key('3',100.1);self.flow.commit_floor_input(100.5)
                current=self.snapshot(100.6)
                if change=='color':current.color[:]=255
                elif change=='depth':current.depth_mm[:]=410
                elif change=='pose':current.T_base_end[0,3]+=.01
                elif change=='generation':current=self.snapshot(100.6,generation=self.flow.generation+1)
                elif change=='conflict':self.flow.last=self.panel_result(('4',))
                elif change=='duplicate':self.flow.last=self.panel_result(('3','3'))
                with patch.object(journey.app,'execute_button') as execute:
                    self.assertFalse(self.flow.press(current))
                execute.assert_not_called()

    def test_legacy_timed_mode_still_rejects_expiration_and_unstable_variants(self):
        self.flow.options['elevator']['floor_cache_until_update']=False
        for change in ('time','variant_conflict'):
            self.save_floor()
            self.flow.key('3',100.1);self.flow.commit_floor_input(100.5)
            current=self.snapshot(102.2 if change=='time' else 100.6)
            if change=='variant_conflict':
                self.flow.last=self.panel_result(())
                self.flow.last.buttons=[perception.ButtonObservation(perception.Ring(40.,40.,10.,1.),
                    variants=[('3',.99),('8',.99)])]
            with patch.object(journey.app,'execute_button') as execute:
                self.assertFalse(self.flow.press(current))
            execute.assert_not_called()

    def test_floor_cache_zero_restores_post_input_only_mode_and_invalidation_clears_it(self):
        self.flow.options['elevator']['floor_cache_until_update']=False
        self.save_floor()
        self.flow.options['elevator']['stable_snapshot_age_s']=0.
        self.flow.request=dict(value='3',at=100.1)
        with patch.object(journey.app,'execute_button') as execute:
            self.assertFalse(self.flow.press(self.snapshot(100.2)))
        execute.assert_not_called()
        self.assertEqual(self.flow.floor_cache,{})
        self.flow.options['elevator']['stable_snapshot_age_s']=2.
        self.save_floor()
        self.flow.key('3',100.1)
        self.flow.invalidate()
        self.assertEqual(self.flow.floor_cache,{})
        self.assertIsNone(self.flow.floor_input_due)
        self.assertEqual(self.flow.typed,'')

    def test_single_digit_automatically_submits_after_pause_without_enter(self):
        self.evidence('FLOOR',self.panel_result(('3',)))
        self.flow.key('3',100.1)
        self.flow.commit_floor_input(100.49)
        self.assertIsNone(self.flow.request)
        self.flow.commit_floor_input(100.51)
        self.assertEqual(self.flow.request,{'value':'3','at':100.51})
        self.assertEqual(self.flow.typed,'')
        self.assertIsNone(self.flow.floor_input_due)

    def test_two_digits_and_negative_floors_commit_together_without_enter(self):
        for label in ('12','-1','-12','0'):
            with self.subTest(label=label):
                self.flow.invalidate()
                self.flow.tracker.panel.stable=True
                self.evidence('FLOOR',self.panel_result((label,)))
                for index,char in enumerate(label):
                    at=100.1+index*.1
                    self.flow.key(char,at);self.flow.commit_floor_input(at)
                    self.assertIsNone(self.flow.request)
                self.flow.commit_floor_input(at+.41)
                self.assertEqual(self.flow.request['value'],label)

    def test_incomplete_minus_backspace_invalid_length_and_other_phases_do_not_submit(self):
        self.evidence('FLOOR',self.panel_result(('12',)))
        self.flow.key('-',100.1);self.flow.commit_floor_input(1000.)
        self.assertIsNone(self.flow.request)
        self.flow.key('BACKSPACE',100.2)
        for key in ('1','3','BACKSPACE','2'):self.flow.key(key,100.3)
        self.flow.commit_floor_input(100.71)
        self.assertEqual(self.flow.request['value'],'12')
        self.flow.key('1',100.8);self.flow.key('2',100.9);self.flow.key('3',101.)
        self.assertIsNone(self.flow.request);self.assertEqual(self.flow.typed,'')
        self.flow.state.enter('CALL',101.)
        self.flow.key('3',101.1);self.flow.commit_floor_input(102.)
        self.assertIsNone(self.flow.request)

    def test_terminal_digit_needs_no_newline_and_settings_are_restored(self):
        master,slave=pty.openpty()
        stream=os.fdopen(slave,'r',buffering=1)
        original=termios.tcgetattr(stream.fileno())
        try:
            with patch.object(journey.sys,'stdin',stream):
                self.evidence('FLOOR',self.panel_result(('3',)))
                self.flow.enable_floor_terminal()
                self.assertIsNotNone(self.flow.terminal_state)
                os.write(master,b'3')
                key=self.flow.read_terminal()
                self.assertEqual(key,'3')
                self.flow.key(key,100.1);self.flow.commit_floor_input(100.51)
                self.assertEqual(self.flow.request['value'],'3')
                self.flow.restore_terminal()
                self.assertEqual(termios.tcgetattr(stream.fileno()),original)
        finally:
            self.flow.restore_terminal();stream.close();os.close(master)

    def test_terminal_q_and_stop_work_while_per_key_input_is_active(self):
        master,slave=pty.openpty();stream=os.fdopen(slave,'r',buffering=1)
        try:
            with patch.object(journey.sys,'stdin',stream),patch.object(journey.cv2,'waitKey',return_value=-1), \
                    patch.object(journey.cv2,'getWindowProperty',return_value=1):
                self.flow.enable_floor_terminal()
                for key,home in ((b'q',True),(b' ',False),(b's',False)):
                    os.write(master,key)
                    self.assertTrue(self.flow.cancelled())
                    self.assertEqual(self.flow.quit_home_requested,home)
        finally:
            self.flow.restore_terminal();stream.close();os.close(master)

    def test_floor_press_records_target_restores_display_pose_and_clears_old_evidence(self):
        self.evidence('FLOOR', self.panel_result(('3',)))
        self.flow.request = dict(value='3', at=100.)
        with patch.object(journey.app, 'execute_button') as execute, \
                patch.object(journey.time, 'perf_counter', return_value=101.):
            self.assertTrue(self.flow.press(self.snapshot(100.15)))
        execute.assert_called_once()
        target = execute.call_args.args[1]
        expected = np.array([.22, 0., .6])+self.options['elevator']['target_offset_base_m']
        np.testing.assert_allclose(target, expected, atol=1e-10)
        self.assertEqual(self.flow.floor, '3')
        self.assertEqual(self.flow.state.phase, 'WAIT_DISPLAY')
        np.testing.assert_allclose(self.flow.robot.head_up.call_args.args[0][:6], self.settings['display_view_deg'])
        self.assertIsNone(self.flow.arrival_deadline)
        self.assertIsNone(self.flow.last); self.assertIsNone(self.flow.last_snapshot)
        self.assertIsNone(self.flow.request)
        self.assertEqual(self.flow.generation, 1)
        with patch.object(journey.time, 'perf_counter', return_value=102.):
            self.flow.key('c',102.)
        self.assertEqual(self.flow.state.phase,'DISPLAY')
        self.assertEqual(self.flow.arrival_deadline,102.+self.settings['arrival_timeout_s'])
        self.flow.robot.head_up.assert_called_once()

    def test_n_skips_call_and_pre_call_wait_without_visual_evidence(self):
        for phase in ('WAIT_CALL','CALL'):
            with self.subTest(phase=phase):
                self.flow.state.enter(phase,100.)
                self.flow.last=self.flow.last_snapshot=None
                self.flow.call_cache=(object(),object())
                self.flow.floor_cache={'3':(object(),object(),object())}
                self.flow.request=dict(value='U',at=100.)
                self.flow.robot.stop.reset_mock();self.flow.robot.head_up.reset_mock()
                with patch.object(journey.app,'execute_button') as execute:
                    self.flow.key('n',100.1)
                execute.assert_not_called()
                self.flow.robot.stop.assert_called_once()
                self.flow.robot.head_up.assert_called_once()
                self.assertEqual(self.flow.state.phase,'BOARD')
                self.assertIsNone(self.flow.request)
                self.assertIsNone(self.flow.call_cache);self.assertEqual(self.flow.floor_cache,{})
                self.assertFalse(self.flow.home)

    def test_n_skips_floor_preserving_pending_or_typed_arrival_target(self):
        for selected,typed in (('3',''),(None,'12'),(None,'-1')):
            with self.subTest(selected=selected,typed=typed):
                self.flow.state.enter('FLOOR',100.)
                self.flow.floor=None;self.flow.typed=typed;self.flow.floor_input_due=101.
                self.flow.request=None if selected is None else dict(value=selected,at=100.)
                self.flow.robot.head_up.reset_mock()
                with patch.object(journey.app,'execute_button') as execute:
                    self.flow.key('N',100.1)
                execute.assert_not_called()
                self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')
                self.assertEqual(self.flow.floor,selected or typed)
                self.assertIsNone(self.flow.arrival_deadline)
                self.assertIsNone(self.flow.request);self.assertIsNone(self.flow.floor_input_due)
                np.testing.assert_allclose(self.flow.robot.head_up.call_args.args[0][:6],self.settings['display_view_deg'])

    def test_skip_floor_without_target_allows_arrival_only_input_and_never_presses(self):
        self.flow.state.enter('FLOOR',100.)
        self.flow.key('N',100.1)
        self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')
        self.assertIsNone(self.flow.floor);self.assertIsNone(self.flow.arrival_deadline)
        self.flow.key('C',100.15)
        # No button panel or OCR is required to set the LCD comparison target.
        self.assertIsNone(self.flow.last)
        self.flow.key('-',100.2);self.flow.key('1',100.3)
        with patch.object(journey.app,'execute_button') as execute:
            self.flow.commit_floor_input(100.71)
        execute.assert_not_called();self.flow.robot.move.assert_not_called()
        self.assertEqual(self.flow.floor,'-1')
        self.assertEqual(self.flow.state.phase,'DISPLAY')
        self.assertEqual(self.flow.arrival_deadline,100.71+self.settings['arrival_timeout_s'])
        self.assertIsNone(self.flow.request)

    def test_n_in_display_marks_normal_home_exit_and_discards_pending_input(self):
        self.flow.state.enter('DISPLAY',100.)
        self.flow.floor='3';self.flow.typed='12';self.flow.floor_input_due=101.
        self.flow.key('N',100.1)
        self.assertEqual(self.flow.state.phase,'DONE');self.assertTrue(self.flow.home)
        self.flow.robot.stop.assert_called_once();self.flow.robot.head_up.assert_not_called()
        self.assertEqual(self.flow.typed,'');self.assertIsNone(self.flow.floor_input_due)
        self.assertIsNone(self.flow.arrival_deadline)

    def test_failed_stop_during_skip_does_not_move_or_request_home(self):
        self.flow.state.enter('CALL',100.)
        self.flow.robot.stop.side_effect=RuntimeError('synthetic failed stop')
        with self.assertRaisesRegex(RuntimeError,'synthetic failed stop'):
            self.flow.key('N',100.1)
        self.flow.robot.head_up.assert_not_called();self.assertFalse(self.flow.home)
        self.assertEqual(self.flow.state.phase,'CALL');self.assertFalse(self.flow.skip_in_progress)

    def test_motion_cancelled_by_n_advances_only_after_stop(self):
        self.flow.state.enter('CALL',100.)
        events=[]
        self.flow.robot.stop.side_effect=lambda:events.append('stop')
        self.flow.robot.head_up.side_effect=lambda *a,**kw:events.append('head')
        def interrupted():
            self.flow.skip_requested=True
            self.flow.robot.stop()
            raise journey.motion.MotionCancelled('N')
        self.flow.stage_action(interrupted)
        self.assertEqual(self.flow.state.phase,'BOARD')
        self.assertGreater(events.index('head'),events.index('stop'))
        self.assertFalse(self.flow.skip_requested);self.assertFalse(self.flow.home)

    def test_n_repeat_during_skip_posture_does_not_skip_an_additional_stage(self):
        self.flow.state.enter('CALL',100.)
        def head(*args,**kwargs):
            with patch.object(journey.cv2,'waitKey',return_value=ord('n')), \
                    patch.object(journey.cv2,'getWindowProperty',return_value=1), \
                    patch.object(journey.sys,'stdin',NS(isatty=lambda:False)):
                self.assertFalse(self.flow.cancelled())
        self.flow.robot.head_up.side_effect=head
        self.flow.key('N',100.1)
        self.assertEqual(self.flow.state.phase,'BOARD')

    def test_unrelated_control_error_after_skip_request_does_not_advance(self):
        self.flow.state.enter('FLOOR',100.)
        def failed():
            self.flow.skip_requested=True
            raise RuntimeError('SDK stop failed')
        with self.assertRaisesRegex(RuntimeError,'SDK stop failed'):self.flow.stage_action(failed)
        self.assertEqual(self.flow.state.phase,'FLOOR')
        self.flow.robot.head_up.assert_not_called();self.assertFalse(self.flow.home)

    def test_skip_during_button_motion_retracts_to_original_standoff_before_head(self):
        self.flow.state.enter('FLOOR',100.)
        self.flow.request=dict(value='3',at=100.)
        retreat=np.eye(4);retreat[:3,3]=[.3,-.1,.4]
        self.flow.skip_retreat=retreat.copy()
        events=[]
        self.flow.robot.stop.side_effect=lambda:events.append('stop')
        self.flow.robot.move.side_effect=lambda *a:events.append('retreat')
        self.flow.robot.head_up.side_effect=lambda *a,**kw:events.append('head')
        self.flow.key('N',100.1)
        self.assertEqual(events,['stop','retreat','head'])
        np.testing.assert_allclose(self.flow.robot.move.call_args.args[0],retreat)
        self.assertIsNone(self.flow.skip_retreat)
        self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')

    def test_failed_skip_retreat_does_not_change_posture_or_request_home(self):
        self.flow.state.enter('FLOOR',100.)
        self.flow.skip_retreat=np.eye(4)
        self.flow.robot.move.side_effect=RuntimeError('synthetic retreat failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic retreat failure'):
            self.flow.key('N',100.1)
        self.flow.robot.head_up.assert_not_called();self.assertFalse(self.flow.home)
        self.assertEqual(self.flow.state.phase,'FLOOR')

    def test_n_before_first_button_move_does_not_move_toward_standoff(self):
        self.evidence('FLOOR',self.panel_result(('3',)))
        self.flow.request=dict(value='3',at=100.)
        def cancelled_before_move(*a,**kw):
            self.flow.skip_requested=True
            raise journey.motion.MotionCancelled('N before SDK sends a goal')
        with patch.object(journey.app,'execute_button',side_effect=cancelled_before_move):
            self.flow.stage_action(self.flow.press,self.snapshot(100.15))
        self.flow.robot.move.assert_not_called()
        self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')
        self.assertEqual(self.flow.floor,'3')

    def test_free_position_press_skips_level_pose_and_n_uses_position_retreat(self):
        from utils.transforms import pose6d_to_mat4
        self.pose[:]=pose6d_to_mat4(.26,0,.2,.1,-.6,.2)
        self.flow.button_orientation='free'
        self.evidence('FLOOR',self.panel_result(('3',)))
        self.flow.request=dict(value='3',at=100.)
        self.flow.robot.move_position=Mock(return_value=np.eye(4))
        def interrupted(*args,**kwargs):
            self.assertTrue(kwargs['position_only'])
            self.assertGreater(abs(args[2][2,0]),.1)
            self.pose[0,3]+=.01;self.flow.skip_requested=True
            raise journey.motion.MotionCancelled('N')
        with patch.object(journey.app,'execute_button',side_effect=interrupted), \
                patch.object(journey.geometry,'level_pose',side_effect=AssertionError('must not level')):
            self.flow.stage_action(self.flow.press,self.snapshot(100.15))
        self.flow.robot.move.assert_not_called();self.flow.robot.move_position.assert_called_once()
        self.assertEqual(self.flow.state.phase,'WAIT_DISPLAY')

    def test_display_historical_or_unstable_value_cannot_be_ready(self):
        for state, current in (('NO_TEXT', None), ('UNSTABLE', '3'), ('STABLE', None)):
            with self.subTest(state=state, current=current):
                result = NS(state=state, current_floor=current, last_confirmed_floor='3')
                self.evidence('DISPLAY', result)
                self.assertFalse(self.flow.ready(100.1))
        self.evidence('DISPLAY', NS(state='STABLE', current_floor='4'))
        self.flow.floor = '3'
        # A real non-target floor pauses viewpoint retries during elevator travel.
        self.assertTrue(self.flow.ready(100.1))

    def test_receive_discards_expired_future_and_cross_generation_results(self):
        sentinel = object()
        self.flow.last = sentinel
        for snap in (self.snapshot(99.), self.snapshot(100.2), self.snapshot(100., generation=-1)):
            with self.subTest(at=snap.captured_at, generation=snap.generation):
                self.flow.worker.receive.return_value = (snap, object())
                self.flow.receive(100.1)
                self.assertIs(self.flow.last, sentinel)
                self.assertIsNone(self.flow.last_snapshot)
        self.flow.tracker.update.assert_not_called()

    def test_receive_current_floor_panel_updates_tracker_with_capture_pose(self):
        self.flow.state.enter('FLOOR', 100.)
        snap, result = self.snapshot(100.05), self.panel_result()
        self.flow.worker.receive.return_value = (snap, result)
        self.flow.receive(100.1)
        self.assertIs(self.flow.last_snapshot, snap)
        self.assertIs(self.flow.last, result)
        self.flow.tracker.update.assert_called_once_with(result, self.flow.generation,
                                                        camera_pose=snap.T_base_end, now=100.1)


if __name__ == '__main__':
    unittest.main()
