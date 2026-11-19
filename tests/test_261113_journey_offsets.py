"""Offline compensation checks; no camera, robot or OCR engine is started."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from elevator import app, geometry, journey_261113 as journey, perception
from utils.camera_utils import compose_cam_to_base_transform
from utils.transforms import pose6d_to_mat4


ROOT = Path(__file__).resolve().parents[1]


class JourneyOffsetTests(unittest.TestCase):
    def setUp(self):
        self.options = app.load_workflow(ROOT/'config/261105_my_elevator.yaml')
        self.options['elevator']['refinement']['enabled'] = False
        self.settings = journey.load_settings(ROOT/'config/261113_my_journey.yaml',self.options)
        self.original_floor_offset = np.asarray(self.options['elevator']['target_offset_base_m'], float)

    def test_default_null_explicitly_inherits_the_current_floor_configuration(self):
        self.assertIsNone(self.settings['call_target_offset_base_m'])
        effective = journey.effective_call_offset(self.settings, self.options)
        self.assertIsInstance(effective, np.ndarray)
        np.testing.assert_allclose(effective, self.original_floor_offset)

    def test_inheritance_follows_updated_floor_value_without_copying_a_constant(self):
        self.settings['call_target_offset_base_m'] = None
        for value in ([.01, -.02, .03], [-.004, .005, .006]):
            self.options['elevator']['target_offset_base_m'] = value
            effective = journey.effective_call_offset(self.settings, self.options)
            np.testing.assert_allclose(effective, value)

    def test_explicit_override_including_zero_remains_effective(self):
        for value in ([.01, -.02, .03], [0., 0., 0.]):
            with self.subTest(value=value):
                self.settings['call_target_offset_base_m'] = value
                effective = journey.effective_call_offset(self.settings, self.options)
                np.testing.assert_allclose(effective, value)
                np.testing.assert_allclose(self.options['elevator']['target_offset_base_m'], self.original_floor_offset)

    def test_invalid_effective_vectors_fail_before_motion(self):
        for value in ([1., 2.], [float('nan'), 0., 0.], [0., float('inf'), 0.], 'bad'):
            with self.subTest(value=value):
                self.settings['call_target_offset_base_m'] = value
                with self.assertRaises((ValueError, TypeError)):
                    journey.effective_call_offset(self.settings, self.options)
        self.settings['call_target_offset_base_m'] = None
        self.options['elevator']['target_offset_base_m'] = [0., float('nan'), 0.]
        with self.assertRaises((ValueError, TypeError)):
            journey.effective_call_offset(self.settings, self.options)

    def press_target(self, phase, command, pixel, *, override=None):
        cfg = {'camera': {'type': 'realsense_d405'}, 'robot': {},
               'calibration': {'hand_eye_compensation_m': {'x': -.02, 'y': -.02, 'z': -.05}}}
        settings = dict(self.settings, call_target_offset_base_m=override)
        args = NS(place_xyz_m=[.25, -.38, .35], place_rpy_deg=None,
                  place_duration_s=3., release_wait_s=.5, press_stage='contact')
        flow = journey.Journey(cfg, self.options, settings, args)
        pose = np.eye(4); pose[:3, 3] = [.26, 0., .2]
        flow.robot = NS(pose=Mock(return_value=pose.copy()))
        flow.head = Mock()  # Suppress post-contact posture actions in this test.
        flow.hand_eye = np.eye(4)
        flow.tracker = NS(panel=NS(stable=True), remembered_associations=lambda *a, **kw: ())
        flow.state.enter(phase, 100.)
        ring = perception.Ring(float(pixel[0]), float(pixel[1]), 12., 1.)
        if phase == 'CALL':
            other_y = pixel[1]+25 if command == 'U' else pixel[1]-25
            other = perception.Ring(float(pixel[0]), float(other_y), 12., 1.)
            pair = (ring, other) if command == 'U' else (other, ring)
            flow.last = NS(state='STABLE', pair=pair)
        else:
            button = perception.ButtonObservation(ring, command, .99,
                       (pixel[0]-3, pixel[1]-3, pixel[0]+3, pixel[1]+3), status='STABLE')
            flow.last = perception.FrameResult(100.05, (80, 100), (5, 5, 95, 75), 'TEST', [button])
        K = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
        def snapshot(at):
            return geometry.Snapshot(1, at, np.zeros((80, 100, 3), np.uint8),
                     np.full((80, 100), 400, np.uint16), K, np.zeros(5), pose.copy(), flow.generation)
        flow.last_snapshot = snapshot(100.05)
        flow.request = dict(value=command, at=100.)
        with patch.object(journey.app, 'execute_button') as execute, \
                patch.object(journey.geometry, 'offset_target_base', wraps=geometry.offset_target_base) as offset, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(flow.press(snapshot(100.15)))
        execute.assert_called_once()
        # A helper may validate an offset with point=None; only one invocation
        # may apply compensation to an actual measured base point.
        applied = [c for c in offset.call_args_list if c.args and c.args[0] is not None]
        self.assertEqual(len(applied), 1)
        self.assertEqual(execute.call_args.args[5], 'contact')
        self.assertIs(execute.call_args.args[3], self.options['elevator'])
        return np.asarray(execute.call_args.args[1]), np.asarray(applied[0].args[0])

    def test_u_d_and_floor_use_equal_corrected_points_for_equal_rgbd_and_pose(self):
        for direction, pixel in (('U', (40, 30)), ('D', (60, 55))):
            with self.subTest(direction=direction):
                call, call_raw = self.press_target('CALL', direction, pixel)
                floor, floor_raw = self.press_target('FLOOR', '3', pixel)
                expected_raw = np.array([.24+(pixel[0]-50)*.004,
                                         -.02+(pixel[1]-40)*.004, .55])
                np.testing.assert_allclose(call_raw, expected_raw, atol=1e-10)
                np.testing.assert_allclose(floor_raw, expected_raw, atol=1e-10)
                np.testing.assert_allclose(call, floor, atol=1e-10)
                np.testing.assert_allclose(call, expected_raw+self.original_floor_offset, atol=1e-10)

    def test_nonzero_correction_is_applied_once_and_handeye_offset_is_not_doubled(self):
        # Keep this regression meaningful when the active trial configuration
        # deliberately sets the per-button correction to zero.
        self.options['elevator']['target_offset_base_m'] = [.070, .0265, .0225]
        self.original_floor_offset = np.asarray(self.options['elevator']['target_offset_base_m'], float)
        point, raw = self.press_target('CALL', 'U', (40, 30))
        np.testing.assert_allclose(raw, [.20, -.06, .55], atol=1e-10)
        np.testing.assert_allclose(point-raw, self.original_floor_offset, atol=1e-10)
        self.assertFalse(np.allclose(point-raw, 2*self.original_floor_offset))

    def test_override_changes_only_call_and_explicit_zero_keeps_raw_point(self):
        call, raw = self.press_target('CALL', 'U', (40, 30), override=[0., 0., 0.])
        np.testing.assert_allclose(call, raw, atol=1e-10)
        overridden, raw = self.press_target('CALL', 'U', (40, 30), override=[.01, .02, .03])
        np.testing.assert_allclose(overridden, raw+[.01, .02, .03], atol=1e-10)
        floor, floor_raw = self.press_target('FLOOR', '3', (40, 30), override=[.01, .02, .03])
        np.testing.assert_allclose(floor, floor_raw+self.original_floor_offset, atol=1e-10)

    def test_global_trial_keeps_button_targets_unchanged_across_poses(self):
        # Explicit trial fixtures keep this mathematical regression valid even
        # after the operator restores the two active YAML configurations.
        cfg = {'calibration': {'hand_eye_compensation_m': dict(zip('xyz', [.05, .0065, -.0275]))}}
        trial_options = {'elevator': {'target_offset_base_m': [0., 0., 0.]}}
        trial_settings = {'call_target_offset_base_m': None}
        old_cfg = {'calibration': {'hand_eye_compensation_m': dict(zip('xyz', [-.02, -.02, -.05]))}}
        old_button = np.array([.070, .0265, .0225])
        hand_eye = pose6d_to_mat4(.03, -.01, .08, .1, -.2, .3)
        for pose, point in (
                ((.26, 0., .192, 0., 0., 0.), [.02, -.03, .4]),
                ((.30, -.10, .35, .2, -.6, .5), [-.1, .04, .6]),
                ((.20, .15, .40, -.3, .4, -1.0), [.12, -.06, .25])):
            with self.subTest(pose=pose, point=point):
                tcp = pose6d_to_mat4(*pose)
                homogeneous = np.r_[point, 1.]
                old_base = (compose_cam_to_base_transform(tcp, hand_eye, old_cfg)@homogeneous)[:3]
                new_base = (compose_cam_to_base_transform(tcp, hand_eye, cfg)@homogeneous)[:3]
                expected = geometry.offset_target_base(old_base, old_button)
                floor = geometry.offset_target_base(new_base, trial_options['elevator']['target_offset_base_m'])
                call = geometry.offset_target_base(new_base, journey.effective_call_offset(trial_settings, trial_options))
                np.testing.assert_allclose(floor, expected, atol=1e-12)
                np.testing.assert_allclose(call, expected, atol=1e-12)

    def test_global_trial_moves_visual_grasp_by_previous_button_correction(self):
        cfg = {'calibration': {'hand_eye_compensation_m': dict(zip('xyz', [.05, .0065, -.0275]))}}
        old_cfg = {'calibration': {'hand_eye_compensation_m': dict(zip('xyz', [-.02, -.02, -.05]))}}
        delta = np.array([.070, .0265, .0225])
        hand_eye = pose6d_to_mat4(-.01, .02, .04, -.2, .15, .4)
        for pose, point in (
                ((.26, 0., .192, 0., 0., 0.), [.02, -.03, .4]),
                ((.30, -.10, .35, .2, -.6, .5), [-.1, .04, .6]),
                ((.20, .15, .40, -.3, .4, -1.0), [.12, -.06, .25])):
            with self.subTest(pose=pose, point=point):
                tcp = pose6d_to_mat4(*pose)
                old = (compose_cam_to_base_transform(tcp, hand_eye, old_cfg)@np.r_[point, 1.])[:3]
                trial = (compose_cam_to_base_transform(tcp, hand_eye, cfg)@np.r_[point, 1.])[:3]
                np.testing.assert_allclose(trial-old, delta, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
