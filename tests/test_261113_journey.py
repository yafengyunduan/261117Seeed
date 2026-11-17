"""Offline timing, shared-model and joint-target checks for the journey entry.

All robot feedback is synthetic. These tests never create a camera, motor
controller or OCR model, and never send a joint target.
"""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from elevator import journey_261113 as journey
from elevator import floor_display_261113 as display


class SearchStateTests(unittest.TestCase):
    def test_default_search_window_is_ten_seconds_for_each_stage(self):
        for phase, deltas in (('CALL',(-15.,-15.)),('FLOOR',(-15.,-15.)),('DISPLAY',(-15.,-10.))):
            with self.subTest(phase=phase):
                state=journey.SearchState()
                self.assertEqual(state.timeout_s,10.)
                state.enter(phase,100.)
                for now,delta in zip((110.,125.),deltas):
                    self.assertIsNone(state.tick(now-.01,ready=False))
                    self.assertEqual(state.tick(now,ready=False),delta)
                    state.retry_complete(now+5.)
                self.assertIsNone(state.tick(139.99,ready=False))
                self.assertEqual(state.tick(140.,ready=False),'HOME')

    def test_start_requires_explicit_stage_and_grasp_has_no_search_timeout(self):
        state = journey.SearchState(timeout_s=8.)
        self.assertEqual(state.phase, 'GRASP')
        self.assertIsNone(state.tick(1000., ready=False))
        self.assertEqual(state.phase, 'GRASP')

    def test_call_and_floor_allow_exactly_two_relative_fifteen_degree_retries(self):
        for phase in ('CALL', 'FLOOR'):
            with self.subTest(phase=phase):
                state = journey.SearchState(timeout_s=8.)
                state.enter(phase, 100.)
                self.assertEqual(state.retry_index, 0)
                self.assertEqual(state.deadline, 108.)
                self.assertIsNone(state.tick(107.99, ready=False))
                self.assertEqual(state.tick(108., ready=False), -15.)
                self.assertEqual(state.retry_index, 1)
                self.assertIsNone(state.deadline)
                # The motion itself consumes five seconds; all eight seconds
                # for the next image search begin after the motion completes.
                state.retry_complete(113.)
                self.assertEqual(state.deadline, 121.)
                self.assertIsNone(state.tick(120.99, ready=False))
                self.assertEqual(state.tick(121., ready=False), -15.)
                self.assertEqual(state.retry_index, 2)
                state.retry_complete(125.)
                self.assertIsNone(state.tick(132.99, ready=False))
                self.assertEqual(state.tick(133., ready=False), 'HOME')
                self.assertEqual(state.phase, 'TIMEOUT')

    def test_display_uses_fifteen_then_ten_degrees_then_home(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('DISPLAY', 10.)
        self.assertEqual(state.tick(18., ready=False), -15.)
        state.retry_complete(22.)
        self.assertEqual(state.tick(30., ready=False), -10.)
        state.retry_complete(34.)
        self.assertEqual(state.tick(42., ready=False), 'HOME')
        self.assertEqual(state.phase, 'TIMEOUT')

    def test_ready_pauses_timeout_while_operator_selects_a_button(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('CALL', 1.)
        self.assertIsNone(state.tick(2., ready=True))
        self.assertIsNone(state.deadline)
        self.assertIsNone(state.tick(500., ready=True))
        self.assertEqual(state.retry_index, 0)
        self.assertEqual(state.phase, 'CALL')
        self.assertIsNone(state.tick(501., ready=False))
        self.assertEqual(state.deadline, 509.)
        self.assertIsNone(state.tick(508.99, ready=False))
        self.assertEqual(state.tick(509., ready=False), -15.)

    def test_recovered_panel_does_not_erase_retry_budget_in_the_same_stage(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('FLOOR', 0.)
        self.assertEqual(state.tick(8., ready=False), -15.)
        state.retry_complete(12.)
        self.assertIsNone(state.tick(13., ready=True))
        self.assertEqual(state.retry_index, 1)
        self.assertIsNone(state.tick(50., ready=False))
        self.assertEqual(state.tick(58., ready=False), -15.)
        self.assertEqual(state.retry_index, 2)

    def test_new_stage_resets_retry_counter_and_has_its_own_full_window(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('CALL', 0.)
        self.assertEqual(state.tick(8., ready=False), -15.)
        state.retry_complete(12.)
        self.assertEqual(state.tick(20., ready=False), -15.)
        state.enter('FLOOR', 100.)
        self.assertEqual(state.retry_index, 0)
        self.assertEqual(state.deadline, 108.)
        self.assertEqual(state.tick(108., ready=False), -15.)

    def test_manual_waits_grasp_and_done_do_not_trigger_search_joint_moves(self):
        for phase in ('GRASP', 'WAIT_CALL', 'BOARD', 'DONE'):
            with self.subTest(phase=phase):
                state = journey.SearchState(timeout_s=8.)
                state.enter(phase, 0.)
                self.assertIsNone(state.deadline)
                self.assertIsNone(state.tick(1000., ready=False))
                self.assertEqual(state.phase, phase)


class RelativeJ3Tests(unittest.TestCase):
    def feedback_robot(self, degrees):
        raw = np.radians(np.asarray(degrees, float))
        arm = NS(get_state=Mock(return_value=(raw, np.zeros_like(raw))))
        return NS(arm=arm), raw

    def test_reads_radians_and_changes_only_third_arm_joint_in_degrees(self):
        robot, raw = self.feedback_robot([1., -2., -10., -20., 5., 6., 10.])
        original = raw.copy()
        actual = journey.relative_j3_target(robot, -15.)
        np.testing.assert_allclose(actual, [1., -2., -25., -20., 5., 6.], atol=1e-10)
        np.testing.assert_array_equal(raw, original)
        robot.arm.get_state.assert_called_once()

    def test_each_retry_uses_actual_feedback_instead_of_previous_command(self):
        robot, _ = self.feedback_robot([0., 0., -10., -20., 0., 0.])
        first = journey.relative_j3_target(robot, -15.)
        np.testing.assert_allclose(first, [0., 0., -25., -20., 0., 0.])
        robot.arm.get_state.return_value = (np.radians([0., 0., -24., -20., 0., 0.]), np.zeros(6))
        second = journey.relative_j3_target(robot, -15.)
        self.assertAlmostEqual(second[2], -39.)
        self.assertEqual(robot.arm.get_state.call_count, 2)

    def test_rejects_target_outside_both_joint_limits(self):
        for degrees, delta in (([0., 0., -179., -20., 0., 0.], -15.),
                               ([0., 0., -1., -20., 0., 0.], 15.)):
            with self.subTest(degrees=degrees, delta=delta):
                robot, _ = self.feedback_robot(degrees)
                with self.assertRaises((ValueError, RuntimeError)):
                    journey.relative_j3_target(robot, delta)

    def test_explicit_joint_bounds_are_honored_without_clamping_target(self):
        robot, _ = self.feedback_robot([0., 0., -20., -20., 0., 0.])
        actual = journey.relative_j3_target(robot, -5., j3_bounds=(-.6, -.1))
        self.assertAlmostEqual(actual[2], -25.)
        with self.assertRaises((ValueError, RuntimeError)):
            journey.relative_j3_target(robot, -30., j3_bounds=(-.6, -.1))

    def test_invalid_feedback_never_produces_a_target(self):
        for values in ([0., 0., float('nan'), 0., 0., 0.], [0., 0.],
                       [0., 0., float('inf'), 0., 0., 0.]):
            with self.subTest(values=values):
                robot, _ = self.feedback_robot(values)
                with self.assertRaises((ValueError, RuntimeError)):
                    journey.relative_j3_target(robot, -15.)

    def test_nonfinite_delta_cannot_reach_a_joint_target(self):
        robot, _ = self.feedback_robot([0., 0., -20., -20., 0., 0.])
        for delta in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(delta=delta), self.assertRaises((ValueError, RuntimeError)):
                journey.relative_j3_target(robot, delta)


class SharedDisplayOCRTests(unittest.TestCase):
    def test_display_reader_reuses_existing_recognizer_and_verified_backend(self):
        engine, input_type = object(), object()
        info = dict(actual_backend='FAKE_TEST', actual_device='test',
                    recognition_only=True, enabled_stages=['rec'])
        base = NS(engine=engine, input_type=input_type, backend_info=info)
        detector = NS(ocr=NS(retry_backend=base))
        with patch.object(display, 'build_ocr', side_effect=AssertionError('new model forbidden')) as build:
            reader = journey.shared_display_ocr(detector)
        self.assertIsInstance(reader, display.DisplayOCR)
        self.assertIs(reader.engine, engine)
        self.assertIs(reader.input_type, input_type)
        self.assertEqual(reader.info, info)
        self.assertEqual(reader.calls, 0)
        build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
