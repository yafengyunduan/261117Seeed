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

RETRY_DELTA = (-15., -25.)


class SearchStateTests(unittest.TestCase):
    def test_each_stage_uses_its_own_configured_timeout_after_every_view_change(self):
        state=journey.SearchState({'call':5.25,'floor':9.,'display':13.5})
        state.enter('WAIT_CALL',0.)
        self.assertIsNone(state.tick(100.,ready=False))
        for phase,seconds in (('CALL',5.25),('FLOOR',9.),('DISPLAY',13.5)):
            with self.subTest(phase=phase):
                state.enter(phase,100.)
                self.assertEqual(state.deadline,100.+seconds)
                for _ in range(2):
                    self.assertIsNone(state.tick(state.deadline-.01,ready=False))
                    now=state.deadline
                    self.assertEqual(state.tick(now,ready=False),RETRY_DELTA)
                    self.assertIsNone(state.tick(now+120.,ready=False))
                    state.retry_complete(now+3.)
                    self.assertEqual(state.deadline,now+3.+seconds)
                self.assertEqual(state.tick(state.deadline,ready=False),'HOME')

    def test_default_search_window_is_ten_seconds_for_each_stage(self):
        for phase in ('CALL', 'FLOOR', 'DISPLAY'):
            with self.subTest(phase=phase):
                state=journey.SearchState()
                self.assertEqual(state.timeout_s,10.)
                state.enter(phase,100.)
                for now in (110.,125.):
                    self.assertIsNone(state.tick(now-.01,ready=False))
                    self.assertEqual(state.tick(now,ready=False),RETRY_DELTA)
                    state.retry_complete(now+5.)
                self.assertIsNone(state.tick(139.99,ready=False))
                self.assertEqual(state.tick(140.,ready=False),'HOME')

    def test_start_requires_explicit_stage_and_grasp_has_no_search_timeout(self):
        state = journey.SearchState(timeout_s=8.)
        self.assertEqual(state.phase, 'GRASP')
        self.assertIsNone(state.tick(1000., ready=False))
        self.assertEqual(state.phase, 'GRASP')

    def test_each_phase_allows_exactly_two_j2_and_j3_retries(self):
        for phase in ('CALL', 'FLOOR', 'DISPLAY'):
            with self.subTest(phase=phase):
                state = journey.SearchState(timeout_s=8.)
                state.enter(phase, 100.)
                self.assertEqual(state.retry_index, 0)
                self.assertEqual(state.deadline, 108.)
                self.assertIsNone(state.tick(107.99, ready=False))
                self.assertEqual(state.tick(108., ready=False), RETRY_DELTA)
                self.assertEqual(state.retry_index, 1)
                self.assertIsNone(state.deadline)
                # The motion itself consumes five seconds; all eight seconds
                # for the next image search begin after the motion completes.
                state.retry_complete(113.)
                self.assertEqual(state.deadline, 121.)
                self.assertIsNone(state.tick(120.99, ready=False))
                self.assertEqual(state.tick(121., ready=False), RETRY_DELTA)
                self.assertEqual(state.retry_index, 2)
                state.retry_complete(125.)
                self.assertIsNone(state.tick(132.99, ready=False))
                self.assertEqual(state.tick(133., ready=False), 'HOME')
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
        self.assertEqual(state.tick(509., ready=False), RETRY_DELTA)

    def test_recovered_panel_does_not_erase_retry_budget_in_the_same_stage(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('FLOOR', 0.)
        self.assertEqual(state.tick(8., ready=False), RETRY_DELTA)
        state.retry_complete(12.)
        self.assertIsNone(state.tick(13., ready=True))
        self.assertEqual(state.retry_index, 1)
        self.assertIsNone(state.tick(50., ready=False))
        self.assertEqual(state.tick(58., ready=False), RETRY_DELTA)
        self.assertEqual(state.retry_index, 2)

    def test_new_stage_resets_retry_counter_and_has_its_own_full_window(self):
        state = journey.SearchState(timeout_s=8.)
        state.enter('CALL', 0.)
        self.assertEqual(state.tick(8., ready=False), RETRY_DELTA)
        state.retry_complete(12.)
        self.assertEqual(state.tick(20., ready=False), RETRY_DELTA)
        state.enter('FLOOR', 100.)
        self.assertEqual(state.retry_index, 0)
        self.assertEqual(state.deadline, 108.)
        self.assertEqual(state.tick(108., ready=False), RETRY_DELTA)

    def test_manual_waits_grasp_and_done_do_not_trigger_search_joint_moves(self):
        for phase in ('GRASP', 'WAIT_CALL', 'BOARD', 'DONE'):
            with self.subTest(phase=phase):
                state = journey.SearchState(timeout_s=8.)
                state.enter(phase, 0.)
                self.assertIsNone(state.deadline)
                self.assertIsNone(state.tick(1000., ready=False))
                self.assertEqual(state.phase, phase)


class RelativeHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=journey.motion.configured_arm_position_limits({})

    def test_changes_only_second_and_third_arm_joints_using_degree_feedback(self):
        raw=np.array([1.,-2.,-10.,-20.,5.,6.])
        original=raw.copy()
        actual=journey.relative_head_target(raw,RETRY_DELTA,self.limits)
        np.testing.assert_allclose(actual, [1., -17., -35., -20., 5., 6.], atol=1e-10)
        np.testing.assert_array_equal(raw, original)

    def test_each_retry_uses_actual_feedback_instead_of_previous_command(self):
        first=journey.relative_head_target([0.,0.,-10.,-20.,0.,0.],RETRY_DELTA,self.limits)
        np.testing.assert_allclose(first, [0., -15., -35., -20., 0., 0.])
        second=journey.relative_head_target([0.,-13.,-34.,-20.,0.,0.],RETRY_DELTA,self.limits,last_goal_deg=first)
        np.testing.assert_allclose(second, [0.,-28.,-59.,-20.,0.,0.])

    def test_shortens_both_axes_and_does_not_repeat_an_already_searched_limit(self):
        lower,upper=np.degrees(self.limits[1:3]).T
        for delta,bound in ((RETRY_DELTA,lower),((15.,25.),upper)):
            with self.subTest(delta=delta):
                joints=np.array([0.,*(bound-np.sign(delta)),-20.,0.,0.])
                goal=journey.relative_head_target(joints,delta,self.limits)
                np.testing.assert_allclose(goal[1:3],bound)
                self.assertIsNone(journey.relative_head_target(goal,delta,self.limits))
                self.assertIsNone(journey.relative_head_target(joints,delta,self.limits,last_goal_deg=goal))

    def test_axis_with_remaining_range_can_move_when_other_axis_is_at_limit(self):
        for limited_axis in (1,2):
            with self.subTest(limited_axis=limited_axis):
                joints=np.array([0.,-30.,-60.,-20.,0.,0.])
                expected=joints.copy()
                expected[1:3]+=RETRY_DELTA
                limit=np.degrees(self.limits[limited_axis,0])
                joints[limited_axis]=expected[limited_axis]=limit
                goal=journey.relative_head_target(joints,RETRY_DELTA,self.limits,last_goal_deg=joints)
                np.testing.assert_allclose(goal,expected)

    def test_uses_active_model_limits_and_rejects_out_of_range_held_axes(self):
        limits=self.limits.copy();limits[2]=[-.6,-.1]
        actual=journey.relative_head_target([0.,0.,-20.,-20.,0.,0.],(0.,-5.),limits)
        self.assertAlmostEqual(actual[2], -25.)
        self.assertAlmostEqual(journey.relative_head_target(actual,(0.,-30.),limits)[2],np.degrees(-.6))
        actual[3]=95.
        with self.assertRaisesRegex(journey.motion.HeadJointLimitError,'J4'):
            journey.relative_head_target(actual,(0.,-5.),limits)

    def test_invalid_feedback_never_produces_a_target(self):
        for values in ([0., 0., float('nan'), 0., 0., 0.], [0., 0.],
                       [0., 0., float('inf'), 0., 0., 0.]):
            with self.subTest(values=values):
                with self.assertRaises((ValueError, RuntimeError)):
                    journey.relative_head_target(values,RETRY_DELTA,self.limits)

    def test_nonfinite_delta_cannot_reach_a_joint_target(self):
        for delta in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(delta=delta), self.assertRaises((ValueError, RuntimeError)):
                journey.relative_head_target([0.,0.,-20.,-20.,0.,0.],(-15.,delta),self.limits)


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
