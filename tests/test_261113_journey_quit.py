"""Q requests orderly home; emergency exits and failed stops never request it.

All camera, model, robot and SDK operations are fake. No hardware is opened.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from elevator import journey_261113 as j
from tests import test_261113_journey_runtime as runtime


class QuitCallbackTests(unittest.TestCase):
    def owner(self):
        cfg, options, settings, args = runtime.JourneyRuntimeTests().settings()
        with patch.object(j.app, '_tracker', return_value=runtime.FakeTracker()):
            return j.Journey(cfg, options, settings, args)

    def poll(self, owner, key=-1, *, visible=1, terminal=None):
        stream = NS(isatty=lambda: terminal is not None,
                    readline=Mock(return_value='' if terminal is None else terminal+'\n'))
        with patch.object(j.cv2, 'waitKey', return_value=key), \
                patch.object(j.cv2, 'getWindowProperty', return_value=visible), \
                patch.object(j.sys, 'stdin', stream), \
                patch.object(j.select, 'select', return_value=([stream], [], [])):
            return owner.cancelled()

    def test_q_and_uppercase_q_set_home_intent_and_return_bool(self):
        for key in (ord('q'), ord('Q')):
            with self.subTest(key=key):
                owner = self.owner()
                self.assertFalse(owner.quit_home_requested)
                self.assertIs(self.poll(owner, key), True)
                self.assertTrue(owner.quit_home_requested)
                # Do not set home here: an SDK stop may subsequently fail.
                self.assertFalse(owner.home)

    def test_space_escape_and_closed_window_cancel_without_home(self):
        for key, visible in ((32, 1), (27, 1), (-1, 0)):
            with self.subTest(key=key, visible=visible):
                owner = self.owner()
                owner.quit_home_requested = True
                self.assertTrue(self.poll(owner, key, visible=visible))
                self.assertFalse(owner.quit_home_requested)
                self.assertFalse(owner.home)

    def test_terminal_q_requests_home_but_stop_is_emergency(self):
        for value, expected in (('q', True), ('Q', True), ('stop', False)):
            with self.subTest(value=value):
                owner = self.owner()
                self.assertTrue(self.poll(owner, terminal=value))
                self.assertEqual(owner.quit_home_requested, expected)

    def test_unrelated_keys_do_not_cancel_or_request_home(self):
        owner = self.owner()
        self.assertFalse(self.poll(owner, ord('u')))
        self.assertFalse(owner.quit_home_requested)
        self.assertFalse(self.poll(owner, terminal='3'))
        self.assertFalse(owner.quit_home_requested)

    def test_n_requests_cooperative_stage_skip_from_window_or_terminal(self):
        for phase in ('WAIT_CALL','CALL','FLOOR','WAIT_DISPLAY','DISPLAY'):
            for source in ('window','terminal'):
                with self.subTest(phase=phase,source=source):
                    owner=self.owner();owner.state.enter(phase,1.)
                    result=self.poll(owner,ord('n') if source=='window' else -1,
                                     terminal='N' if source=='terminal' else None)
                    self.assertTrue(result);self.assertTrue(owner.skip_requested)
                    self.assertFalse(owner.quit_home_requested);self.assertFalse(owner.home)

    def test_q_and_emergency_cancel_override_pending_skip(self):
        for key,home in ((ord('q'),True),(32,False),(27,False)):
            with self.subTest(key=key):
                owner=self.owner();owner.state.enter('CALL',1.);owner.skip_requested=True
                self.assertTrue(self.poll(owner,key))
                self.assertFalse(owner.skip_requested);self.assertEqual(owner.quit_home_requested,home)


class QuitRunTests(unittest.TestCase):
    def run_case(self, trigger='idle_q', fail_stop=False, ordinary_error=False, fail_cleanup=False):
        """Run real Journey.run with fake IO, including actual key polling."""
        clock, events = runtime.Clock(), []
        cfg, options, settings, args = runtime.JourneyRuntimeTests().settings()
        with ExitStack() as stack:
            stack.enter_context(patch.object(j.app, '_tracker', return_value=runtime.FakeTracker()))
            owner = j.Journey(cfg, options, settings, args)
            camera = runtime.FakeCamera(clock, events)
            robot = runtime.FakeRobot(events, fail_cleanup)
            worker = runtime.FakeWorker(owner, events, 'complete')
            stack.enter_context(patch('drivers.camera.make_camera', return_value=camera))
            stack.enter_context(patch('utils.camera_utils.load_hand_eye', return_value=(np.eye(4), 'eye_in_hand')))
            stack.enter_context(patch.object(j.motion, 'Robot', return_value=robot))
            stack.enter_context(patch.object(j, 'JourneyWorker', return_value=worker))
            stack.enter_context(patch.object(j.time, 'perf_counter', side_effect=clock))
            for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch.object(j.cv2, name))
            stack.enter_context(patch.object(j.cv2, 'getWindowProperty', return_value=1))
            interactive, moving, sent = [False], [False], set()

            def draw(actual, snap):
                actual.display = snap.color
                if actual.state.phase == 'GRASP':
                    interactive[0] = True
                events.append(('draw', actual.state.phase))

            def adapter(actual_robot, cancelled):
                def ready(controller, ready_cfg):
                    if cancelled():
                        raise j.motion.MotionCancelled('startup cancellation')
                    events.append(('ready',))
                return NS(_move_ready=ready), object()

            def key(delay=1):
                if trigger == 'startup_q':
                    return ord('q')
                if moving[0]:
                    return 32 if trigger == 'motion_space' else ord('q')
                if owner.state.phase == 'GRASP' and interactive[0]:
                    if trigger == 'idle_q': return ord('q')
                    if trigger == 'idle_space': return 32
                    if trigger == 'idle_escape': return 27
                    if trigger == 'terminal_q': return -1
                    if 'N' not in sent:
                        sent.add('N'); return ord('n')
                if owner.state.phase == 'WAIT_CALL' and 'CALL_C' not in sent:
                    sent.add('CALL_C'); return ord('c')
                if owner.state.phase == 'CALL' and owner.ready(clock.now) and 'U' not in sent:
                    sent.add('U'); return ord('u')
                return -1

            def stop():
                events.append(('stop',))
                if fail_stop:
                    raise RuntimeError('synthetic stop failure')

            def move(target, duration, cancelled):
                events.append(('motion_begin',))
                moving[0] = True
                try:
                    if cancelled():
                        stop()
                        if ordinary_error:
                            raise RuntimeError('ordinary error after Q')
                        raise j.motion.MotionCancelled('synthetic SDK cancellation')
                    raise AssertionError('test must cancel at first motion callback')
                finally:
                    moving[0] = False

            robot.move, robot.stop = move, stop
            stream = NS(isatty=lambda: trigger == 'terminal_q', readline=lambda: 'q\n')
            def stdin_ready(*_):
                return ([stream], [], []) if interactive[0] else ([], [], [])
            stack.enter_context(patch.object(j.sys, 'stdin', stream))
            stack.enter_context(patch.object(j.select, 'select', side_effect=stdin_ready))
            stack.enter_context(patch.object(j.Journey, 'draw', draw))
            stack.enter_context(patch.object(j.grasp, 'original_motion_adapter', side_effect=adapter))
            stack.enter_context(patch.object(j.cv2, 'waitKey', side_effect=key))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            try:
                result = owner.run()
            except Exception as exc:
                result = exc
            return result, events, owner

    def test_main_loop_q_and_terminal_q_home_and_exit_normally(self):
        for trigger in ('idle_q', 'terminal_q'):
            with self.subTest(trigger=trigger):
                result, events, owner = self.run_case(trigger)
                self.assertEqual(result, 0)
                self.assertTrue(owner.home)
                self.assertIn(('robot_finish', True), events)
                self.assertNotIn(('motion_begin',), events)
                self.assertIn(('camera_close',), events)
                self.assertIn(('worker_close',), events)

    def test_q_from_startup_callback_homes_and_releases_resources(self):
        result, events, owner = self.run_case('startup_q')
        self.assertEqual(result, 0)
        self.assertIn(('robot_finish', True), events)
        self.assertNotIn(('ready',), events)
        self.assertIn(('camera_close',), events)
        self.assertIn(('worker_close',), events)

    def test_motion_q_stops_before_orderly_home(self):
        result, events, owner = self.run_case('motion_q')
        self.assertEqual(result, 0)
        self.assertIn(('motion_begin',), events)
        self.assertLess(events.index(('stop',)), events.index(('robot_finish', True)))
        self.assertTrue(owner.quit_home_requested)

    def test_space_and_escape_never_home(self):
        for trigger in ('idle_space', 'idle_escape', 'motion_space'):
            with self.subTest(trigger=trigger):
                result, events, owner = self.run_case(trigger)
                self.assertEqual(result, 130)
                self.assertFalse(owner.home)
                self.assertIn(('robot_finish', False), events)
                self.assertNotIn(('robot_finish', True), events)

    def test_failed_stop_after_q_never_requests_home(self):
        result, events, owner = self.run_case('motion_q', fail_stop=True)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('synthetic stop failure', str(result))
        self.assertTrue(owner.quit_home_requested)
        self.assertFalse(owner.home)
        self.assertIn(('robot_finish', False), events)
        self.assertNotIn(('robot_finish', True), events)
        self.assertIn(('camera_close',), events)
        self.assertIn(('worker_close',), events)

    def test_other_failure_after_q_does_not_override_failure_path_to_home(self):
        result, events, owner = self.run_case('motion_q', ordinary_error=True)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('ordinary error after Q', str(result))
        self.assertFalse(owner.home)
        self.assertIn(('robot_finish', False), events)

    def test_home_cleanup_failure_is_reported_and_other_resources_close(self):
        result, events, owner = self.run_case('motion_q', fail_cleanup=True)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('synthetic home/hold failure', str(result))
        self.assertIn(('robot_finish', True), events)
        self.assertIn(('camera_close',), events)
        self.assertIn(('worker_close',), events)


if __name__ == '__main__':
    unittest.main()
