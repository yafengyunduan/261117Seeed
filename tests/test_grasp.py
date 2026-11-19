"""Execution delegates to the original main; previews never call it."""
from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from elevator import grasp,app

class GraspTests(unittest.TestCase):
    def test_startup_selects_staged_motion_and_zero_restores_original_ready_move(self):
        ready=app.load_workflow(app.ROOT/'config/261105_my_elevator.yaml')['initial_poses']['grasp_ready']
        ready=dict(ready,joint4_stage_duration_s=1.5)
        cfg={'robot':{'ready_pose':ready}}
        robot=SimpleNamespace(move_ready_in_stages=Mock())
        module=SimpleNamespace(_move_ready=Mock())
        controller=object()
        cancelled=Mock(return_value=False)
        with patch.object(grasp,'original_motion_adapter',return_value=(module,controller)):
            self.assertIs(grasp.enter_ready_pose(robot,cfg,cancelled),module)
            module._move_ready.assert_not_called()
            args=robot.move_ready_in_stages.call_args.args
            self.assertEqual(args[0][:3,3].tolist(),[ready['x'],ready['y'],ready['z']])
            self.assertEqual(args[1:],(ready['duration'],1.5,cancelled))
            ready['joint4_stage_duration_s']=0.
            self.assertIs(grasp.enter_ready_pose(robot,cfg,cancelled),module)
            module._move_ready.assert_called_once_with(controller,ready)
        robot.move_ready_in_stages.assert_called_once()

    def test_execute_calls_original_main_with_same_config(self):
        cfg={"robot":{"ready_pose":{"x":.3}}};original=SimpleNamespace(main=Mock(return_value=0))
        args=Namespace(execute=True,config=Path("config/default.yaml"))
        argv=sys.argv
        with patch.object(grasp,"original_main",return_value=original):
            self.assertEqual(grasp.run(cfg,args),0)
        original.main.assert_called_once()
        self.assertIs(original.load_config("any"),cfg)
        self.assertIs(sys.argv,argv)

    def test_original_exit_or_home_exception_always_runs_final_disconnect(self):
        for error in (None,RuntimeError("home failed"),KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                arm=SimpleNamespace(disconnect=Mock(),_application_shutdown_error="")
                original=SimpleNamespace()
                def main():
                    original.RebotArm()
                    if error is not None:raise error
                    return 0
                original.main=main
                with patch.object(grasp,"original_main",return_value=original),patch.object(grasp,"sdk_types",return_value=(lambda:arm,object)):
                    if error is None:
                        self.assertEqual(grasp.run({},Namespace(execute=True,config="unused")),0)
                    else:
                        with self.assertRaises(type(error)):
                            grasp.run({},Namespace(execute=True,config="unused"))
                arm.disconnect.assert_called_once()

    def test_original_swallowed_disconnect_failure_does_not_report_success(self):
        arm=SimpleNamespace(disconnect=Mock(),_application_shutdown_error="joint2 disable unconfirmed")
        original=SimpleNamespace()
        def main():
            original.RebotArm()
            return 0  # Original main caught and printed the shutdown error.
        original.main=main
        with patch.object(grasp,"original_main",return_value=original),patch.object(grasp,"sdk_types",return_value=(lambda:arm,object)),self.assertRaisesRegex(RuntimeError,"joint2"):
            grasp.run({},Namespace(execute=True,config="unused"))

    def test_dry_preview_never_loads_original_robot_entry(self):
        with patch.object(grasp,"original_main",side_effect=AssertionError("SDK must not load")),patch.object(grasp,"preview",return_value=0) as p:
            self.assertEqual(grasp.run({},Namespace(execute=False)),0)
            p.assert_called_once()

    def test_mode_selection_dispatches_grasp_without_elevator_config(self):
        with patch("builtins.input",return_value="1"),patch.object(grasp,"run",return_value=0) as run,patch.object(app,"load_workflow",side_effect=AssertionError("not elevator")):
            self.assertEqual(app.main(["--execute"]),0)
        self.assertTrue(run.call_args.args[1].execute)
