"""SDK adapter and pure local URDF planning checks. No device access."""
from types import SimpleNamespace as NS
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np
from elevator import motion,geometry
from utils.transforms import pose6d_to_mat4


class PositionOnlyPlanningTests(unittest.TestCase):
    """Real SDK IK/URDF; synthetic state/sender, never connect or enable."""
    @classmethod
    def setUpClass(cls):
        motion.sdk_types()
        import pinocchio as pin
        from reBotArm_control_py.kinematics import load_robot_model,get_end_effector_frame_id,pad_q_for_model,compute_fk
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams
        from reBotArm_control_py.trajectory import TrajPlanParams,TrajProfile,plan_cartesian_geodesic_trajectory
        cls.pin=pin;cls.model=load_robot_model();cls.frame=get_end_effector_frame_id(cls.model)
        cls.pad=staticmethod(pad_q_for_model);cls.fk=staticmethod(compute_fk)
        cls.IKParams=IKParams;cls.TrajPlanParams=TrajPlanParams;cls.profile=TrajProfile.MIN_JERK
        cls.cartesian=staticmethod(plan_cartesian_geodesic_trajectory)

    def controller(self):
        return NS(_model=self.model,_end_frame_id=self.frame,_n=6,
            _traj_params=self.TrajPlanParams(dt=.1,profile=self.profile),
            _ik_solver_params=self.IKParams(max_iter=200,tolerance=1e-4,step_size=.5,damping=1e-6))

    def test_logged_high_target_approach_surface_retreat_are_reachable_without_leveling(self):
        c=self.controller();q=np.radians([0,0,-10,-20,0,0])
        surface=np.array([.4729605675918177,-.028825335055700858,.7339024156014431])
        for name,point,duration in (('approach',surface-[.1,0,0],3.),('surface',surface,1.5),('retreat',surface-[.1,0,0],2.)):
            with self.subTest(segment=name):
                start=self.fk(self.model,self.pad(self.model,q,6))[2]
                plan=motion.plan_position_trajectory(c,q,point,duration)
                self.assertLess(np.linalg.norm(plan['end_pose'][:3,3]-point),1.01e-4)
                self.assertGreater(abs(plan['report']['end_rpy_deg'][1]),10.)
                knots=np.asarray(plan['points'])
                self.assertTrue(np.all(knots>=self.model.lowerPositionLimit[:6]-1e-9))
                self.assertTrue(np.all(knots<=self.model.upperPositionLimit[:6]+1e-9))
                dense=motion.densify_joint_samples(knots,duration,.01)
                ideal=self.cartesian(self.pin.SE3(start),self.pin.SE3(start[:3,:3],point),duration,
                    self.TrajPlanParams(dt=.01,profile=self.profile)).trajectory.points()
                errors=[np.linalg.norm(self.fk(self.model,self.pad(self.model,joints,6))[0]-target.pose.translation)
                        for joints,target in zip(dense,ideal)]
                self.assertLess(max(errors),.005)
                q=knots[-1]

    def test_nonzero_tip_offset_is_solved_without_mutating_sdk_model(self):
        c=self.controller();frames=self.model.nframes
        point=np.array([.4,-.1,.45]);offset=np.array([.02,.005,-.01])
        plan=motion.plan_position_trajectory(c,np.radians([0,0,-10,-20,0,0]),point,3.,offset)
        pose=plan['end_pose'];tip=pose[:3,3]+pose[:3,:3]@offset
        self.assertLess(np.linalg.norm(tip-point),1.01e-4)
        self.assertEqual(self.model.nframes,frames)
        self.assertGreater(np.linalg.norm(pose[:3,3]-point),.01)

    def test_positive_measured_j2_is_preserved_at_start_without_expanding_limits(self):
        c=self.controller();upper=self.model.upperPositionLimit.copy()
        measured=np.radians([0,1.8,-10,-20,0,0])
        plan=motion.plan_position_trajectory(c,measured,[.373,-.029,.734],3.)
        np.testing.assert_allclose(plan['points'][0],measured)
        self.assertTrue(np.all(np.asarray(plan['points'][1:])<=upper[:6]+1e-9))
        np.testing.assert_array_equal(self.model.upperPositionLimit,upper)

    def test_unreachable_position_and_planning_cancel_produce_no_sender(self):
        c=self.controller();q=np.radians([0,0,-10,-20,0,0])
        with self.assertRaises(motion.TrajectoryPlanningError) as caught:
            motion.plan_position_trajectory(c,q,[2.,0.,2.],3.)
        self.assertTrue(caught.exception.position_only);self.assertFalse(caught.exception.trajectory_sent)
        with self.assertRaises(motion.MotionCancelled):
            motion.plan_position_trajectory(c,q,[.4,0.,.4],3.,cancelled=lambda:True)
        self.assertFalse(hasattr(c,'_send_thread'))

    def executing_robot(self):
        arm,c,d=stack();planning=self.controller()
        c.__dict__.update(planning.__dict__);c._dt=.01;c._moving=False
        c._q_target[:]=np.radians([0,0,-10,-20,0,0])
        d.T=self.fk(self.model,self.pad(self.model,c._q_target,6))[2]
        def send(duration):
            c._q_target[:]=c._traj[-1]
            d.T=self.fk(self.model,self.pad(self.model,c._q_target,6))[2]
        c._send_loop=send
        motion.configure_trajectory_sampling(c,.1)
        robot=motion.Robot({},factory=lambda cfg:(arm,c,d));robot.start()
        def thread(*,target,args,daemon):
            return NS(start=lambda:target(*args),is_alive=lambda:False,join=Mock())
        return robot,arm,c,d,thread

    def test_position_execution_reuses_sdk_sender_density_without_full_pose_ik(self):
        robot,arm,c,d,thread=self.executing_robot()
        point=np.array([.373,-.029,.734])
        with patch.object(motion.threading,'Thread',side_effect=thread) as sender:
            pose=robot.move_position(point,3.)
        sender.assert_called_once();c.move_to_traj.assert_not_called()
        self.assertEqual(c._application_trajectory_report['ik_points'],31)
        self.assertEqual(c._application_trajectory_report['send_points'],301)
        self.assertLess(np.linalg.norm(pose[:3,3]-point),1.01e-4)
        arm.disconnect.assert_not_called()

    def test_position_execution_uses_feedback_fk_and_waits_for_a_quiet_window(self):
        robot,arm,c,d,thread=self.executing_robot()
        arm.control_loop_active=True
        c._arm_control_mode='posvel'
        c._arm_group=NS(_pv_vlim=np.array([5.,5.,5.,3.,3.,3.]))
        d._model,d._n=self.model,6
        d._pad_q_for_model,d._compute_fk=self.pad,self.fk
        point=np.array([.373,-.029,.734])
        clock=[0.]
        def sleep(seconds):clock[0]+=seconds
        def read():
            clock[0]+=.06
            return c._q_target.copy(),np.zeros(6)
        with patch.object(motion.threading,'Thread',side_effect=thread), \
                patch.object(motion.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(motion.time,'sleep',side_effect=sleep), \
                patch.object(robot,'_arm_feedback',side_effect=read) as feedback:
            robot.move_position(point,3.,settle=motion.SettledArrival('approach',.15,
                brake_tail_s=.7,stationary_vlim_rad_s=.05))
        self.assertGreaterEqual(feedback.call_count,4)
        np.testing.assert_allclose(robot._pose_from_joints(c._q_target)[:3,3],point,atol=.0001)
        np.testing.assert_allclose(c._vlim_override,.05)

    def test_ready_move_sends_j4_only_after_the_other_axes_settle(self):
        target=pose6d_to_mat4(.30,0.,.30,0.,.7,0.)
        for stalled in (False,True):
            with self.subTest(stalled=stalled):
                robot,arm,c,driver,thread=self.executing_robot()
                c._q_target.fill(0.)
                arm.control_loop_active=True
                c._arm_control_mode='posvel'
                c._arm_group=NS(_pv_vlim=np.array([5.,5.,5.,3.,3.,3.]))
                driver._model,driver._n=self.model,6
                driver._pad_q_for_model,driver._compute_fk=self.pad,self.fk
                clock=[0.]
                sent=[]
                native=c._application_original_sender
                def send(duration):
                    sent.append((clock[0],duration,np.asarray(c._traj).copy()))
                    native(duration)
                c._application_original_sender=send
                def sleep(seconds):clock[0]+=seconds
                def feedback():
                    clock[0]+=.06
                    actual=c._q_target.copy()
                    if stalled and sent:actual[2]+=.03
                    return actual,np.zeros(6)
                with patch.object(motion.threading,'Thread',side_effect=thread), \
                        patch.object(motion.time,'monotonic',side_effect=lambda:clock[0]), \
                        patch.object(motion.time,'sleep',side_effect=sleep), \
                        patch.object(robot,'_arm_feedback',side_effect=feedback):
                    if stalled:
                        with self.assertRaisesRegex(TimeoutError,'grasp_ready_1'):
                            robot.move_ready_in_stages(target,3.,1.5)
                    else:
                        robot.move_ready_in_stages(target,3.,1.5)
                self.assertEqual(len(sent),1 if stalled else 2)
                first=sent[0][2]
                self.assertTrue(np.all(first[:,3]==0.))
                self.assertGreater(abs(first[-1,2]),.5)
                if not stalled:
                    second=sent[1][2]
                    others=[0,1,2,4,5]
                    np.testing.assert_array_equal(second[0],first[-1])
                    np.testing.assert_array_equal(second[:,others],np.broadcast_to(second[0,others],(len(second),5)))
                    self.assertGreater(abs(second[-1,3]),.5)
                    self.assertGreaterEqual(sent[1][0]-sent[0][0],.25)
                    np.testing.assert_allclose(robot._pose_from_joints(second[-1]),target,atol=.0001)

    def test_failed_position_plan_never_starts_sender_and_leaves_recovery_to_journey(self):
        robot,arm,c,d,thread=self.executing_robot()
        c._application_standoff_hold=True
        c._vlim_override=np.full(6,.05)
        held=c._q_target.copy()
        with patch.object(motion.threading,'Thread',side_effect=thread) as sender, \
                self.assertRaises(motion.TrajectoryPlanningError):robot.move_position([2.,0.,2.],3.)
        sender.assert_not_called();self.assertTrue(robot.started)
        self.assertTrue(c._application_standoff_hold)
        np.testing.assert_array_equal(c._q_target,held)
        np.testing.assert_allclose(c._vlim_override,.05)
        arm.disconnect.assert_not_called();c.safe_home.assert_not_called()

    def test_invalid_position_does_not_change_targets_or_call_planner(self):
        robot,arm,c,d,thread=self.executing_robot();before=c._q_target.copy()
        with patch.object(motion,'plan_position_trajectory') as planner:
            with self.assertRaises(ValueError):robot.move_position([float('nan'),0.,.4])
        planner.assert_not_called();np.testing.assert_array_equal(c._q_target,before)

    def test_position_arrival_checks_tip_xyz_without_requiring_planned_orientation(self):
        robot,arm,c,d,thread=self.executing_robot()
        point=np.array([.3,0.,.4]);d.T=pose6d_to_mat4(*point,.4,-.7,.2)
        intended=np.eye(4);intended[:3,3]=point
        now=motion.time.monotonic()
        robot._wait_move(intended,3.,lambda:False,now,now,tip_target=point,tip_offset=np.zeros(3))


class GripperZeroTests(unittest.TestCase):
    """Native bus mocked: calibration writes must be explicit and ID-7-only."""
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("gripper_zero",Path(__file__).resolve().parents[1]/
                                                  "scripts/261110_my_gripper_zero.py")
        cls.tool=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report=Path(self.tmp.name)/"audit.json"
        self.state=NS(can_id=7,arbitration_id=23,status_code=1,pos=np.radians(9.87),vel=0.,torq=0.)
        self.motor=Mock(spec=["get_register_u32","get_state","request_feedback","disable",
                              "set_zero_position","close"])
        self.motor.get_register_u32.side_effect=lambda rid,timeout:{7:23,8:7}[rid]
        self.motor.get_state.side_effect=lambda:self.state
        self.motor.disable.side_effect=lambda:setattr(self.state,"status_code",0)
        self.motor.set_zero_position.side_effect=lambda:setattr(self.state,"pos",0.)
        self.bus=Mock(spec=["add_damiao_motor","poll_feedback_once","close_bus","close"])
        self.bus.add_damiao_motor.return_value=self.motor
        self.factory=Mock(return_value=self.bus)
        self.confirm=Mock(return_value="CLOSED")

    def run_tool(self,write=False):
        with patch.object(self.tool.time,"sleep"),contextlib.redirect_stdout(io.StringIO()):
            return self.tool.run("/dev/test-only",write,self.report,
                                 controller_factory=self.factory,confirm=self.confirm)

    def audit(self):
        return json.loads(self.report.read_text())

    def assert_closed(self):
        self.bus.close_bus.assert_called_once()
        self.motor.close.assert_called_once()
        self.bus.close.assert_called_once()

    def test_default_is_query_only_and_registers_only_gripper(self):
        self.assertEqual(self.run_tool(),0)
        self.bus.add_damiao_motor.assert_called_once_with(7,23,"4310")
        self.motor.disable.assert_not_called();self.motor.set_zero_position.assert_not_called()
        self.confirm.assert_not_called();self.assert_closed()
        self.assertFalse(self.audit()["zero_command_attempted"])

    def test_zero_requires_confirmation_and_checks_raw_zero_readback(self):
        self.assertEqual(self.run_tool(True),0)
        self.motor.disable.assert_called_once();self.motor.set_zero_position.assert_called_once()
        self.confirm.assert_called_once();self.assert_closed()
        report=self.audit()
        self.assertEqual(report["result"],"ZERO_READBACK_OK")
        self.assertAlmostEqual(report["before"]["position_deg"],9.87)
        self.assertEqual(report["after"]["position_deg"],0.)
        self.assertFalse(report["packet_freshness_verified"])

    def test_cancel_never_writes_zero_and_leaves_gripper_disabled(self):
        self.confirm.return_value="cancel"
        self.assertEqual(self.run_tool(True),1)
        self.motor.set_zero_position.assert_not_called();self.assert_closed()
        self.assertEqual(self.state.status_code,0)
        self.assertEqual(self.audit()["result"],"CANCELLED_NO_ZERO_WRITE")

    def test_id_mismatch_fails_before_disable_or_zero(self):
        self.motor.get_register_u32.return_value=1
        self.motor.get_register_u32.side_effect=None
        with self.assertRaisesRegex(RuntimeError,"ID不匹配"):self.run_tool(True)
        self.motor.disable.assert_not_called();self.motor.set_zero_position.assert_not_called()
        self.assert_closed()

    def test_no_feedback_is_not_treated_as_zero(self):
        self.motor.get_state.side_effect=lambda:None
        with self.assertRaisesRegex(RuntimeError,"无反馈"):self.run_tool(True)
        self.motor.set_zero_position.assert_not_called();self.assert_closed()

    def test_failed_disable_or_unstable_position_never_reaches_zero(self):
        self.motor.disable.side_effect=None
        with self.assertRaisesRegex(RuntimeError,"失能未确认"):self.run_tool(True)
        self.motor.set_zero_position.assert_not_called();self.confirm.assert_not_called()
        self.assert_closed()

    def test_moving_gripper_cannot_be_zeroed(self):
        self.state.vel=.1
        with self.assertRaisesRegex(RuntimeError,"静止"):self.run_tool(True)
        self.motor.set_zero_position.assert_not_called();self.assert_closed()

    def test_zero_write_failure_is_not_retried_and_records_uncertainty(self):
        self.motor.set_zero_position.side_effect=RuntimeError("write failed")
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaisesRegex(RuntimeError,"write failed"):
            self.run_tool(True)
        self.motor.set_zero_position.assert_called_once();self.assert_closed()
        self.assertTrue(self.audit()["zero_command_attempted"])

    def test_nonzero_readback_is_not_reported_as_success(self):
        self.motor.set_zero_position.side_effect=None
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaisesRegex(RuntimeError,"未确认零角"):
            self.run_tool(True)
        self.assertEqual(self.audit()["result"],"ERROR");self.assert_closed()

    def test_cleanup_failure_still_closes_other_handles(self):
        self.bus.close_bus.side_effect=RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError,"清理异常"):self.run_tool()
        self.assert_closed();self.assertIn("cleanup_errors",self.audit())

    def test_noninteractive_zero_cli_is_rejected_before_device_access(self):
        with patch.object(self.tool.sys,"stdin") as stdin, \
                patch.object(self.tool,"run",side_effect=AssertionError("device")), \
                contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            stdin.isatty.return_value=False
            self.tool.main(["--set-zero"])

def stack():
    c=NS(_q_target=np.zeros(6),_qd_target=np.zeros(6),_vlim_override=None,_stop_send=threading.Event(),
         _send_thread=None,_running=True,end=Mock(),safe_home=Mock(),
         _model=NS(lowerPositionLimit=np.array([-2.8,-3.14,-3.14,-1.87,-1.57,-3.14]),
                   upperPositionLimit=np.array([2.8,0.,0.,1.57,1.57,3.14])))
    arm=NS(get_state=lambda:(c._q_target.copy(),np.zeros(6),np.zeros(6)),disconnect=Mock())
    d=NS(start=Mock(),open_gripper=Mock(),_state_lock=threading.Lock(),_state="IDLE",_STATE_IDLE="IDLE",_gripper_motor=None,
         T=np.eye(4),_elevator_gripper_target=np.array([-.15]),_gripper_group=NS(send_pos_vel=Mock()))
    d._gripper_motor=NS(get_state=Mock(side_effect=lambda:NS(pos=float(d._elevator_gripper_target[0]),vel=0.,torq=.01)),
                        request_feedback=Mock(),send_mit=Mock())
    d.get_tcp_pose=lambda:d.T.copy()
    def move(*pose,duration):
        d.T=pose6d_to_mat4(*pose);return True
    c.move_to_traj=Mock(side_effect=move)
    return arm,c,d

class MotionTests(unittest.TestCase):
    def test_application_stack_installs_workflow_sampling_without_starting_sdk(self):
        motion.sdk_types()
        arm=NS()
        c=NS(_traj_params=NS(dt=.01),_dt=.01,_send_loop=Mock())
        with patch.object(motion,'sdk_types',return_value=(lambda:arm,Mock())), \
                patch('reBotArm_control_py.controllers.RebotArmEndPose',return_value=c):
            motion.make_stack({'trajectory':{'sample_interval_s':.1}})
        self.assertEqual(c._traj_params.dt,.1)
        self.assertEqual(c._dt,.01)
        c._application_original_sender.assert_not_called()

    def test_coarse_ik_preserves_original_sender_density_endpoint_and_cancel(self):
        # Real SDK sender with mocked sleep and arm (no SDK start/connect).
        motion.sdk_types()
        from reBotArm_control_py.controllers import RebotArmEndPose
        arm=NS(groups={'arm':NS(num_joints=6)},has_gripper=False,arm_control_mode='posvel')
        controller=RebotArmEndPose(arm)
        goal=np.array([.1,-.2,-.3,-.4,.2,.1])
        coarse=np.linspace(np.zeros(6),goal,31)
        controller._traj=list(coarse)
        motion.configure_trajectory_sampling(controller,.1)
        self.assertEqual(controller._dt,.01)
        with patch('reBotArm_control_py.controllers.rebotarm_endpose_controller.time.sleep') as sleep:
            controller._send_loop(3.)
        self.assertEqual(sleep.call_count,301)
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list),3.)
        np.testing.assert_allclose(controller._q_target,goal)
        np.testing.assert_allclose(np.array(controller._traj)[::10],coarse)
        self.assertEqual(controller._application_trajectory_report['ik_points'],31)
        controller._stop_send.set();controller._q_target[:]=0
        with patch('reBotArm_control_py.controllers.rebotarm_endpose_controller.time.sleep') as sleep:
            controller._send_loop(3.)
        sleep.assert_not_called();np.testing.assert_allclose(controller._q_target,0)

    def test_interpolation_has_no_overshoot_and_supports_non_integer_duration(self):
        q=np.array([[0.,-.2],[.4,-.1],[.1,-.4]])
        dense=motion.densify_joint_samples(q,.215,.01)
        self.assertEqual(len(dense),23)
        np.testing.assert_array_equal(dense[[0,-1]],q[[0,-1]])
        self.assertTrue(np.all(dense>=q.min(axis=0)))
        self.assertTrue(np.all(dense<=q.max(axis=0)))

    def test_approach_brakes_on_the_sdk_time_grid_and_resumes_normal_speed_for_next_move(self):
        motion.sdk_types()
        from reBotArm_control_py.controllers import RebotArmEndPose
        native=np.array([.10,5.,5.,3.,3.,3.])
        arm=NS(groups={'arm':NS(num_joints=6,_pv_vlim=native)},has_gripper=False,arm_control_mode='posvel')
        controller=RebotArmEndPose(arm)
        motion.configure_trajectory_sampling(controller,.1)
        progress=np.linspace(0.,1.,31)
        eased=10*progress**3-15*progress**4+6*progress**5
        coarse=np.outer(eased,[.05,-.2,-.3,-.8,.2,.1])
        controller._traj=list(coarse)
        controller._application_standoff_brake=motion.SettledArrival('approach',.15,True,.7)
        sent_positions,sent_caps,intervals=[],[],[]
        def sample(interval):
            sent_positions.append(controller._q_target.copy())
            cap=controller._vlim_override
            sent_caps.append(native.copy() if cap is None else cap.copy())
            intervals.append(interval)
        with patch.object(motion.time,'sleep',side_effect=sample):
            controller._send_loop(3.)
        dense=motion.densify_joint_samples(coarse,3.,.01)
        np.testing.assert_array_equal(sent_positions,dense)
        self.assertAlmostEqual(sum(intervals),3.)
        caps=np.array(sent_caps)
        np.testing.assert_array_equal(caps[0],native)
        self.assertTrue(np.all(np.diff(caps,axis=0)<=1e-12))
        required=np.abs(np.diff(dense,axis=0))/intervals[0]
        self.assertTrue(np.all(caps[1:]>=np.minimum(required,native)-1e-12))
        np.testing.assert_allclose(caps[-1],np.minimum(native,.15))
        np.testing.assert_array_equal(controller._vlim_override,np.minimum(native,.15))
        self.assertIsNone(controller._application_sender_error)
        self.assertIsNone(controller._application_standoff_brake)
        with patch.object(motion.time,'sleep'):
            controller._traj=[dense[-1],dense[0]]
            controller._send_loop(.3)
        self.assertIsNone(controller._vlim_override)

    def test_stop_during_approach_braking_does_not_send_the_remaining_targets(self):
        native=np.full(6,3.)
        controller=NS(_traj_params=NS(dt=.1),_dt=.01,_traj=[np.zeros(6),np.ones(6)*.06],
            _send_loop=Mock(),_q_target=np.zeros(6),_stop_send=threading.Event(),_moving=True,
            _vlim_override=None,_arm_group=NS(_pv_vlim=native))
        motion.configure_trajectory_sampling(controller,.1)
        controller._application_standoff_brake=motion.SettledArrival('approach',.15,True,.7)
        stopped_at=[]
        def stop_when_braking(_):
            if controller._vlim_override is not None and np.max(controller._vlim_override)<3.:
                stopped_at.append(controller._q_target.copy())
                controller._stop_send.set()
        with patch.object(motion.time,'sleep',side_effect=stop_when_braking):
            controller._send_loop(1.)
        self.assertEqual(len(stopped_at),1)
        np.testing.assert_array_equal(controller._q_target,stopped_at[0])
        self.assertLess(controller._q_target[0],.06)
        self.assertFalse(controller._moving)
        controller._application_original_sender.assert_not_called()

    def test_sampling_configuration_rejects_nonfinite_or_invalid_before_devices(self):
        from elevator import app
        for value in (float('nan'),0.,.2):
            with patch.object(app,'read_yaml',return_value={'trajectory':{'sample_interval_s':value}}), \
                    self.assertRaisesRegex(ValueError,'sample_interval_s'):
                app.load_workflow('unused')

    def robot(self):
        self.arm,self.c,self.d=stack()
        r=motion.Robot({},factory=lambda cfg:(self.arm,self.c,self.d));r.start()
        return r

    def test_seven_axis_head_sets_gripper_zero_without_opening_or_mit_commands(self):
        r=self.robot();r.head_up([0,0,-10,-20,0,0,0])
        self.d.start.assert_called_once()
        self.assertTrue(self.d._elevator_gripper_posvel)
        np.testing.assert_allclose(self.c._q_target,np.radians([0,0,-10,-20,0,0]))
        np.testing.assert_allclose(self.d._elevator_gripper_target,[0.])
        self.d.open_gripper.assert_not_called();self.d._gripper_motor.send_mit.assert_not_called()

    def test_six_axis_head_retains_gripper_angle_for_skip_option(self):
        r=self.robot();r.head_up([0,0,-10,-20,0,0])
        np.testing.assert_allclose(self.d._elevator_gripper_target,[-.15])

    def test_unchecked_head_waits_only_reference_time_without_arrival_reads_or_speed_jump(self):
        r=self.robot();clock=[0.]
        self.arm.get_state=Mock(return_value=(np.zeros(6),np.zeros(6),np.zeros(6)))
        self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=-.15,vel=.2,torq=.1)
        output=io.StringIO()
        def sleep(seconds):clock[0]+=seconds
        with patch.object(motion.time,"monotonic",side_effect=lambda:clock[0]), \
                patch.object(motion.time,"sleep",side_effect=sleep),contextlib.redirect_stdout(output):
            r.head_up([0,0,-10,-20,0,0,0],duration=3.,wait_for_arrival=False)
        self.assertAlmostEqual(clock[0],3.)
        self.arm.get_state.assert_called_once();self.assertEqual(self.d._gripper_motor.get_state.call_count,2)
        self.d._gripper_motor.request_feedback.assert_not_called()
        np.testing.assert_allclose(self.c._q_target,np.radians([0,0,-10,-20,0,0]))
        np.testing.assert_allclose(self.d._elevator_gripper_target,[0.])
        self.assertIsNotNone(self.c._vlim_override)
        self.assertIn("Head:UNVERIFIED",output.getvalue());self.assertNotIn("CLOSED_REFERENCE",output.getvalue())
        r.move(np.eye(4));self.assertIsNone(self.c._vlim_override)

    def test_unchecked_head_still_cancels_during_nominal_wait(self):
        r=self.robot();clock=[0.]
        self.arm.get_state=lambda:(np.zeros(6),np.zeros(6),np.zeros(6))
        def sleep(seconds):clock[0]+=seconds
        with patch.object(motion.time,"monotonic",side_effect=lambda:clock[0]), \
                patch.object(motion.time,"sleep",side_effect=sleep),self.assertRaises(motion.MotionCancelled):
            r.head_up([0,0,-10,-20,0,0,0],duration=3.,wait_for_arrival=False,cancelled=lambda:clock[0]>=.5)
        self.assertLess(clock[0],3.)
        np.testing.assert_allclose(self.c._q_target,np.zeros(6))
        self.assertIsNone(self.c._vlim_override);self.c.move_to_traj.assert_not_called()

    def test_missing_gripper_feedback_never_becomes_zero_or_starts_head(self):
        r=self.robot();self.d._gripper_motor.get_state.side_effect=lambda:None
        with self.assertRaisesRegex(RuntimeError,"第七轴缺少有效反馈"):
            r.head_up([0,0,-10,-20,0,0,0])
        np.testing.assert_allclose(self.c._q_target,np.zeros(6))
        np.testing.assert_allclose(self.d._elevator_gripper_target,[-.15])

    def test_arm_arrival_does_not_hide_a_gripper_left_open_or_still_moving(self):
        for pos,vel in ((-.15,0.),(0.,.2)):
            r=self.robot();self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=pos,vel=vel,torq=.1)
            clock=[0.]
            def tick():clock[0]+=.1;return clock[0]
            with patch.object(motion.time,"monotonic",side_effect=tick),patch.object(motion.time,"sleep"), \
                    self.assertRaisesRegex(TimeoutError,"抬头/夹爪未到位"):
                r.head_up([0,0,-10,-20,0,0,0],duration=.5)
            np.testing.assert_allclose(self.d._elevator_gripper_target,[pos])
            self.d._gripper_motor.send_mit.assert_not_called()

    def test_stop_holds_current_gripper_with_pos_vel_without_switching_to_mit(self):
        r=self.robot();self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=-.08,vel=0.,torq=0.)
        r.stop()
        np.testing.assert_allclose(self.d._gripper_group.send_pos_vel.call_args.args[0],[-.08])
        self.d._gripper_motor.send_mit.assert_not_called()

    def test_head_uses_proven_absolute_target(self):
        r=self.robot();r.head_up([0,0,-10,-20,0,0])
        np.testing.assert_allclose(self.c._q_target,np.radians([0,0,-10,-20,0,0]))
        self.assertIsNone(self.c._vlim_override)
        self.c.move_to_traj.assert_not_called()

    def test_head_rejects_every_out_of_limit_arm_axis_before_sending_a_target(self):
        robot=self.robot()
        getter=Mock(wraps=self.arm.get_state);self.arm.get_state=getter
        for i,angle in enumerate([161.,.1,-180.,91.,90.,180.]):
            angles=[0.]*7;angles[i]=angle
            with self.subTest(axis=i+1),self.assertRaisesRegex(motion.HeadJointLimitError,f'J{i+1}'):
                robot.head_up(angles)
        getter.assert_not_called()
        np.testing.assert_array_equal(self.c._q_target,np.zeros(6))
        np.testing.assert_array_equal(self.d._elevator_gripper_target,[-.15])
        self.assertIsNone(self.c._vlim_override)

    def test_nonfinite_user_targets_never_reach_sdk(self):
        r=self.robot()
        with self.assertRaises(ValueError):r.head_up([0,0,float("nan"),0,0,0])
        with self.assertRaises(ValueError):r.head_up([0]*6,duration=0)
        with self.assertRaises(ValueError):r.head_up([0,0,-10,-20,0,0,float("nan")])
        with self.assertRaises(ValueError):r.head_up([0,0,-10,-20,0,0,10],closed_tolerance=float('nan'))
        with self.assertRaises(ValueError):r.head_up([0]*7,wait_for_arrival="false")
        with self.assertRaises(ValueError):r.move(np.full((4,4),np.nan))
        self.c.move_to_traj.assert_not_called();self.d.open_gripper.assert_not_called()

    def test_cartesian_move_is_sdk_move_to_traj_with_native_limits(self):
        r=self.robot();T=pose6d_to_mat4(.4,-.1,.4,0,0,.2);r.move(T,3.)
        self.assertEqual(self.c.move_to_traj.call_args.kwargs,{"duration":3.})
        np.testing.assert_allclose(self.c.move_to_traj.call_args.args,[.4,-.1,.4,0,0,.2])
        self.assertIsNone(self.c._vlim_override)

    def test_failed_ik_and_cancel_do_not_continue(self):
        r=self.robot();self.c.move_to_traj.side_effect=None;self.c.move_to_traj.return_value=False
        with self.assertRaisesRegex(RuntimeError,"IK"):r.move(np.eye(4))
        r=self.robot()
        with self.assertRaises(motion.MotionCancelled):r.move(np.eye(4),cancelled=lambda:True)
        self.c.move_to_traj.assert_not_called()

    def test_planning_failure_is_typed_and_keeps_control_enabled_until_home(self):
        r=self.robot();self.c.move_to_traj.side_effect=None;self.c.move_to_traj.return_value=False
        target=pose6d_to_mat4(.373,-.029,.734,0.,0.,0.)
        with self.assertRaises(motion.TrajectoryPlanningError) as caught:r.move(target,3.)
        self.assertFalse(caught.exception.trajectory_sent)
        np.testing.assert_allclose(caught.exception.target,target)
        self.assertIn('本次轨迹尚未发送',str(caught.exception))
        self.assertTrue(r.started);self.assertTrue(self.c._running)
        self.assertTrue(self.c._stop_send.is_set())
        self.arm.disconnect.assert_not_called();self.c.safe_home.assert_not_called()
        events=[]
        self.c.safe_home.side_effect=lambda:events.append('home')
        self.arm.disconnect.side_effect=lambda:events.append('disconnect')
        r.close(home=True)
        self.assertEqual(events,['home','disconnect'])

    def test_failed_stop_cannot_propagate_recoverable_planning_error(self):
        r=self.robot();self.c.move_to_traj.side_effect=None;self.c.move_to_traj.return_value=False
        self.arm.get_state=Mock(side_effect=RuntimeError('feedback unavailable during stop'))
        with self.assertRaisesRegex(RuntimeError,'feedback unavailable during stop') as caught:r.move(np.eye(4))
        self.assertNotIsInstance(caught.exception,motion.TrajectoryPlanningError)
        self.arm.disconnect.assert_called();self.assertFalse(r.started)
        self.c.safe_home.assert_not_called()

    def test_inactive_controller_or_existing_sender_is_not_recoverable_ik_failure(self):
        for cause in ('inactive','sender'):
            with self.subTest(cause=cause):
                r=self.robot();self.c.move_to_traj.side_effect=None;self.c.move_to_traj.return_value=False
                if cause=='inactive':self.c._running=False
                else:self.c._send_thread=NS(is_alive=Mock(side_effect=[True,False]),join=Mock())
                with self.assertRaisesRegex(RuntimeError,'状态异常') as caught:r.move(np.eye(4))
                self.assertNotIsInstance(caught.exception,motion.TrajectoryPlanningError)
                self.c.safe_home.assert_not_called()

    def test_sdk_exception_stays_a_fault_instead_of_becoming_planning_rejection(self):
        r=self.robot();self.c.move_to_traj.side_effect=RuntimeError('serial communication error')
        with self.assertRaisesRegex(RuntimeError,'serial communication error') as caught:r.move(np.eye(4))
        self.assertNotIsInstance(caught.exception,motion.TrajectoryPlanningError)
        self.c.safe_home.assert_not_called()

    def test_cancel_queued_during_failed_ik_takes_priority_over_home_recovery(self):
        r=self.robot();self.c.move_to_traj.side_effect=None;self.c.move_to_traj.return_value=False
        cancelled=Mock(side_effect=[False,True])
        with self.assertRaises(motion.MotionCancelled):r.move(np.eye(4),cancelled=cancelled)
        self.assertEqual(cancelled.call_count,2)
        self.c.safe_home.assert_not_called()

    def test_logged_high_level_target_is_rejected_before_sdk_sender_creation(self):
        # Real model and SDK planner, fake state only: no connect/start/enable.
        motion.sdk_types()
        from reBotArm_control_py.controllers import RebotArmEndPose
        q=np.radians([0,0,-10,-20,0,0])
        arm=NS(groups={'arm':NS(num_joints=6)},has_gripper=False,arm_control_mode='posvel',
               get_state=lambda:(q.copy(),np.zeros(6),np.zeros(6)))
        controller=RebotArmEndPose(arm);controller._running=True
        with patch('reBotArm_control_py.controllers.rebotarm_endpose_controller.threading.Thread') as sender:
            self.assertFalse(controller.move_to_traj(.3729605675918177,-.028825335055700858,.7339024156014431,
                                                    0.,0.,0.,duration=3.))
        sender.assert_not_called();self.assertIsNone(controller._send_thread)
        self.assertFalse(controller._moving)

    def test_normal_exit_homes_but_fault_exit_does_not(self):
        for home in (False,True):
            r=self.robot();r.close(home=home)
            self.assertEqual(self.c.safe_home.call_count,int(home))
            self.c.end.assert_not_called()
            self.arm.disconnect.assert_called()

    def test_interrupted_sdk_home_still_disconnects(self):
        r=self.robot();self.c.safe_home.side_effect=KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):r.close(home=True)
        self.arm.disconnect.assert_called()

    def enabled_for_task_finish(self):
        robot=self.robot()
        self.c._n=6
        self.c._arm_control_mode='posvel'
        self.c._arm_group=NS(_pv_vlim=np.array([5.,5.,5.,3.,3.,3.]))
        self.c._model=NS(lowerPositionLimit=np.full(6,-3.),upperPositionLimit=np.full(6,3.))
        self.arm.control_loop_active=True
        self.arm.close_control_session=Mock()
        self.arm.joint_names=[f'joint{i}' for i in range(1,7)]+['gripper']
        self.c._q_target[:]=-.2
        def states(_, names=None, *, request_feedback=True):
            return {name:NS(status_code=1,pos=q,vel=0.)
                    for name,q in zip(self.arm.joint_names,self.c._q_target)}
        return robot,states

    @contextlib.contextmanager
    def motion_clock(self):
        now=[0.]
        def sleep(seconds):now[0]+=seconds
        with patch.object(motion.time,'monotonic',side_effect=lambda:now[0]), \
                patch.object(motion.time,'sleep',side_effect=sleep):
            yield now

    def test_pose_and_hold_trace_share_one_valid_read_without_new_feedback_requests(self):
        robot,_=self.enabled_for_task_finish()
        q=np.arange(6)*.01
        qd=q*.5
        motors={name:Mock(get_state=Mock(return_value=NS(status_code=1,pos=position,vel=velocity)))
                for name,position,velocity in zip(self.arm.joint_names,q,qd)}
        motors['joint4'].get_state.return_value.torq=-1.4
        bus=Mock()
        self.arm._motor_map=motors
        self.arm._ctrl_map={'arm':bus}
        modeled_pose=np.eye(4)
        with patch.object(robot,'_pose_from_joints',return_value=modeled_pose) as fk:
            pose,feedback=robot.pose_with_feedback()
            np.testing.assert_array_equal(fk.call_args.args[0],q)
            self.assertIs(pose,modeled_pose)
            np.testing.assert_array_equal(feedback.position_rad,q)
            np.testing.assert_array_equal(feedback.velocity_rad_s,qd)
            np.testing.assert_array_equal(feedback.target_rad,self.c._q_target)
            self.assertEqual(feedback.joint4_torque_nm,-1.4)
            bus.poll_feedback_once.assert_called_once()
            for motor in motors.values():motor.request_feedback.assert_not_called()
            motors['joint4'].get_state.return_value=None
            with self.assertRaisesRegex(motion.ArmFeedbackError,'joint4'):
                robot.pose_with_feedback()
            fk.assert_called_once()

    def test_finished_elevator_homes_and_automatically_closes_the_host_session(self):
        robot,states=self.enabled_for_task_finish()
        self.c._application_standoff_hold=True
        events=[]
        def safe_home():
            self.assertFalse(self.c._application_standoff_hold)
            events.append('home')
            self.c._q_target.fill(0.)
        self.c.safe_home.side_effect=safe_home
        def record_close():
            events.append('close')
            self.assertGreaterEqual(now[0],.25)  # Zero crossing alone is insufficient.
            self.assertLess(now[0],3.)
        self.arm.close_control_session.side_effect=record_close
        with patch.object(motion,'motor_feedback',side_effect=states), \
                self.motion_clock() as now:
            robot.finish_task(home=True)
        self.assertEqual(events,['home','close'])
        np.testing.assert_array_equal(self.c._q_target,np.zeros(6))
        self.assertFalse(self.c._running or robot.started)
        self.arm.disconnect.assert_not_called()
        self.c.end.assert_not_called()

    def test_unconfirmed_home_stops_and_cleans_up_without_disabling(self):
        robot,states=self.enabled_for_task_finish()
        with patch.object(motion,'motor_feedback',side_effect=states), \
                self.motion_clock(), \
                self.assertRaisesRegex(RuntimeError,'未确认回零'):
            robot.finish_task(home=True)
        self.c.safe_home.assert_called_once()
        np.testing.assert_allclose(self.c._q_target,np.full(6,-.2))
        self.assertFalse(robot.started)
        self.arm.close_control_session.assert_called_once()
        self.arm.disconnect.assert_not_called()

    def test_elevator_soft_stop_does_not_home_or_disable(self):
        robot,states=self.enabled_for_task_finish()
        with patch.object(motion,'motor_feedback',side_effect=states):
            robot.finish_task(home=False)
        self.c.safe_home.assert_not_called()
        self.arm.disconnect.assert_not_called()
        self.arm.close_control_session.assert_called_once()
        np.testing.assert_allclose(self.c._q_target,np.full(6,-.2))

    def test_disabled_or_missing_joint_feedback_cannot_authorize_homing(self):
        robot,states=self.enabled_for_task_finish()
        current=states(self.arm)
        current['joint3'].status_code=0
        del current['joint5']
        with patch.object(motion,'motor_feedback',return_value=current), \
                self.assertRaisesRegex(RuntimeError,'joint3.*joint5'):
            robot.finish_task(home=True)
        self.c.safe_home.assert_not_called()
        self.arm.close_control_session.assert_called_once()
        self.arm.disconnect.assert_not_called()

    def test_standoff_hold_caps_correction_without_changing_final_joint_target(self):
        robot,_=self.enabled_for_task_finish()
        self.c._arm_control_mode='posvel'
        self.c._arm_group=NS(_pv_vlim=np.array([.1,5.,5.,3.,3.,3.]))
        self.c._vlim_override=np.array([.2,.08,.2,.4,.3,.4])
        target=self.c._q_target.copy()
        robot.limit_hold_correction(.15)
        np.testing.assert_array_equal(self.c._q_target,target)
        np.testing.assert_allclose(self.c._vlim_override,[.10,.08,.15,.15,.15,.15])
        robot.move(np.eye(4))
        self.assertIsNone(self.c._vlim_override)  # No cap inherited by contact/grasp.

    def test_previous_hold_cap_is_retained_during_ik_then_released_by_sender(self):
        robot,_=self.enabled_for_task_finish()
        self.c._traj_params=NS(dt=.1);self.c._dt=.01
        self.c._traj=[np.zeros(6),np.ones(6)*.002]
        def send(duration):
            self.assertFalse(self.c._application_standoff_hold)
            self.assertIsNone(self.c._vlim_override)
        self.c._send_loop=Mock(side_effect=send)
        motion.configure_trajectory_sampling(self.c,.1)
        self.c._application_standoff_hold=True
        robot.limit_hold_correction(.05)
        def plan(*args,duration):
            self.assertTrue(self.c._application_standoff_hold)
            np.testing.assert_allclose(self.c._vlim_override,.05)
            self.c._send_loop(duration)
            return True
        self.c.move_to_traj.side_effect=plan
        robot.move(np.eye(4))
        self.c._application_original_sender.assert_called_once()

    def test_cancelled_or_empty_following_sender_keeps_the_standoff_hold(self):
        robot,_=self.enabled_for_task_finish()
        self.c._traj_params=NS(dt=.1);self.c._dt=.01
        self.c._send_loop=Mock()
        motion.configure_trajectory_sampling(self.c,.1)
        self.c._application_standoff_hold=True
        robot.limit_hold_correction(.05)
        held=self.c._q_target.copy()
        self.c._traj=[held,np.zeros(6)]
        self.c._stop_send.set()
        self.c._send_loop(1.)
        self.assertIsNone(self.c._application_sender_error)
        self.c._stop_send.clear()
        self.c._traj=[]
        self.c._send_loop(1.)
        self.assertIn('后续轨迹点不足',str(self.c._application_sender_error))
        self.c._application_original_sender.assert_not_called()
        self.assertTrue(self.c._application_standoff_hold)
        np.testing.assert_array_equal(self.c._q_target,held)
        np.testing.assert_allclose(self.c._vlim_override,.05)

    def test_approach_sender_parameter_error_does_not_count_as_arrival(self):
        robot,_=self.enabled_for_task_finish()
        self.c._arm_control_mode='posvel'
        self.c._arm_group=NS()
        self.c._traj_params=NS(dt=.1);self.c._dt=.01
        self.c._send_loop=Mock()
        motion.configure_trajectory_sampling(self.c,.1)
        def plan(*args,duration):
            self.c._stop_send.clear()
            self.c._traj=[np.zeros(6),np.ones(6)*.002]
            self.c._send_loop(duration)
            return True
        self.c.move_to_traj.side_effect=plan
        for native,hold in ((np.ones(5),.15),(np.ones(6),float('nan'))):
            self.c._arm_group._pv_vlim=native
            with self.subTest(native=native,hold=hold),patch.object(robot,'_wait_feedback_settle') as wait, \
                    self.assertRaisesRegex(motion.TrajectorySendError,'SDK轨迹发送失败.*速度参数无效'):
                robot.move(np.eye(4),settle=motion.SettledArrival('approach',hold,True,.7))
            wait.assert_not_called()
            self.assertTrue(self.c._stop_send.is_set())
        self.arm.disconnect.assert_not_called()

    def test_single_in_position_sample_with_joint_velocity_does_not_latch_or_finish(self):
        robot,_=self.enabled_for_task_finish()
        nominal=self.c._q_target.copy()
        quiet=nominal.copy();quiet[3]+=.004
        later=quiet.copy();later[3]+=.0004
        counts=[0]
        aim_during_read=[]
        def measure():
            aim_during_read.append(self.c._q_target.copy())
            counts[0]+=1
            now[0]+=.06
            if counts[0]==1:return nominal,np.array([0.,0.,0.,.22,0.,0.])
            return (quiet if counts[0]==2 else later),np.zeros(6)
        with self.motion_clock() as now,patch.object(robot,'_arm_feedback',side_effect=measure):
            actual=robot._wait_feedback_settle(motion.SettledArrival('retreat',.15,True),lambda:False,
                                               lambda q:(True,q[:3],True))
        self.assertGreaterEqual(counts[0],5)
        np.testing.assert_array_equal(aim_during_read[1],nominal)
        np.testing.assert_array_equal(self.c._q_target,quiet)
        np.testing.assert_array_equal(actual,later)

    def test_standoff_latches_measured_angles_only_after_a_quiet_window(self):
        robot,_=self.enabled_for_task_finish()
        nominal=self.c._q_target.copy()
        observed=nominal.copy();observed[1]+=.004
        self.c._qd_target[:]=.01
        caps=[]
        def measure():
            caps.append(self.c._vlim_override.copy())
            np.testing.assert_array_equal(self.c._q_target,nominal)
            self.assertFalse(getattr(self.c,'_application_standoff_hold',False))
            now[0]+=.06
            return observed,np.zeros(6)
        with self.motion_clock() as now,patch.object(robot,'_arm_feedback',side_effect=measure):
            actual=robot._wait_feedback_settle(
                motion.SettledArrival('approach',.15,stationary_vlim_rad_s=.05,lock_after_settle=True),
                lambda:False,lambda q:(True,q[:3],True))
        self.assertGreaterEqual(len(caps),4)
        np.testing.assert_allclose(caps[0],.15)
        np.testing.assert_allclose(caps[1:],.05)
        self.assertTrue(self.c._application_standoff_hold)
        np.testing.assert_array_equal(self.c._q_target,observed)
        np.testing.assert_array_equal(self.c._qd_target,np.zeros(6))
        np.testing.assert_array_equal(actual,observed)

    def test_three_mm_reversal_around_the_first_sample_does_not_count_as_settled(self):
        robot,_=self.enabled_for_task_finish()
        nominal=self.c._q_target.copy()
        count=[0]
        def measure():
            q=nominal.copy()
            if count[0]:q[0]+=.0015*(-1)**count[0]
            count[0]+=1
            now[0]+=.06
            return q,np.zeros(6)
        report=io.StringIO()
        with self.motion_clock() as now,patch.object(robot,'_arm_feedback',side_effect=measure), \
                contextlib.redirect_stdout(report),self.assertRaisesRegex(TimeoutError,'末端未停稳.*approach'):
            robot._wait_feedback_settle(
                motion.SettledArrival('approach',.15,stationary_vlim_rad_s=.05,lock_after_settle=True),
                lambda:False,lambda q:(True,q[:3],True))
        self.assertFalse(getattr(self.c,'_application_standoff_hold',False))
        self.assertIn('fk_axis_range_mm=[3.0, 0.0, 0.0]',report.getvalue())
        self.assertIn('hold_target_range_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',report.getvalue())

    def test_position_oscillation_times_out_even_with_low_instantaneous_velocity(self):
        robot,_=self.enabled_for_task_finish()
        nominal=self.c._q_target.copy()
        row=[0]
        def measure():
            row[0]+=1;now[0]+=.06
            q=nominal.copy();q[4]+=.006*(-1)**row[0]
            return q,np.zeros(6)
        with self.motion_clock() as now,patch.object(robot,'_arm_feedback',side_effect=measure), \
                self.assertRaisesRegex(TimeoutError,'末端未停稳.*retreat'):
            robot._wait_feedback_settle(motion.SettledArrival('retreat',.15,True),lambda:False,
                                        lambda q:(True,q[:3],True))
        self.assertGreaterEqual(now[0],3.)
        self.assertLess(now[0],3.1)

    def test_non_contact_hold_does_not_recommand_an_angle_outside_sdk_joint_limits(self):
        robot,_=self.enabled_for_task_finish()
        self.c._q_target[1]=0.
        self.c._model.upperPositionLimit[1]=0.
        q=self.c._q_target.copy()
        q[1]=.005  # Feedback extends a little past the limit, e.g. zero calibration.
        with self.motion_clock(),patch.object(robot,'_arm_feedback',return_value=(q,np.zeros(6))):
            robot._wait_feedback_settle(
                motion.SettledArrival('approach',.15,stationary_vlim_rad_s=.05,lock_after_settle=True),
                lambda:False,lambda sample:(True,sample[:3],True))
        self.assertTrue(self.c._application_standoff_hold)
        self.assertEqual(self.c._q_target[1],0.)

    def test_home_with_zero_position_but_moving_feedback_is_reported_as_failure(self):
        robot,_=self.enabled_for_task_finish()
        self.c.safe_home.side_effect=lambda:self.c._q_target.fill(0)
        with self.motion_clock(),patch.object(robot,'_arm_feedback',side_effect=lambda:(
                self.c._q_target.copy(),np.full(6,.1))), \
                self.assertRaisesRegex(RuntimeError,'未确认回零.*关节速度'):
            robot.finish_task(home=True)
        self.arm.close_control_session.assert_called_once()
        self.arm.disconnect.assert_not_called()

    def test_measured_ten_degree_close_reference_is_separate_from_six_axis_zero(self):
        r=self.robot()
        r.head_up([0,0,-10,-20,0,0,10])
        np.testing.assert_allclose(self.c._q_target,np.radians([0,0,-10,-20,0,0]))
        np.testing.assert_allclose(self.d._elevator_gripper_target,np.radians([10]))
        row=r.report_gripper('TEST')
        self.assertAlmostEqual(row['target_deg'],10.)
        self.assertAlmostEqual(row['actual_deg'],10.)
        self.assertTrue(row['angle_matches_reference'])
        self.d.open_gripper.assert_not_called()
        self.d._gripper_motor.send_mit.assert_not_called()

    def test_zero_feedback_does_not_count_as_arrival_at_ten_degree_reference(self):
        r=self.robot()
        self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=0.,vel=0.,torq=.01)
        clock=[0.]
        def tick():clock[0]+=.1;return clock[0]
        with patch.object(motion.time,'monotonic',side_effect=tick),patch.object(motion.time,'sleep'), \
                self.assertRaisesRegex(TimeoutError,'抬头/夹爪未到位'):
            r.head_up([0,0,-10,-20,0,0,10],duration=.5)
        self.assertEqual(self.d._elevator_gripper_target[0],0.) # fault holds actual

    def test_normal_home_keeps_closed_target_instead_of_latching_estimated_open_angle(self):
        r=self.robot();r.head_up([0,0,-10,-20,0,0,10])
        # Model a tracking error. Normal home must not latch this angle as target.
        self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=0.,vel=0.,torq=.01)
        self.c.safe_home.side_effect=lambda:np.testing.assert_allclose(
            self.d._elevator_gripper_target,np.radians([10.]))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):r.close(home=True)
        np.testing.assert_allclose(self.d._elevator_gripper_target,np.radians([10.]))
        self.c.safe_home.assert_called_once();self.arm.disconnect.assert_called_once()
        self.assertIn('AFTER_HOME_BEFORE_DISABLE',output.getvalue())
        self.assertIn('"angle_matches_reference": false',output.getvalue())

    def test_fault_exit_does_not_force_reclosure_or_home(self):
        r=self.robot();r.head_up([0,0,-10,-20,0,0,10])
        self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=-.08,vel=0.,torq=.01)
        r.close(home=False)
        np.testing.assert_allclose(self.d._elevator_gripper_target,[-.08])
        self.c.safe_home.assert_not_called();self.arm.disconnect.assert_called_once()

    def test_explicit_next_move_restores_reference_but_cancelled_move_does_not(self):
        for cancelled in (False,True):
            r=self.robot();r.head_up([0,0,-10,-20,0,0,10])
            self.d._gripper_motor.get_state.side_effect=lambda:NS(pos=.05,vel=0.,torq=.01)
            r.stop()
            self.assertEqual(self.d._elevator_gripper_target[0],.05)
            if cancelled:
                with self.assertRaises(motion.MotionCancelled):r.move(np.eye(4),cancelled=lambda:True)
                self.assertEqual(self.d._elevator_gripper_target[0],.05)
                self.c.move_to_traj.assert_not_called()
            else:
                r.move(np.eye(4))
                np.testing.assert_allclose(self.d._elevator_gripper_target,np.radians([10.]))
                self.assertEqual(len(self.c.move_to_traj.call_args.args),6)

    def test_partial_start_failure_disconnects_without_home(self):
        arm,c,d=stack();d.start.side_effect=RuntimeError("connect failed")
        r=motion.Robot({},factory=lambda cfg:(arm,c,d))
        with self.assertRaisesRegex(RuntimeError,"connect failed"):r.start()
        r.close()
        c.end.assert_not_called();arm.disconnect.assert_called()

    def transition_robot(self):
        r=self.robot();events=[];self.d._elevator_gripper_posvel=False
        self.arm.control_loop_active=True
        def stop_loop():events.append("stop_loop");self.arm.control_loop_active=False
        def start_loop(cb):
            events.append("start_loop");self.arm.control_loop_active=True
            self.assertTrue(self.d._elevator_gripper_posvel)
        self.arm.stop_control_loop=Mock(side_effect=stop_loop)
        self.arm.start_control_loop=Mock(side_effect=start_loop)
        self.d._prime_arm_target=Mock(side_effect=lambda:events.append("hold_arm"))
        self.d._loop_cb=Mock()
        self.d._gripper_group._jcfgs=[NS(vel_kp=.0008,vel_ki=.002,pos_kp=50.,pos_ki=1.)]
        m=self.d._gripper_motor
        m.get_state.side_effect=lambda:NS(pos=-1.,vel=0.,torq=.01)
        m.disable=Mock(side_effect=lambda:events.append("disable_gripper"))
        m.enable=Mock(side_effect=lambda:events.append("enable_gripper"))
        m.write_register_f32=Mock(side_effect=lambda rid,val:events.append("write:"+str(rid)))
        m.ensure_mode=Mock(side_effect=lambda *a:events.append("mode2"))
        return r,events,m

    def test_switch_only_gripper_after_pausing_callback_preserves_arm_and_connection(self):
        r,events,m=self.transition_robot()
        with patch.object(motion.time,"sleep"):r.switch_gripper_to_posvel()
        self.assertEqual(events,["stop_loop","hold_arm","disable_gripper","write:25","write:26","write:27","write:28",
                                 "mode2","enable_gripper","start_loop"])
        m.ensure_mode.assert_called_once_with(2,1000)
        self.arm.disconnect.assert_not_called();self.c.end.assert_not_called()
        np.testing.assert_allclose(self.d._elevator_gripper_target,[-1.]) # hold open, no premature close
        r.switch_gripper_to_posvel();m.enable.assert_called_once() # idempotent

    def test_switch_mode_ack_failure_or_live_callback_never_reenables(self):
        for cause in ("ack","loop"):
            r,events,m=self.transition_robot()
            if cause=="ack":m.ensure_mode.side_effect=RuntimeError("mode ack failed")
            else:self.arm.stop_control_loop.side_effect=None
            with patch.object(motion.time,"sleep"),self.assertRaises(RuntimeError):r.switch_gripper_to_posvel()
            m.enable.assert_not_called();self.arm.start_control_loop.assert_not_called()
            self.arm.disconnect.assert_called_once();self.assertFalse(r.started)

    def test_gripper_wait_uses_cancellation_only_when_sequence_installs_callback(self):
        _,driver_type=motion.sdk_types()
        with self.assertRaises(motion.MotionCancelled):
            driver_type._wait_until(NS(_application_cancelled=lambda:True),lambda:False,3.)


class LifecycleTests(unittest.TestCase):
    def hardware(self):
        events=[]
        class Arm(motion.ArmLifecycle):
            control_loop_active=False
        arm=Arm();arm._connected=False;arm._application_controller=None;arm._channel="/dev/mock"
        arm.connect=Mock(side_effect=lambda:events.append("connect"))
        arm.stop_control_loop=Mock(side_effect=lambda:events.append("stop_loop"))
        bus=NS(poll_feedback_once=Mock(),shutdown=Mock(side_effect=lambda:events.append("shutdown")),
               close_bus=Mock(side_effect=lambda:events.append("close_bus")),
               close=Mock(side_effect=lambda:events.append("close")))
        arm._ctrl_map={"damiao":bus};arm._motor_map={};arm._all_joints=[]
        for i,name in enumerate(["joint1","joint2","joint3","joint4","joint5","joint6","gripper"]):
            m=NS(state=NS(status_code=1,pos=.01*i,can_id=i+1,arbitration_id=i+17),request_feedback=Mock())
            def disable(m=m,name=name):events.append("disable:"+name);m.state.status_code=0
            def enable(m=m,name=name):events.append("enable:"+name);m.state.status_code=1
            m.disable=Mock(side_effect=disable);m.enable=Mock(side_effect=enable)
            m.get_state=Mock(side_effect=lambda m=m:m.state)
            m.write_register_f32=Mock(side_effect=lambda *a,name=name:events.append("write:"+name))
            m.ensure_mode=Mock(side_effect=lambda *a,name=name:events.append("mode:"+name))
            m.close=Mock(side_effect=lambda name=name:events.append("close_handle:"+name))
            m.get_register_u32=Mock(side_effect=lambda rid,timeout,i=i:{7:i+17,8:i+1,10:2}[rid])
            arm._motor_map[name]=m
            arm._all_joints.append(NS(name=name,vendor="damiao",motor_id=i+1,feedback_id=i+17,
                                     vel_kp=.01,vel_ki=.004,pos_kp=150.,pos_ki=.5))
        c=NS(_running=False,_arm_control_mode="posvel",_stop_send=threading.Event(),_send_thread=None)
        arm._application_controller=c
        d=NS(_arm=arm,_controller=c,_gripper_name="gripper",_loop_cb=Mock(),
             _arm_group=NS(_jcfgs=arm._all_joints[:6]),_gripper_group=NS(_jcfgs=arm._all_joints[6:]),
             _prime_arm_target=Mock(side_effect=lambda:events.append("prime_arm")),
             _prime_gripper_state=Mock(side_effect=lambda:events.append("prime_gripper")))
        arm.start_control_loop=Mock(side_effect=lambda cb:events.append("start_loop"))
        return arm,bus,d,events

    def test_modes_and_hold_targets_precede_any_enable(self):
        arm,bus,d,events=self.hardware()
        with patch.object(motion.time,"sleep"):
            motion.start_driver(d)
        self.assertTrue(d._controller._running)
        first_enable=events.index("enable:joint1")
        self.assertLess(events.index("mode:gripper"),first_enable)
        self.assertLess(events.index("prime_arm"),first_enable)
        self.assertLess(events.index("prime_gripper"),first_enable)
        self.assertEqual(sum(e.startswith("enable:") for e in events),7)
        self.assertEqual(events[-1],"start_loop")
        arm._motor_map["gripper"].ensure_mode.assert_called_once_with(1,1000)

    def test_joint4_integral_adjustment_and_rollback_preserve_kp_and_other_joints(self):
        for ki_scale in (.5,1.):
            with self.subTest(ki_scale=ki_scale):
                arm,_,driver,events=self.hardware()
                arm._all_joints[3].pos_kp=70.
                arm._all_joints[3].pos_ki=1.
                joint4=arm._motor_map['joint4']
                def readback(rid,timeout):
                    events.append(f'read_j4_{rid}')
                    return {27:56.,28:ki_scale}[rid]
                joint4.get_register_f32=Mock(side_effect=readback)
                driver.start=lambda:motion.start_driver(driver)
                robot=motion.Robot({'robot_control':{'joint4_pos_kp_scale':.8,'joint4_pos_ki_scale':ki_scale}},
                                   factory=lambda cfg:(arm,driver._controller,driver))
                with patch.object(motion.time,'sleep'):
                    robot.start()
                self.assertEqual([call.args for call in joint4.get_register_f32.call_args_list],[(27,1000),(28,1000)])
                self.assertLess(events.index('read_j4_28'),events.index('enable:joint1'))
                self.assertEqual(arm._all_joints[3].pos_kp,70.)
                self.assertEqual(arm._all_joints[3].pos_ki,1.)
                for joint in arm._all_joints:
                    kp,ki=(56.,ki_scale) if joint.name=='joint4' else (joint.pos_kp,joint.pos_ki)
                    values=[call.args for call in arm._motor_map[joint.name].write_register_f32.call_args_list]
                    self.assertEqual(values,[(25,joint.vel_kp),(26,joint.vel_ki),(27,kp),(28,ki)])

    def test_joint4_gain_readback_mismatch_prevents_any_enable(self):
        for readback,failed_gain in (({27:70.,28:.5},'pos_kp'),({27:56.,28:1.},'pos_ki')):
            with self.subTest(gain=failed_gain):
                arm,_,driver,events=self.hardware()
                driver._joint4_pos_kp_scale=.8
                driver._joint4_pos_ki_scale=.5
                arm._all_joints[3].pos_kp=70.
                arm._all_joints[3].pos_ki=1.
                arm._motor_map['joint4'].get_register_f32=Mock(side_effect=lambda rid,timeout:readback[rid])
                with patch.object(motion.time,'sleep'),self.assertRaisesRegex(RuntimeError,f'joint4.*{failed_gain}.*读回='):
                    motion.start_driver(driver)
                self.assertFalse(any(event.startswith('enable:') for event in events))
                arm.start_control_loop.assert_not_called()

    def test_invalid_integral_scale_is_rejected_before_constructing_devices(self):
        for value in (-.1,float('nan'),True,1.1):
            with self.subTest(scale=value):
                factory=Mock(side_effect=AssertionError('device'))
                robot=motion.Robot({'robot_control':{'joint4_pos_ki_scale':value}},factory=factory)
                with self.assertRaisesRegex(ValueError,'joint4_pos_ki_scale'):
                    robot.start()
                factory.assert_not_called()

    def test_elevator_initializes_seventh_pos_vel_before_enable_and_holds_measured_angle(self):
        arm,bus,d,events=self.hardware();d._elevator_gripper_posvel=True
        arm._motor_map["gripper"].state.pos=-.15
        with patch.object(motion.time,"sleep"):motion.start_driver(d)
        m=arm._motor_map["gripper"]
        m.ensure_mode.assert_called_once_with(2,1000)
        self.assertEqual(m.write_register_f32.call_count,4)
        np.testing.assert_allclose(d._elevator_gripper_target,[-.15])
        self.assertLess(events.index("mode:gripper"),events.index("enable:joint1"))

    def test_elevator_gripper_parameter_failure_prevents_all_enabling(self):
        arm,bus,d,events=self.hardware();d._elevator_gripper_posvel=True
        arm._motor_map["gripper"].write_register_f32.side_effect=RuntimeError("gripper ack timeout")
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"gripper.*ack"):
            motion.start_driver(d)
        self.assertFalse(any(e.startswith("enable:") for e in events))
        arm.start_control_loop.assert_not_called()

    def test_sdk_callback_sends_seven_pos_vel_without_competing_mit_tick(self):
        from drivers.robot.grasp_driver import GraspDriver
        _,Driver=motion.sdk_types()
        d=object.__new__(Driver)
        d._controller=NS(_loop_cb=Mock())
        d._elevator_gripper_posvel=True;d._state_lock=threading.Lock();d._elevator_gripper_target=np.radians([10.])
        d._gripper_group=NS(send_pos_vel=Mock(),send_mit=Mock())
        d.gripper_tick=Mock()
        d._loop_cb(None,.002)
        d._controller._loop_cb.assert_called_once_with(None,.002)
        np.testing.assert_allclose(d._gripper_group.send_pos_vel.call_args.args[0],np.radians([10.]))
        d.gripper_tick.assert_not_called();d._gripper_group.send_mit.assert_not_called()
        # Grasp mode delegates exactly to the original driver callback.
        d._elevator_gripper_posvel=False
        with patch.object(GraspDriver,"_loop_cb") as original:
            d._loop_cb(None,.002)
        original.assert_called_once_with(None,.002)

    def test_parameter_and_mode_ack_failures_abort_before_enabling(self):
        for operation in ("write_register_f32","ensure_mode"):
            with self.subTest(operation=operation):
                arm,bus,d,events=self.hardware()
                getattr(arm._motor_map["joint2"],operation).side_effect=RuntimeError("ack not received within 50ms")
                with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"joint2.*ack"):
                    motion.start_driver(d)
                self.assertFalse(any(e.startswith("enable:") for e in events))
                arm.start_control_loop.assert_not_called()
                bus.close.assert_called_once()
                self.assertFalse(d._controller._running)

    def test_missing_feedback_never_becomes_zero_position(self):
        arm,bus,d,events=self.hardware()
        m=arm._motor_map["joint3"];m.get_state.side_effect=[NS(status_code=0,pos=0),None]+[None]*4
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"缺少有效.*joint3"):
            motion.start_driver(d)
        self.assertFalse(any(e.startswith("enable:") for e in events))
        bus.close.assert_called_once()

    def test_partial_connect_and_shutdown_error_still_close_bus(self):
        arm,bus,d,events=self.hardware()
        bus.shutdown.side_effect=RuntimeError("native shutdown failure")
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"native shutdown failure"):
            arm.disconnect()
        self.assertLess(events.index("stop_loop"),events.index("disable:joint1"))
        self.assertLess(events.index("close_handle:gripper"),events.index("close"))
        bus.close.assert_called_once()
        self.assertFalse(arm._connected)
        arm.disconnect()  # Idempotent even after an unsuccessful native shutdown.
        bus.close.assert_called_once()

    def test_local_exit_stops_loop_and_frees_handles_without_disable_or_native_shutdown(self):
        arm,bus,d,events=self.hardware()
        motors=list(arm._motor_map.values())
        arm.close_control_session()
        arm.close_control_session()
        self.assertEqual(events[:2],['stop_loop','close_bus'])
        bus.shutdown.assert_not_called()
        bus.close.assert_called_once()
        self.assertFalse(arm._ctrl_map or arm._motor_map)
        for motor in motors:
            motor.disable.assert_not_called()
            motor.close.assert_called_once()

    def test_local_exit_never_frees_handles_beneath_a_running_control_thread(self):
        arm,bus,d,_=self.hardware()
        arm.control_loop_active=True
        with self.assertRaisesRegex(RuntimeError,'控制循环线程未退出'):
            arm.close_control_session()
        bus.close.assert_not_called()
        bus.close_bus.assert_not_called()
        bus.shutdown.assert_not_called()

    def test_connect_exception_cleans_partial_bus_without_enabling(self):
        arm,bus,d,events=self.hardware()
        arm.connect.side_effect=RuntimeError("motor registration failed")
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"motor registration failed"):
            motion.start_driver(d)
        self.assertFalse(any(e.startswith("enable:") for e in events))
        bus.close.assert_called_once()

    def test_unconfirmed_disable_is_reported_and_does_not_skip_other_motors(self):
        arm,bus,d,events=self.hardware()
        bad=arm._motor_map["joint2"]
        bad.disable.side_effect=RuntimeError("write failed")
        good=arm._motor_map["gripper"]
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"失能反馈未确认"):
            arm.disconnect()
        self.assertEqual(bad.disable.call_count,3)
        good.disable.assert_called_once();bus.close.assert_called_once()
        self.assertIn("joint2",arm._application_shutdown_error)

    def test_disabled_status_is_required_and_successful_close_is_idempotent(self):
        arm,bus,d,events=self.hardware()
        motor=arm._motor_map["joint4"];motor.disable.side_effect=None
        # Successful write alone is insufficient: status remains enabled.
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"joint4"):
            arm.disconnect()
        self.assertEqual(motor.disable.call_count,3)
        arm,bus,d,events=self.hardware()
        with patch.object(motion.time,"sleep"):
            arm.disconnect();arm.disconnect()
        bus.close.assert_called_once();self.assertEqual(arm._application_shutdown_error,"")

    def test_check_bus_queries_enabled_motors_without_changing_state(self):
        arm,bus,d,events=self.hardware()
        motors=list(arm._motor_map.values())
        with patch.object(motion.time,"sleep"):
            self.assertEqual(motion.check_bus({},arm_factory=lambda:arm),0)
        for m in motors:
            self.assertEqual(m.state.status_code,1)
            m.enable.assert_not_called();m.disable.assert_not_called()
            m.ensure_mode.assert_not_called();m.write_register_f32.assert_not_called()
            m.close.assert_called_once()
        arm.start_control_loop.assert_not_called();bus.shutdown.assert_not_called()
        bus.close_bus.assert_called_once();bus.close.assert_called_once()

    def test_check_bus_reports_missing_feedback_and_register_timeout(self):
        arm,bus,d,events=self.hardware()
        for name,m in arm._motor_map.items():
            if name!="joint1":
                m.get_state.side_effect=None;m.get_state.return_value=None
                m.get_register_u32.side_effect=RuntimeError("no reply")
        with patch.object(motion.time,"sleep"):
            self.assertEqual(motion.check_bus({},arm_factory=lambda:arm),2)
        bus.shutdown.assert_not_called();bus.close.assert_called_once()

    def test_check_bus_rejects_id_mismatch_without_auto_writing_ids(self):
        arm,bus,d,events=self.hardware()
        m=arm._motor_map["joint2"];m.get_register_u32.side_effect=lambda rid,timeout:{7:99,8:2,10:2}[rid]
        with patch.object(motion.time,"sleep"):
            self.assertEqual(motion.check_bus({},arm_factory=lambda:arm),2)
        m.write_register_f32.assert_not_called();m.ensure_mode.assert_not_called()

    def test_check_bus_connection_failure_still_closes_partial_handles(self):
        arm,bus,d,events=self.hardware()
        arm.connect.side_effect=RuntimeError("partial registration")
        with self.assertRaisesRegex(RuntimeError,"partial registration"):
            motion.check_bus({},arm_factory=lambda:arm)
        bus.close_bus.assert_called_once();bus.close.assert_called_once();bus.shutdown.assert_not_called()
        self.assertEqual(sum(e.startswith("close_handle:") for e in events),7)

    def test_check_bus_close_failure_still_frees_other_handles(self):
        arm,bus,d,events=self.hardware()
        bus.close_bus.side_effect=RuntimeError("close bus failed")
        with patch.object(motion.time,"sleep"),self.assertRaisesRegex(RuntimeError,"资源释放失败"):
            motion.check_bus({},arm_factory=lambda:arm)
        bus.close.assert_called_once()
        self.assertEqual(sum(e.startswith("close_handle:") for e in events),7)


class ActualSDKPlanningTests(unittest.TestCase):
    def test_coarse_ik_interpolated_approach_contact_retreat_path(self):
        self.check_sdk_path(combined=True,point_override=[.473837607879697,-.10437780881625228,.37006380717012877],sample_dt=.1)

    def test_place_location_with_ready_pitch_and_outward_yaw_has_valid_sdk_path(self):
        import pinocchio as pin
        from elevator import grasp
        motion.sdk_types()
        from reBotArm_control_py.kinematics import load_robot_model,compute_fk,pad_q_for_model,get_end_effector_frame_id
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik,IKParams
        from reBotArm_control_py.trajectory import TrajPlanParams,TrajProfile,IKParams as CLIKParams,plan_cartesian_geodesic_trajectory,track_trajectory
        model=load_robot_model();frame=get_end_effector_frame_id(model)
        q=pad_q_for_model(model,np.zeros(6),6)
        cfg={"robot":{"ready_pose":{"roll":0.,"pitch":.7}}}
        ready=pose6d_to_mat4(.3,0,.3,0,.7,0);target=grasp.placement_pose(cfg,[.25,-.38,.35])
        params=IKParams(max_iter=200,tolerance=1e-4,step_size=.5,damping=1e-6)
        first=solve_ik(model,model.createData(),frame,pin.SE3(ready),q,params,controlled_joints=6)
        self.assertTrue(first.success);q=pad_q_for_model(model,first.q,6)
        finish=solve_ik(model,model.createData(),frame,pin.SE3(target),q,params,controlled_joints=6)
        self.assertTrue(finish.success)
        cart=plan_cartesian_geodesic_trajectory(pin.SE3(compute_fk(model,q)[2]),
            pin.SE3(compute_fk(model,pad_q_for_model(model,finish.q,6))[2]),3.,TrajPlanParams(dt=.01,profile=TrajProfile.MIN_JERK))
        path=track_trajectory(model,frame,cart.trajectory,q,
            CLIKParams(max_iter=200,tolerance=1e-4,damping=1e-6,step_size=.8),null_gain=.1,controlled_joints=6)
        errors=[pin.log6(pin.SE3(compute_fk(model,p.q)[2]).inverse()*cp.pose).vector for p,cp in zip(path,cart.trajectory.points())]
        pos=max(np.linalg.norm(e[:3]) for e in errors);rot=max(np.linalg.norm(e[3:]) for e in errors)
        print(f"[OfflinePlace] samples={len(path)} strict_ik={sum(p.ik_success for p in path)} max_pos_m={pos:.6f} max_angle_rad={rot:.6f}")
        self.assertLess(pos,.005);self.assertLess(rot,.02)
        joints=np.array([p.q for p in path]);self.assertTrue(np.isfinite(joints).all())
        self.assertTrue(np.all(joints>=model.lowerPositionLimit-1e-6));self.assertTrue(np.all(joints<=model.upperPositionLimit+1e-6))
        self.assertLess(np.linalg.norm(compute_fk(model,joints[-1])[2][:3,3]-target[:3,3]),.002)

    def test_level_approach_surface_retreat_with_original_sdk_planner(self):
        self.check_sdk_path(combined=False)

    def test_combined_level_and_approach_with_logged_floor3_target(self):
        self.check_sdk_path(combined=True)

    def test_latest_floor3_with_measured_base_correction_uses_original_sdk_planner(self):
        from elevator.app import load_workflow,ROOT
        offset=load_workflow(ROOT/"config/261105_my_elevator.yaml")["elevator"]["target_offset_base_m"]
        point=geometry.offset_target_base([.38876175478615094,-.11099564406050222,.4314943958436071],offset)
        self.check_sdk_path(combined=True,point_override=point)

    def check_sdk_path(self,combined,point_override=None,sample_dt=.01):
        # Read local configuration/model only; never construct RebotArm or a controller.
        import pinocchio as pin
        from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath
        ensure_rebot_sdk_in_syspath()
        from reBotArm_control_py.kinematics import load_robot_model,compute_fk,pad_q_for_model,get_end_effector_frame_id
        from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik,IKParams
        from reBotArm_control_py.trajectory import TrajPlanParams,TrajProfile,IKParams as CLIKParams,plan_cartesian_geodesic_trajectory,track_trajectory
        model=load_robot_model();frame=get_end_effector_frame_id(model)
        q=pad_q_for_model(model,np.radians([0,0,-10,-20,0,0]),6)
        start=compute_fk(model,q)[2];level=geometry.level_pose(start)
        point=[.39102363866428197,-.11358868289549706,.4307580929315735] if combined else [.39450668,-.11021567,.42816687]
        if point_override is not None:point=point_override
        poses=geometry.button_poses(np.array(point),level,[0,0,0],.05,0.)
        goals=[("level",level,3.),("approach",poses["approach"],3.),("surface",poses["surface"],1.5),("retreat",poses["retreat"],2.)]
        if combined:goals=goals[1:]
        for name,target,duration in goals:
            with self.subTest(segment=name):
                ik=solve_ik(model,model.createData(),frame,pin.SE3(target),q,
                    IKParams(max_iter=200,tolerance=1e-4,step_size=.5,damping=1e-6),controlled_joints=6)
                self.assertTrue(ik.success,f"{name}: IK error={ik.error}")
                finish=compute_fk(model,pad_q_for_model(model,ik.q,6))[2]
                cart=plan_cartesian_geodesic_trajectory(pin.SE3(compute_fk(model,q)[2]),pin.SE3(finish),duration,
                    TrajPlanParams(dt=sample_dt,profile=TrajProfile.MIN_JERK))
                tracked=track_trajectory(model,frame,cart.trajectory,q,
                    CLIKParams(max_iter=200,tolerance=1e-4,damping=1e-6,step_size=.8),null_gain=.1,controlled_joints=6)
                self.assertTrue(tracked)
                ideal=plan_cartesian_geodesic_trajectory(pin.SE3(compute_fk(model,q)[2]),pin.SE3(finish),duration,
                    TrajPlanParams(dt=.01,profile=TrajProfile.MIN_JERK))
                dense=motion.densify_joint_samples([p.q for p in tracked],duration,.01)
                # SDK's 1e-4 log6 convergence flag is stricter than practical
                # path accuracy. Check FK error rather than treating every
                # sub-millimeter nonconvergence flag as an unusable path.
                errors=[pin.log6(pin.SE3(compute_fk(model,p)[2]).inverse()*cp.pose).vector
                    for p,cp in zip(dense,ideal.trajectory.points())]
                pos_error=max(np.linalg.norm(e[:3]) for e in errors)
                angle_error=max(np.linalg.norm(e[3:]) for e in errors)
                print(f"[OfflineSDK] {name}: samples={len(tracked)} strict_ik={sum(p.ik_success for p in tracked)} "
                      f"max_FK_position_m={pos_error:.6f} max_FK_angle_rad={angle_error:.6f}",flush=True)
                self.assertLess(pos_error,.001 if combined else .005,name)
                self.assertLess(angle_error,.02,name)
                joints=np.array([p.q for p in tracked])
                peak=np.max(np.abs(np.diff(joints[:,:6],axis=0))/np.diff([p.time for p in tracked])[:,None])
                print(f"[OfflineSDK] {name}: max_joint_reference_rad_s={peak:.6f}",flush=True)
                self.assertTrue(np.isfinite(joints).all())
                self.assertTrue(np.all(joints>=np.array(model.lowerPositionLimit)-1e-6))
                self.assertTrue(np.all(joints<=np.array(model.upperPositionLimit)+1e-6))
                q=joints[-1]
                self.assertLess(np.linalg.norm(compute_fk(model,q)[2][:3,3]-target[:3,3]),.002)
