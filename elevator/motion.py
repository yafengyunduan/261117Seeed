"""Thin SDK adapter matching scripts/main.py; no custom velocity watchdog/planner."""
from __future__ import annotations

import time
import json
import threading
import numpy as np
from utils.transforms import mat4_to_pose6d


class MotionCancelled(RuntimeError):
    pass


class TrajectoryPlanningError(RuntimeError):
    """SDK rejected a new plan before starting its trajectory sender.

    Robot.move only propagates this type after stop/hold succeeds. Control,
    feedback or sender-stop errors keep their original exception types.
    """
    def __init__(self, target, duration, *, position_only=False, tip_target=None, sample_index=None, ik_error=None):
        self.target = np.asarray(target, dtype=float).copy()
        self.duration_s = float(duration)
        self.trajectory_sent = False
        self.position_only = bool(position_only)
        pose = mat4_to_pose6d(self.target)
        detail = (f'tip_target_base_m={np.asarray(tip_target).tolist()} orientation=FREE '
                  f'sample_index={sample_index} ik_error_m={ik_error}' if position_only else
                  f'target_base_m={self.target[:3,3].tolist()} target_rpy_deg={np.degrees(pose[3:]).tolist()}')
        super().__init__('SDK IK/轨迹规划失败，本次轨迹尚未发送；'+detail)


def plan_position_trajectory(controller, q_start, tip_target, duration, tip_offset=(0.,0.,0.), cancelled=lambda:False):
    """Pure local plan: original SDK sampler + IK, constraining only tool-tip XYZ.

    Joint limits and IK tolerance come from the loaded SDK model/parameters.
    All knots are solved before any sender starts. No hardware access here.
    """
    import pinocchio as pin
    from reBotArm_control_py.kinematics import pad_q_for_model
    from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik
    from reBotArm_control_py.trajectory import plan_cartesian_geodesic_trajectory
    point, offset = np.asarray(tip_target,float), np.asarray(tip_offset,float)
    q_start = np.asarray(q_start,float)
    if (point.shape!=(3,) or offset.shape!=(3,) or q_start.ndim!=1 or len(q_start)<controller._n
            or not np.isfinite(point).all() or not np.isfinite(offset).all()
            or not np.isfinite(q_start[:controller._n]).all() or not np.isfinite(duration) or duration<=0):
        raise ValueError('位置IK需要有效的尖端XYZ、工具偏移、关节反馈和运动时间')
    model = controller._model
    end_frame = controller._end_frame_id
    tip_frame = end_frame
    if np.any(offset):
        model = model.copy()  # A virtual tip frame never changes the SDK model or URDF.
        end = model.frames[end_frame]
        tip_frame = model.addFrame(pin.Frame('__application_tip',end.parentJoint,end_frame,
            end.placement*pin.SE3(np.eye(3),offset),pin.OP_FRAME),False)
    data = model.createData()
    q = pad_q_for_model(model,q_start,controller._n)
    pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
    start = data.oMf[tip_frame].copy()
    placeholder = data.oMf[end_frame].homogeneous.copy()
    placeholder[:3,3] = point-placeholder[:3,:3]@offset
    cart = plan_cartesian_geodesic_trajectory(start,pin.SE3(start.rotation.copy(),point),
                                             float(duration),controller._traj_params)
    points, errors = [], []
    for index, sample in enumerate(cart.trajectory.points()):
        if cancelled():raise MotionCancelled('用户取消位置IK规划')
        # Preserve measured q at t=0, then seed subsequent knots inside the
        # existing limits. This handles a slightly positive measured J2 without
        # extending URDF limits or commanding a separate zeroing movement.
        seed = q if index==0 else np.clip(q,model.lowerPositionLimit,model.upperPositionLimit)
        result = solve_ik(model,data,tip_frame,sample.pose,seed,controller._ik_solver_params,
                          controlled_joints=controller._n,position_only=True)
        if not result.success or not np.isfinite(result.q).all():
            raise TrajectoryPlanningError(placeholder,duration,position_only=True,tip_target=point,
                                           sample_index=index,ik_error=float(result.error))
        q = pad_q_for_model(model,result.q,controller._n)
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        error = float(np.linalg.norm(data.oMf[tip_frame].translation-sample.pose.translation))
        if not np.isfinite(error) or error>controller._ik_solver_params.tolerance*1.01:
            raise TrajectoryPlanningError(placeholder,duration,position_only=True,tip_target=point,
                                           sample_index=index,ik_error=error)
        points.append(q[:controller._n].copy());errors.append(error)
    if len(points)<2:
        raise TrajectoryPlanningError(placeholder,duration,position_only=True,tip_target=point)
    end_pose = data.oMf[end_frame].homogeneous.copy()
    return dict(points=points,end_pose=end_pose,tip_target=point.copy(),tip_offset=offset.copy(),
                report=dict(ik_points=len(points),max_tip_error_m=max(errors),position_only=True,
                            end_rpy_deg=np.degrees(mat4_to_pose6d(end_pose)[3:]).tolist()))


def densify_joint_samples(samples, duration, interval):
    """Interpolate coarse SDK IK knots onto the original sender's time grid.

    Joint interpolation stays between adjacent solutions; no new IK is run.
    This is a model path approximation, not a hardware accuracy guarantee.
    """
    q = np.asarray(samples, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("SDK轨迹需要至少两个有限关节点")
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(interval) or interval <= 0:
        raise ValueError("轨迹时间和发送间隔必须为有限正数")
    count = max(len(q), int(np.ceil(duration / interval)) + 1)
    source = np.linspace(0., duration, len(q))
    target = np.linspace(0., duration, count)
    return np.column_stack([np.interp(target, source, column) for column in q.T])


def configure_trajectory_sampling(controller, sample_interval_s):
    """Instance-local adaptation; SDK source, callback and speed limits unchanged."""
    sample_interval_s = float(sample_interval_s)
    if not np.isfinite(sample_interval_s) or not .01 <= sample_interval_s <= .1:
        raise ValueError("trajectory.sample_interval_s需在[0.01,0.1]秒")
    controller._traj_params.dt = sample_interval_s
    if not hasattr(controller, "_application_original_sender"):
        controller._application_original_sender = controller._send_loop

        def send_with_interpolation(duration):
            planned = len(controller._traj)
            if planned >= 2:
                controller._traj = list(densify_joint_samples(controller._traj, duration, controller._dt))
            controller._application_trajectory_report = dict(
                sample_interval_s=controller._traj_params.dt, ik_points=planned,
                send_points=len(controller._traj), send_interval_s=duration/max(1, len(controller._traj)))
            print(f"[Trajectory] IK采样={controller._traj_params.dt:.3f}s/{planned}点；"
                  f"插值发送={len(controller._traj)}点；电机控制循环频率不变", flush=True)
            controller._application_original_sender(duration)

        controller._send_loop = send_with_interpolation


def motor_feedback(arm):
    """Read actual MotorState objects; SDK get_state silently replaces None by 0."""
    for motor in arm._motor_map.values():
        motor.request_feedback()
        time.sleep(.01)
    for bus in arm._ctrl_map.values():
        bus.poll_feedback_once()
    return {name: motor.get_state() for name, motor in arm._motor_map.items()}


def disable_motors(arm):
    """Paced native disable commands, bounded retries, and reported DM status."""
    pending = dict(arm._motor_map)
    details = {}
    for _ in range(3):
        for name, motor in list(pending.items()):
            try:
                motor.disable()
                time.sleep(.03)
                motor.request_feedback()
                time.sleep(.02)
                for bus in arm._ctrl_map.values():
                    bus.poll_feedback_once()
                state = motor.get_state()
                details[name] = None if state is None else state.status_code
                if state is not None and state.status_code == 0:
                    pending.pop(name)
            except Exception as exc:
                details[name] = str(exc)
        if not pending:
            return details
    absent = [name for name, value in details.items() if value is None]
    hint = f"；通信无反馈={absent}，可先运行 --check-bus 查询通信" if absent else ""
    raise RuntimeError(f"失能反馈未确认：{details}{hint}；请保持机械支撑并检查电源/通信")


class ArmLifecycle:
    """Application-only SDK subclass mixin. Original SDK files stay unchanged."""
    def disconnect(self):
        if not self._ctrl_map and not self._motor_map:
            return
        errors = []

        def attempt(label, fn):
            try:
                return fn()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        controller = getattr(self, "_application_controller", None)
        if controller is not None:
            controller._stop_send.set()
            sender = getattr(controller, "_send_thread", None)
            if sender is not None:
                attempt("轨迹线程", lambda: sender.join(timeout=1.))
                if sender.is_alive():
                    errors.append("轨迹线程未退出")
            controller._running = False
        attempt("控制循环停止", self.stop_control_loop)
        if self.control_loop_active:
            errors.append("控制循环线程未退出")
        statuses = attempt("逐电机失能", lambda: disable_motors(self))
        # Even partial connect(), failed home(), or failed shutdown() must close every bus.
        for vendor, bus in list(self._ctrl_map.items()):
            attempt(f"{vendor} shutdown", bus.shutdown)
        # Motor.close frees the native Arc holding the bus. Python Motor has no
        # destructor; clearing the dict alone leaks the serial handle until exit.
        for name, motor in list(self._motor_map.items()):
            attempt(f"{name} handle close", motor.close)
        for vendor, bus in list(self._ctrl_map.items()):
            attempt(f"{vendor} close", bus.close)
        self._ctrl_map.clear()
        self._motor_map.clear()
        self._connected = False
        self._application_shutdown_error = "; ".join(errors)
        if errors:
            raise RuntimeError("退出清理未全部确认：" + self._application_shutdown_error)
        print(f"[Exit] 控制循环已停止；失能状态反馈={statuses}；串口已关闭。反馈无包时间戳。", flush=True)


def check_bus(cfg, *, arm_factory=None):
    """Queries only: no enable/disable, mode/PID writes, home, or control loop."""
    if arm_factory is None:
        from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath
        ensure_rebot_sdk_in_syspath((cfg.get("robot") or {}).get("repo_root"))
        from reBotArm_control_py.actuator import RebotArm
        arm_factory = RebotArm  # Do not use ManagedArm.disconnect in this query path.
    arm = arm_factory()
    rows = []
    print(f"[BusCheck] {arm._channel}；仅查询反馈和ID/模式寄存器，不改变电机使能或目标", flush=True)
    try:
        if any(j.vendor != "damiao" for j in arm._all_joints):
            raise RuntimeError("--check-bus 当前适用于 B601-DM")
        arm.connect()
        for jc in arm._all_joints:
            motor = arm._motor_map[jc.name]
            row = dict(name=jc.name, motor_id=jc.motor_id, expected_feedback_id=jc.feedback_id)
            try:
                # Multiple bounded read windows allow delayed replies, without
                # interpreting an absent MotorState as a zero joint angle.
                for _ in range(3):
                    motor.request_feedback()
                    for _ in range(15):
                        for bus in arm._ctrl_map.values():
                            bus.poll_feedback_once()
                        time.sleep(.01)
                state = motor.get_state()
                row["state"] = None if state is None else dict(
                    can_id=state.can_id, arbitration_id=state.arbitration_id,
                    status=state.status_code, position_deg=float(np.degrees(state.pos)))
            except Exception as exc:
                row.update(state=None, feedback_error=str(exc))
            for rid, label in ((7,"MST_ID"),(8,"ESC_ID"),(10,"CTRL_MODE")):
                try:
                    row[label] = motor.get_register_u32(rid, 200)
                except Exception as exc:
                    row[label] = str(exc)
            rows.append(row)
            print("[BusCheck] " + json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        errors = []
        # Native shutdown()/SDK disconnect() send disable commands. close_bus()
        # only stops native polling and flushes the port (motorbridge 0.5.6).
        operations = [("close_bus",bus.close_bus) for bus in arm._ctrl_map.values()]
        operations += [(name+" handle close",motor.close) for name,motor in arm._motor_map.items()]
        operations += [("controller close",bus.close) for bus in arm._ctrl_map.values()]
        for label, operation in operations:
            try:
                operation()
            except BaseException as exc:
                errors.append(f"{label}: {exc}")
        if errors:
            raise RuntimeError("通信查询资源释放失败：" + "; ".join(errors))
    absent = [r["name"] for r in rows if r["state"] is None]
    mismatched = [r["name"] for r in rows if r["MST_ID"] != r["expected_feedback_id"]
                  or r["ESC_ID"] != r["motor_id"] or r["CTRL_MODE"] not in (1,2,3,4)]
    print(f"[BusCheck] 有反馈={len(rows)-len(absent)}/{len(rows)}；无反馈={absent}；"
          f"寄存器未确认或ID不符={mismatched}；查询结束，未执行动作", flush=True)
    return 2 if absent or mismatched else 0


def start_driver(driver):
    """Same SDK/driver callbacks, with checked initialization before enabling."""
    arm, c = driver._arm, driver._controller
    if c._running:
        return
    c._has_gripper = False
    try:
        if any(j.vendor != "damiao" for j in arm._all_joints):
            raise RuntimeError("当前电梯入口的逐电机初始化适配 B601-DM")
        arm.connect()
        # Previous process may have exited while hardware stayed enabled.
        statuses = disable_motors(arm)
        print(f"[Robot] 初始化前失能状态反馈={statuses}", flush=True)
        gripper_posvel=getattr(driver,"_elevator_gripper_posvel",False)
        for group, mode in ((driver._arm_group, 1 if c._arm_control_mode == "mit" else 2),
                            (driver._gripper_group, 2 if gripper_posvel else 1)):
            for jc in group._jcfgs:
                motor = arm._motor_map[jc.name]
                step = "mode"
                try:
                    if mode == 2:
                        for rid, value in ((25,jc.vel_kp),(26,jc.vel_ki),(27,jc.pos_kp),(28,jc.pos_ki)):
                            step = f"register {rid}"
                            motor.write_register_f32(rid, value)
                            time.sleep(.01)
                    step = f"mode {mode}"
                    motor.ensure_mode(mode, 1000)
                    time.sleep(.05)
                except Exception as exc:
                    raise RuntimeError(f"启动通信失败：{jc.name} {step}: {exc}；未启动动作") from exc
            group._mode = "mit" if mode == 1 else "pos_vel"
        states = motor_feedback(arm)
        missing = [j.name for j in arm._all_joints if states.get(j.name) is None
                   or not np.isfinite(states[j.name].pos)]
        if missing:
            raise RuntimeError(f"启动缺少有效电机反馈：{missing}；不能将缺失反馈当作零位")
        driver._gripper_motor = arm._motor_map[driver._gripper_name]
        driver._prime_arm_target()
        driver._prime_gripper_state()
        if gripper_posvel:
            # Match example/4: hold measured angle until a seven-axis target
            # is submitted. Enabling must not implicitly close the gripper.
            driver._elevator_gripper_target=np.array([states[driver._gripper_name].pos],float)
        # JointGroup.enable() actually enables ALL motors on the vendor bus.
        # Enable each one only after both groups have been configured and primed.
        for name, motor in arm._motor_map.items():
            try:
                motor.enable()
            except Exception as exc:
                raise RuntimeError(f"启动使能失败：{name}: {exc}") from exc
            time.sleep(.03)
        arm.start_control_loop(driver._loop_cb)
        c._running = True
        print("[Robot] 参数/模式写入成功，7电机反馈有效；SDK控制循环已启动", flush=True)
    except BaseException:
        c._running = False
        try:
            arm.disconnect()
        except BaseException as exc:
            print(f"[Exit] {exc}", flush=True)
        raise


def sdk_types(repo_root=None):
    from drivers.robot.grasp_driver import GraspDriver, ensure_rebot_sdk_in_syspath
    ensure_rebot_sdk_in_syspath(repo_root)
    from reBotArm_control_py.actuator import RebotArm

    class ManagedArm(ArmLifecycle, RebotArm):
        pass

    class CheckedDriver(GraspDriver):
        def start(self):
            self._arm._application_controller = self._controller
            start_driver(self)

        def _loop_cb(self,r,dt):
            if not getattr(self,"_elevator_gripper_posvel",False):
                return super()._loop_cb(r,dt)
            # Same group send_pos_vel calls as example/4, in the one SDK loop.
            # Controller._has_gripper stays false; no competing MIT packets.
            self._controller._loop_cb(r,dt)
            with self._state_lock:target=self._elevator_gripper_target.copy()
            self._gripper_group.send_pos_vel(target)

        def _wait_until(self,predicate,timeout):
            cancelled=getattr(self,"_application_cancelled",None)
            if cancelled is None:return super()._wait_until(predicate,timeout)
            until=time.monotonic()+timeout
            while time.monotonic()<until:
                if cancelled():raise MotionCancelled("用户取消夹爪动作")
                if predicate():return True
                time.sleep(.01)
            if cancelled():raise MotionCancelled("用户取消夹爪动作")
            return predicate()

    return ManagedArm, CheckedDriver


def make_stack(cfg):
    """Original SDK/driver, with application-local startup/shutdown fixes."""
    from drivers.robot.grasp_driver import selected_arm_config
    robot = cfg.get("robot") or {}
    RebotArm, GraspDriver = sdk_types(robot.get("repo_root"))
    from reBotArm_control_py.controllers import RebotArmEndPose
    arm = RebotArm()
    controller = RebotArmEndPose(arm, arm_control_mode=selected_arm_config(robot.get("repo_root")).controller_mode)
    configure_trajectory_sampling(controller, cfg.get("trajectory", {}).get("sample_interval_s", .01))
    arm._application_controller = controller
    driver = GraspDriver(arm, controller, gripper_config=robot.get("gripper"), repo_root=robot.get("repo_root"))
    return arm, controller, driver


class Robot:
    def __init__(self, cfg, *, factory=make_stack):
        self.cfg, self.factory = cfg, factory
        self.arm = self.controller = self.driver = None
        self.started = False
        self._gripper_closed_target = None
        self._gripper_closed_tolerance = .01

    def start(self,*,gripper_posvel=True):
        self.arm, self.controller, self.driver = self.factory(self.cfg)
        self.driver._elevator_gripper_posvel=gripper_posvel
        self.driver.start()
        self.started = True
        print("[Robot] 原SDK move_to_traj；夹爪="+("POS_VEL" if gripper_posvel else "原GraspDriver MIT/力控")+
              "，启动应答已检查", flush=True)

    def switch_gripper_to_posvel(self,cancelled=lambda:False):
        """After release: switch only the gripper, keeping the same connected arm."""
        if not self.started:raise RuntimeError("机器人尚未启动")
        if getattr(self.driver,"_elevator_gripper_posvel",False):return
        self._check_cancel(cancelled)
        sender=getattr(self.controller,"_send_thread",None)
        if sender is not None and sender.is_alive():raise RuntimeError("轨迹未结束，不能切换夹爪模式")
        try:
            self.arm.stop_control_loop()
            if self.arm.control_loop_active:raise RuntimeError("原SDK回调未停止，不能混发夹爪模式")
            self.driver._prime_arm_target()
            motor=self.driver._gripper_motor
            motor.disable()
            jc=self.driver._gripper_group._jcfgs[0]
            for rid,value in ((25,jc.vel_kp),(26,jc.vel_ki),(27,jc.pos_kp),(28,jc.pos_ki)):
                motor.write_register_f32(rid,value);time.sleep(.01)
            motor.ensure_mode(2,1000)
            motor.request_feedback();time.sleep(.05)
            raw=motor.get_state()
            if raw is None or not np.isfinite([raw.pos,raw.vel,raw.torq]).all():
                raise RuntimeError("模式切换后夹爪反馈无效")
            with self.driver._state_lock:
                self.driver._state=self.driver._STATE_IDLE
                self.driver._elevator_gripper_target=np.array([raw.pos],float)
                self.driver._elevator_gripper_posvel=True
            self.driver._gripper_group._mode="pos_vel"
            motor.enable()
            self.arm.start_control_loop(self.driver._loop_cb)
            print("[Sequence] 已松爪；夹爪MIT切换为POS_VEL，六轴保持当前位置，串口未断开",flush=True)
        except BaseException:
            try:self.arm.disconnect()
            except BaseException as exc:print(f"[Exit] {exc}",flush=True)
            self.started=False;self.controller._running=False
            raise
        self._check_cancel(cancelled)

    def pose(self):
        T = np.asarray(self.driver.get_tcp_pose(), float)
        if T.shape != (4,4) or not np.isfinite(T).all():
            raise RuntimeError("SDK末端反馈无效")
        return T

    def _check_cancel(self, cancelled):
        if cancelled():
            self.stop()
            raise MotionCancelled("用户停止；本轮不自动回零")

    def report_gripper(self, phase):
        """One cached-state read, no arrival wait or extra control command."""
        if self._gripper_closed_target is None:
            return None
        with self.driver._state_lock:
            target = float(self.driver._elevator_gripper_target[0])
        row = dict(phase=phase, target_deg=float(np.degrees(target)),
                   closed_reference_deg=float(np.degrees(self._gripper_closed_target)),
                   actual_deg=None, feedback_packet_time_available=False)
        try:
            raw = self.driver._gripper_motor.get_state()
            if raw is not None and np.isfinite([raw.pos, raw.vel, raw.torq]).all():
                error = float(raw.pos-self._gripper_closed_target)
                row.update(actual_deg=float(np.degrees(raw.pos)), error_deg=float(np.degrees(error)),
                           velocity_rad_s=float(raw.vel), torque=float(raw.torq),
                           angle_matches_reference=bool(abs(error)<=self._gripper_closed_tolerance))
            else:
                row["feedback_error"] = "NO_VALID_STATE"
        except Exception as exc:
            row["feedback_error"] = str(exc)
        print("[Gripper] "+json.dumps(row,ensure_ascii=False),flush=True)
        return row

    def head_up(self, joints_deg, duration=3., cancelled=lambda: False,closed_tolerance=.01,wait_for_arrival=True):
        """Six arm angles, plus optional measured closed angle, as in example/4."""
        if not isinstance(wait_for_arrival,bool):
            raise ValueError("wait_for_arrival必须为布尔值")
        values=np.asarray(joints_deg,float)
        if values.shape not in ((6,),(7,)) or not np.isfinite(values).all() or not np.isfinite(duration) or duration<=0:
            raise ValueError("抬头目标必须为六或七个有限角度")
        with_gripper=values.size==7
        if with_gripper and (not np.isfinite(closed_tolerance) or not 0<closed_tolerance<=.12):
            raise ValueError("夹爪闭合容差需在(0,0.12]rad")
        goal = np.radians(values[:6])
        q = np.asarray(self.arm.get_state()[0][:6], float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError("六轴反馈无效")
        self._check_cancel(cancelled)
        motor=None
        if with_gripper:
            if not getattr(self.driver,"_elevator_gripper_posvel",False):
                raise RuntimeError("夹爪未配置为POS_VEL，拒绝混用MIT命令")
            motor=self.driver._gripper_motor
            raw=None if motor is None else motor.get_state()
            if raw is None or not np.isfinite([raw.pos,raw.vel,raw.torq]).all():
                raise RuntimeError("第七轴缺少有效反馈；尚未下发抬头目标")
        # One constant final target, as in the proven position-control example.
        self.controller._vlim_override = np.maximum(.01, np.abs(goal-q)/duration)
        self.controller._qd_target[:] = 0.
        self.controller._q_target[:] = goal
        if with_gripper:
            self._gripper_closed_target=float(np.radians(values[6]))
            self._gripper_closed_tolerance=closed_tolerance
            with self.driver._state_lock:self.driver._elevator_gripper_target[:]=self._gripper_closed_target
        print(f"[Head] {'七' if with_gripper else '六'}轴绝对目标deg={list(joints_deg)}，时间参考={duration:.1f}s；"
              +(f"夹爪POS_VEL闭合参考={values[6]:g}deg，无预张开" if with_gripper else "夹爪保持启动时角度"),flush=True)
        began=time.monotonic()
        until = began+duration+8.
        retain_head_limit=False
        gripper_stable_since=None;gripper_state=None;next_query=0.
        try:
            if not wait_for_arrival:
                # Keep the constant seven-axis target, without arrival polling.
                # Only the nominal motion window delays perception; exit keys
                # remain responsive. Do not claim measured arrival/closure.
                while True:
                    self._check_cancel(cancelled)
                    remaining=began+duration-time.monotonic()
                    if remaining<=0:break
                    time.sleep(min(.02,remaining))
                # An unfinished target must not jump to native (higher) vlim
                # just because arrival validation was disabled. move()/stop()
                # reset this override when a new action actually begins.
                retain_head_limit=True
                print(f"[Head:UNVERIFIED] 指令等待={time.monotonic()-began:.2f}s；七轴到位验证已关闭，进入识别；"
                      "保持已下发目标及抬头速度参数，未确认实际到位/夹爪闭合",flush=True)
                self.report_gripper("AFTER_HEAD")
                return
            while time.monotonic() < until:
                self._check_cancel(cancelled)
                actual = np.asarray(self.arm.get_state()[0][:6], float)
                if not np.isfinite(actual).all():
                    raise RuntimeError("抬头反馈无效")
                gripper_ready=not with_gripper
                if with_gripper:
                    now=time.monotonic()
                    if now>=next_query:motor.request_feedback();next_query=now+.05
                    raw=motor.get_state()
                    gripper_state=None if raw is None else np.array([raw.pos,raw.vel,raw.torq],float)
                    if gripper_state is not None and not np.isfinite(gripper_state).all():
                        raise RuntimeError("第七轴反馈含非有限值")
                    if gripper_state is not None and abs(gripper_state[0]-self._gripper_closed_target)<=closed_tolerance and abs(gripper_state[1])<=.05:
                        if gripper_stable_since is None:gripper_stable_since=now
                        gripper_ready=now-gripper_stable_since>=.2
                    else:gripper_stable_since=None
                if np.max(np.abs(actual-goal)) < np.radians(1.) and gripper_ready:
                    if with_gripper:
                        print(f"[Gripper:CLOSED_REFERENCE] target_deg={values[6]:g} actual_deg={np.degrees(gripper_state[0]):.4f} "
                              f"velocity_rad_s={gripper_state[1]:.5f} torque={gripper_state[2]:.4f} "
                              f"tolerance_rad={closed_tolerance:.4f}；持续POS_VEL保持，无反馈包时间戳",flush=True)
                    print("[Head] 六轴到位"+("，夹爪闭合参考角到位" if with_gripper else "")
                          +f"；指令及到位等待={time.monotonic()-began:.2f}s",flush=True)
                    return
                time.sleep(.02)
            raise TimeoutError(f"抬头/夹爪未到位，arm_remaining_deg={np.degrees(goal-actual).tolist()} "
                               f"gripper_pos_vel_torque={None if gripper_state is None else gripper_state.tolist()}；"
                               "已取消后续动作，未重设零点")
        except BaseException:
            self.stop()
            raise
        finally:
            if not retain_head_limit:self.controller._vlim_override = None

    def _prepare_move(self, cancelled):
        self._check_cancel(cancelled)
        if self._gripper_closed_target is not None:
            # An explicit new action after a stop resumes the configured closure;
            # an emergency stop itself still holds the actual angle.
            with self.driver._state_lock:
                self.driver._elevator_gripper_target[:]=self._gripper_closed_target
            self.report_gripper("BEFORE_MOVE")
        self.controller._vlim_override = None  # SDK native settings, exactly as main.py.

    def move(self, target, duration=3., cancelled=lambda: False):
        T = np.asarray(target, float)
        if T.shape != (4,4) or not np.isfinite(T).all() or not np.isfinite(duration) or duration <= 0:
            raise ValueError("运动目标/参考时间无效")
        self._prepare_move(cancelled)
        pose = tuple(float(v) for v in mat4_to_pose6d(T))
        began = time.monotonic()
        try:
            if not self.controller.move_to_traj(*pose, duration=float(duration)):
                # A cancel key queued during synchronous IK takes precedence
                # over automatic homing after a planning rejection.
                self._check_cancel(cancelled)
                sender = getattr(self.controller, '_send_thread', None)
                if not self.started or not self.controller._running or sender is not None and sender.is_alive():
                    raise RuntimeError('SDK规划返回失败且控制器/发送线程状态异常，取消当前动作')
                raise TrajectoryPlanningError(T, duration)
            planned = time.monotonic()
            print(f"[Plan] SDK规划={planned-began:.2f}s；开始发送参考{duration:.2f}s轨迹",flush=True)
            self._wait_move(T,duration,cancelled,began,planned)
        except BaseException:
            self.stop()
            raise

    def move_position(self, tip_target, duration=3., cancelled=lambda:False, *, tip_offset=(0.,0.,0.)):
        """Free orientation at every IK knot; reuse the original SDK sender/loop."""
        tip_target,tip_offset=np.asarray(tip_target,float),np.asarray(tip_offset,float)
        if (tip_target.shape!=(3,) or tip_offset.shape!=(3,) or not np.isfinite(tip_target).all()
                or not np.isfinite(tip_offset).all() or not np.isfinite(duration) or duration<=0):
            raise ValueError('位置IK需要三个有限XYZ、工具偏移和正运动时间')
        if not self.started or self.controller is None or not self.controller._running:
            raise RuntimeError('位置IK运动要求已启动的控制器')
        self._prepare_move(cancelled)
        began = time.monotonic()
        try:
            previous = getattr(self.controller,'_send_thread',None)
            if previous is not None and previous.is_alive():raise RuntimeError('上一轨迹仍在发送，拒绝重叠位置IK运动')
            plan = plan_position_trajectory(self.controller,self.arm.get_state()[0],tip_target,duration,tip_offset,cancelled)
            self._check_cancel(cancelled)
            self.controller._stop_send.set()
            self.controller._traj = plan['points']
            self.controller._moving = True
            self.controller._stop_send.clear()
            self.controller._send_thread = threading.Thread(target=self.controller._send_loop,args=(duration,),daemon=True)
            self.controller._send_thread.start()
            planned = time.monotonic()
            self.last_position_plan = plan['report']
            print('[PositionPlan] '+json.dumps(dict(plan['report'],tip_target_base_m=plan['tip_target'].tolist(),
                                                  planning_s=planned-began),ensure_ascii=False),flush=True)
            self._wait_move(plan['end_pose'],duration,cancelled,began,planned,
                            tip_target=plan['tip_target'],tip_offset=plan['tip_offset'])
            return plan['end_pose']
        except BaseException:
            self.stop()
            raise

    def _wait_move(self, T, duration, cancelled, began, planned, *, tip_target=None, tip_offset=None):
        thread = getattr(self.controller,'_send_thread',None)
        until = time.monotonic()+duration+8.
        while thread is not None and thread.is_alive():
            self._check_cancel(cancelled)
            if time.monotonic()>=until:raise TimeoutError('SDK轨迹发送未结束')
            thread.join(timeout=.02)
        sent = time.monotonic()
        until = time.monotonic()+3.
        while time.monotonic()<until:
            self._check_cancel(cancelled)
            actual = self.pose()
            if tip_target is None:
                error = actual[:3,3]-T[:3,3]
                angle = np.arccos(np.clip((np.trace(actual[:3,:3].T@T[:3,:3])-1)/2,-1,1))
                reached = np.linalg.norm(error)<=.01 and angle<=np.radians(3.)
            else:
                error = actual[:3,3]+actual[:3,:3]@tip_offset-tip_target
                reached = np.linalg.norm(error)<=.01
            if reached:
                finished = time.monotonic()
                print(f"[Move] SDK参考={duration:.2f}s，规划={planned-began:.2f}s "
                      f"发送等待={sent-planned:.2f}s 到位等待={finished-sent:.2f}s 总计={finished-began:.2f}s；"
                      f"mode={'POSE' if tip_target is None else 'FREE_POSITION_IK'} "
                      f"end_target_m={T[:3,3].tolist()} end_actual_m={actual[:3,3].tolist()} "
                      f"model_error_mm={(error*1000).round(2).tolist()}",flush=True)
                self.report_gripper('AFTER_MOVE');return
            time.sleep(.02)
        raise TimeoutError('SDK轨迹结束后末端未到位')

    def stop(self, *, gripper_target=None):
        c = self.controller
        if c is None:
            return
        c._stop_send.set()
        thread = getattr(c, "_send_thread", None)
        if thread is not None:
            thread.join(timeout=.5)
            if thread.is_alive():
                self.arm.disconnect()
                self.started = False
                raise RuntimeError("SDK发送线程未停止，已断开")
        try:
            q = np.asarray(self.arm.get_state()[0][:6],float)
            if not np.isfinite(q).all():
                raise RuntimeError("停止时反馈无效")
            c._q_target[:] = q
            c._qd_target[:] = 0.
            c._vlim_override = None
            if self.driver is not None:
                with self.driver._state_lock:
                    self.driver._state = self.driver._STATE_IDLE
                motor = self.driver._gripper_motor
                if motor is not None and getattr(self.driver,"_elevator_gripper_posvel",False):
                    if gripper_target is None:
                        raw=motor.get_state()
                        if raw is None or not np.isfinite(raw.pos):raise RuntimeError("停止时夹爪反馈无效")
                        gripper_target=float(raw.pos)
                    if not np.isfinite(gripper_target):raise ValueError("夹爪保持目标无效")
                    with self.driver._state_lock:
                        self.driver._elevator_gripper_target[:]=gripper_target
                        target=self.driver._elevator_gripper_target.copy()
                    self.driver._gripper_group.send_pos_vel(target)
                elif motor is not None:
                    motor.send_mit(0.,0.,0.,.5,0.)
        except BaseException:
            self.arm.disconnect()
            self.started = False
            raise

    def close(self, home=False):
        if self.arm is None:
            return
        try:
            if self.started:
                self.stop(gripper_target=self._gripper_closed_target if home else None)
            if home and self.started and self.controller._running:
                print("[Exit] SDK safe_home 六轴回零；夹爪保留闭合参考目标，不重设编码器零点",flush=True)
                # SDK end() immediately disconnects after safe_home(). Use the
                # same home, then report the gripper before final disconnection.
                self.controller.safe_home()
                self.report_gripper("AFTER_HOME_BEFORE_DISABLE")
        finally:
            try:
                self.arm.disconnect()
            finally:
                if self.controller is not None:self.controller._running=False
                self.started = False
