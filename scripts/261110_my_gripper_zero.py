"""B601-DM 第七轴原始反馈/单电机设零；默认只读，不加载整臂控制器。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

MOTOR_ID, FEEDBACK_ID, MODEL = 7, 0x17, "4310"


def read_state(bus, motor):
    # Native MotorState has no receive timestamp. Never claim packet freshness.
    for _ in range(10):
        motor.request_feedback()
        time.sleep(.05)
        bus.poll_feedback_once()
        state = motor.get_state()
        if state is None:
            continue
        if (state.can_id, state.arbitration_id) != (MOTOR_ID, FEEDBACK_ID):
            raise RuntimeError("反馈ID不是夹爪7/23；拒绝继续")
        if not all(math.isfinite(x) for x in (state.pos, state.vel, state.torq)):
            raise RuntimeError("夹爪反馈包含非有限值")
        return dict(position_deg=math.degrees(state.pos), velocity_rad_s=state.vel,
                    torque=state.torq, status=state.status_code)
    raise RuntimeError("夹爪无反馈；未把缺失反馈当作零位")


def run(port, set_zero, report, *, controller_factory=None, confirm=input):
    """Only ID 7 is registered. No enable, mode change or motion command exists here."""
    record = dict(port=port, motor_id=MOTOR_ID, feedback_id=FEEDBACK_ID, model=MODEL,
                  operation="set_zero" if set_zero else "read_only",
                  zero_command_attempted=False, packet_freshness_verified=False)
    # Create the audit before any device action; do not overwrite an old report.
    report = Path(report)
    with report.open("x", encoding="utf-8") as log:
        def save():
            log.seek(0)
            json.dump(record, log, ensure_ascii=False, indent=2, allow_nan=False)
            log.truncate()
            log.flush()

        save()
        bus = motor = None
        try:
            if controller_factory is None:
                from motorbridge import Controller
                controller_factory = Controller.from_dm_serial
            bus = controller_factory(serial_port=port, baud=921600)
            motor = bus.add_damiao_motor(MOTOR_ID, FEEDBACK_ID, MODEL)
            ids = {label: motor.get_register_u32(rid, 500)
                   for rid, label in ((7, "MST_ID"), (8, "ESC_ID"))}
            record["ids"] = ids
            if ids != {"MST_ID": FEEDBACK_ID, "ESC_ID": MOTOR_ID}:
                raise RuntimeError(f"夹爪寄存器ID不匹配：{ids}；拒绝写入")
            record["before"] = read_state(bus, motor)
            print("[GripperRaw] " + json.dumps(record["before"], ensure_ascii=False), flush=True)
            if not set_zero:
                record["result"] = "READ_ONLY"
                return 0

            motor.disable()
            for _ in range(10):
                disabled = read_state(bus, motor)
                if disabled["status"] == 0:
                    break
            else:
                raise RuntimeError("夹爪失能未确认；未写零点")
            record["disabled"] = disabled
            save()
            print("仅夹爪已失能；本工具不改变六轴使能。请先退出其他控制进程并支撑机械臂。\n"
                  "将双指轻轻合拢至真实闭合位置，不强压机构。保持此位置，手离开夹缝。\n"
                  "下一步会将当前位置写成第七电机零点，不会自动驱动夹爪闭合。", flush=True)
            if confirm("确认物理闭合并仅重设第七轴，请输入 CLOSED；其他输入取消：").strip() != "CLOSED":
                record["result"] = "CANCELLED_NO_ZERO_WRITE"
                return 1
            samples = [read_state(bus, motor) for _ in range(5)]
            positions = [s["position_deg"] for s in samples]
            record["closed_before_zero"] = samples
            if (any(s["status"] != 0 or abs(s["velocity_rad_s"]) > .05 for s in samples)
                    or max(positions) - min(positions) > .2):
                raise RuntimeError("闭合参考采样未保持失能/静止；未写零点")

            # A failed API call can still have transmitted the write. Log that
            # uncertainty first; do not retry or roll back hardware calibration.
            record["zero_command_attempted"] = True
            save()
            motor.set_zero_position()
            time.sleep(.2)
            near_zero = 0
            for _ in range(20):
                after = read_state(bus, motor)
                record["after"] = after
                near_zero = (near_zero + 1 if after["status"] == 0
                             and abs(after["position_deg"]) <= .5
                             and abs(after["velocity_rad_s"]) <= .05 else 0)
                if near_zero >= 3:
                    record["result"] = "ZERO_READBACK_OK"
                    print("[GripperZero] ZERO_READBACK_OK " + json.dumps(after, ensure_ascii=False), flush=True)
                    print("硬件设零后，运行电梯前把 config/261105_my_elevator.yaml 中 "
                          "gripper.closed_angle_deg 从10.0改回0.0。\n"
                          "保持真实闭合位置，断电重启后用默认只读命令复核；本次不证明掉电保存或实际闭合。",
                          flush=True)
                    return 0
            raise RuntimeError("设零命令已发送，但未确认零角读回；不要重复发送或继续+10度补偿")
        except BaseException as exc:
            record.update(result="ERROR", error=f"{type(exc).__name__}: {exc}")
            if record["zero_command_attempted"]:
                print("[GripperZero] 可能已改变电机零点；先只读复核，勿继续旧+10度目标。", file=sys.stderr)
            raise
        finally:
            errors = []
            # close_bus only closes transport; shutdown() would send disables.
            # Free the motor handle as well so the serial port is released.
            for label, fn in (("close_bus", None if bus is None else bus.close_bus),
                              ("motor.close", None if motor is None else motor.close),
                              ("controller.close", None if bus is None else bus.close)):
                if fn is not None:
                    try:
                        fn()
                    except BaseException as exc:
                        errors.append(f"{label}: {exc}")
            if errors:
                record["cleanup_errors"] = errors
            save()
            print(f"[Report] {report}", flush=True)
            if errors:
                raise RuntimeError("串口资源清理异常：" + "; ".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--set-zero", action="store_true",
                        help="仅失能第七轴，人工闭合确认后写零点；默认不写")
    parser.add_argument("--report", type=Path, default=Path("/tmp") /
                        (datetime.now().strftime("%y%m%d_%H%M%S_%f") + "_gripper_zero.json"))
    args = parser.parse_args(argv)
    if args.set_zero and not sys.stdin.isatty():
        parser.error("--set-zero需要交互终端人工确认物理闭合；拒绝管道自动确认")
    try:
        return run(args.port, args.set_zero, args.report)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
