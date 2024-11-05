from __future__ import annotations

import math
import random
import threading
import time
from pprint import pprint

import numpy as np
import odrive
import odrive.enums
from odrive import enums as oenum


class Odrive:
    axis0: Axis
    axis1: Axis
    config: Config

    class Axis:
        config: Config
        motor: Motor
        encoder: Encoder
        max_endstop: Endstop
        min_endstop: Endstop
        controller: Controller
        current_state: oenum.AxisState
        requested_state: oenum.AxisState
        error: float
        active_errors: odrive.enums.AxisError
        is_homed: bool

        class Config:
            startup_motor_calibration: bool
            startup_encoder_index_search: bool
            startup_encoder_offset_calibration: bool
            startup_closed_loop_control: bool
            startup_homing: bool

        class Motor:
            config: Config
            is_calibrated: bool

            class Config:
                pole_pairs: int
                torque_constant: float
                motor_type: oenum.MotorType
                pre_calibrated: bool

        class Encoder:
            config: Config
            pos_estimate: float

            class Config:
                cpr: int
                mode: oenum.EncoderMode
                pre_calibrated: bool
                phase_offset: float
                direction: float
                use_index: bool

        class Endstop:
            config: Config

            class Config:
                gpio_num: int
                enabled: bool
                offset: float
                debounce_ms: int
                is_active_high: bool

        class Controller:
            config: Config
            input_pos: float
            input_vel: float

            class Config:
                vel_limit: float
                vel_limit_tolerance: float
                control_mode: oenum.ControlMode

    class Config:
        enable_brake_resistor: bool
        brake_resistance: int
        dc_max_negative_current: float
        gpio5_mode: oenum.GpioMode

    def save_configuration(self):
        pass

    def reboot(self):
        pass

    def clear_errors(self):
        pass


class OdriveAPI:
    _odrv: Odrive

    AXIS_0_MIN_POS: int = 0
    AXIS_0_MAX_POS: int = 10
    AXIS_1_MIN_POS: int = 0
    AXIS_1_MAX_POS: int = 10

    use_robot: bool

    def __init__(self, odrv, use_robot: bool = True):
        if isinstance(odrv, Odrive):
            raise ValueError(
                "The Odrive class is only used for type definitions and "
                "should not be used directly. Use odrive.find_any() "
                "to get the correct odrive object."
            )

        self._odrv = odrv
        self.use_robot = use_robot
        self._speed_mode = False
        self._position_mode = False

        watchdog_thread = threading.Thread(
            target=OdriveAPI.error_watchdog, args=(self,), daemon=True
        )
        watchdog_thread.start()

    def reset_modes(self):
        self._speed_mode = False
        self._position_mode = False

    @property
    def curr_xy(self) -> np.ndarray:
        if self.use_robot:
            return np.array(
                [
                    self._odrv.axis0.encoder.pos_estimate,
                    self._odrv.axis1.encoder.pos_estimate,
                ]
            )
        else:
            return np.array([0, 0])

    def default_config(self):
        if not self.use_robot:
            return

        # disable startup actions
        self._odrv.axis0.config.startup_motor_calibration = False
        self._odrv.axis1.config.startup_motor_calibration = False

        self._odrv.axis0.config.startup_encoder_index_search = False
        self._odrv.axis1.config.startup_encoder_index_search = False

        self._odrv.axis0.config.startup_encoder_offset_calibration = False
        self._odrv.axis1.config.startup_encoder_offset_calibration = False

        self._odrv.axis0.config.startup_closed_loop_control = False
        self._odrv.axis1.config.startup_closed_loop_control = False

        self._odrv.axis0.config.startup_homing = False
        self._odrv.axis1.config.startup_homing = False

        # enable the brake resistor
        self._odrv.config.enable_brake_resistor = True

        # resistance of the brake resistor in Ohms
        self._odrv.config.brake_resistance = 2

        # his is the amount of current [Amps] allowed to flow back into the power
        # supply. The convention is that it is negative. By default, it is set
        # to a conservative value of 10mA. If you are using a brake resistor
        # and getting DC_BUS_OVER_REGEN_CURRENT errors, raise it slightly.
        # If you are not using a brake resistor and you intend to send braking
        # current back to the power supply, set this to a safe level for
        # your power source.
        self._odrv.config.dc_max_negative_current = -0.010  # default of 10mA

        # This is the number of magnet poles in the rotor, divided by two.
        # To find this, you can simply count the number of permanent magnets
        # in the rotor, if you can see them.
        self._odrv.axis0.motor.config.pole_pairs = 7
        self._odrv.axis1.motor.config.pole_pairs = 7

        # This is the ratio of torque produced by the motor per Amp of current
        # delivered to the motor. This should be set to 8.27 / (motor KV).
        # If you decide that you would rather command torque in units of Amps,
        # you could simply set the torque constant to 1.
        self._odrv.axis0.motor.config.torque_constant = 8.27 / 270
        self._odrv.axis1.motor.config.torque_constant = 8.27 / 270

        self._odrv.axis0.motor.config.motor_type = oenum.MotorType.HIGH_CURRENT
        self._odrv.axis1.motor.config.motor_type = oenum.MotorType.HIGH_CURRENT

        # This is 4x the Pulse Per Revolution (PPR) value
        self._odrv.axis0.encoder.config.cpr = 8192
        self._odrv.axis1.encoder.config.cpr = 8192
        # encoder type
        self._odrv.axis0.encoder.config.mode = oenum.EncoderMode.INCREMENTAL
        self._odrv.axis1.encoder.config.mode = oenum.EncoderMode.INCREMENTAL

        # the GPIO pin number, according to the silkscreen labels on ODrive
        self._odrv.axis0.max_endstop.config.gpio_num = 7
        self._odrv.axis0.min_endstop.config.gpio_num = 8

        self._odrv.axis1.max_endstop.config.gpio_num = 6
        self._odrv.axis1.min_endstop.config.gpio_num = 5

        self._odrv.config.gpio5_mode = oenum.GpioMode.DIGITAL_PULL_DOWN
        self._odrv.config.gpio5_mode = oenum.GpioMode.DIGITAL_PULL_DOWN
        self._odrv.config.gpio5_mode = oenum.GpioMode.DIGITAL_PULL_DOWN
        self._odrv.config.gpio5_mode = oenum.GpioMode.DIGITAL_PULL_DOWN

        # enable/disable detection of the endstop. If disabled, homing and e-stop cannot take place
        self._odrv.axis0.max_endstop.config.enabled = True
        self._odrv.axis0.min_endstop.config.enabled = True

        self._odrv.axis1.max_endstop.config.enabled = True
        self._odrv.axis1.min_endstop.config.enabled = True

        # this is the position of the endstops on the relevant axis, in turns
        self._odrv.axis0.min_endstop.config.offset = 0.1
        self._odrv.axis1.min_endstop.config.offset = 0.1

        # the debouncing time for this endstop. Most switches exhibit some sort of
        # bounce, and this setting will help prevent the switch from
        # triggering repeatedly. It works for both HIGH and LOW transitions,
        # regardless of the setting of is_active_high. Debouncing isP a
        # good practice for digital inputs, read up on it here.
        # debounce_ms has units of miliseconds.
        self._odrv.axis0.max_endstop.config.debounce_ms = 10
        self._odrv.axis0.min_endstop.config.debounce_ms = 10

        self._odrv.axis1.max_endstop.config.debounce_ms = 10
        self._odrv.axis1.min_endstop.config.debounce_ms = 10

        # NPN or PNP mode
        self._odrv.axis0.max_endstop.config.is_active_high = False
        self._odrv.axis0.min_endstop.config.is_active_high = False
        self._odrv.axis1.max_endstop.config.is_active_high = False
        self._odrv.axis1.min_endstop.config.is_active_high = False

        # velocity limit in
        self._odrv.axis0.controller.config.vel_limit = 10
        self._odrv.axis1.controller.config.vel_limit = 10
        self._odrv.axis0.controller.config.vel_limit_tolerance = 1.5
        self._odrv.axis0.controller.config.vel_limit_tolerance = 1.5

        # save configuration and reboot
        self._odrv.save_configuration()
        self._odrv.reboot()

    def calibrate_encoders(self, use_index_signal: bool = True):
        """
        Calibrate odrive encoders. This calibration only needs to be run once.
        Before running encoder calibration, make sure the motors are
        disengaged from the belt and can turn freely without any load.

        Parameters:
                use_index_signgal (bool): index signal is a faster way to
                calibrate encoders but may not be available on all odrives.
        """
        if not self.use_robot:
            return

        self._odrv.clear_errors()
        time.sleep(0.1)
        odrive.utils.dump_errors(self._odrv)

        print("this calibration only needs to be run once.")
        print(
            "before running encoder calibration, make sure the motors are disengaged from the belt and can turn freely without any load."
        )
        input("press ENTER to confirm")

        # axis 0 and 1 calibration
        if not self._odrv.axis0.motor.is_calibrated:
            self._odrv.axis0.requested_state = oenum.AxisState.FULL_CALIBRATION_SEQUENCE
            self.wait_for_idle()
        if not self._odrv.axis1.motor.is_calibrated:
            self._odrv.axis1.requested_state = oenum.AxisState.FULL_CALIBRATION_SEQUENCE
            self.wait_for_idle()

        print("running encoder search")
        input("press ENTER to confirm")

        if not use_index_signal:
            self._odrv.axis0.encoder.config.use_index = True
            self._odrv.axis1.encoder.config.use_index = True
            time.sleep(0.1)

            self._odrv.axis0.requested_state = oenum.AxisState.ENCODER_INDEX_SEARCH
            self._odrv.axis1.requested_state = oenum.AxisState.ENCODER_INDEX_SEARCH
            self.wait_for_idle()

        self._odrv.axis0.requested_state = oenum.AxisState.ENCODER_OFFSET_CALIBRATION
        self.wait_for_idle()

        if self._odrv.axis0.error != 0:
            odrive.utils.dump_errors(self._odrv)
            print(
                f"calibration failed. axis0 had error (error code: {self._odrv.axis0.error})"
            )
            return 1

        self._odrv.axis1.requested_state = oenum.AxisState.ENCODER_OFFSET_CALIBRATION
        self.wait_for_idle()

        if self._odrv.axis1.error != 0:
            odrive.utils.dump_errors(self._odrv)
            print(
                f"calibration failed. axis1 had error (error code: {self._odrv.axis1.error})"
            )
            return 1

        try:
            float(self._odrv.axis0.encoder.config.phase_offset)
        except ValueError:
            print(
                f"calibration failed. invalid phase offset for axis0 (got offset: {self._odrv.axis0.encoder.config.phase_offset})"
            )
            return 1

        try:
            float(self._odrv.axis1.encoder.config.phase_offset)
        except ValueError:
            print(
                f"calibration failed. invalid phase offset for axis1 (got offset: {self._odrv.axis1.encoder.config.phase_offset})"
            )
            return 1

        try:
            direction = float(self._odrv.axis0.encoder.config.direction)
            if direction not in [-1, 1]:
                raise ValueError
        except ValueError:
            print(
                f"calibration failed. invalid axis direction for axis0 (got direction: {self._odrv.axis0.encoder.config.direction})"
            )
            return 1

        try:
            direction = float(self._odrv.axis1.encoder.config.direction)
            if direction not in [-1, 1]:
                raise ValueError
        except ValueError:
            print(
                f"calibration failed. invalid axis direction for axis0 (got direction: {self._odrv.axis1.encoder.config.direction})"
            )
            return 1

        print("index calibration done. finishing up.")
        input("press ENTER to confirm")

        self._odrv.axis0.encoder.config.pre_calibrated = True
        self._odrv.axis1.encoder.config.pre_calibrated = True
        self._odrv.axis0.motor.config.pre_calibrated = True
        self._odrv.axis1.motor.config.pre_calibrated = True

        time.sleep(0.1)

        is_calibrated0 = self._odrv.axis0.encoder.config.pre_calibrated
        is_calibrated1 = self._odrv.axis1.encoder.config.pre_calibrated

        if not is_calibrated0:
            odrive.utils.dump_errors(self._odrv)
            print("axis0 calibration failed")
            return 1
        if not is_calibrated1:
            odrive.utils.dump_errors(self._odrv)
            print("axis1 calibration failed")
            return 1

        print("calibration done. saving configuration and rebooting")
        time.sleep(1)

        self._odrv.save_configuration()
        self._odrv.reboot()

    def wait_for_idle(self):
        if not self.use_robot:
            return

        while not (
            self._odrv.axis0.current_state == oenum.AxisState.IDLE
            and self._odrv.axis1.current_state == oenum.AxisState.IDLE
        ):
            time.sleep(0.5)

    def startup(self, homing: bool = True):
        if not self.use_robot:
            return

        self._odrv.clear_errors()
        time.sleep(0.1)

        if all([self._odrv.axis0.is_homed, self._odrv.axis1.is_homed]):
            return

        encoders_calibrated = (
            self._odrv.axis0.encoder.config.pre_calibrated,
            self._odrv.axis1.encoder.config.pre_calibrated,
        )
        motors_calibrated = (
            self._odrv.axis0.motor.config.pre_calibrated,
            self._odrv.axis1.motor.config.pre_calibrated,
        )

        if not all(encoders_calibrated) and not all(
            motors_calibrated
        ):  # nothing was pre-calibrated
            # axis 0 and 1 calibration
            print("running full calibration sequence")
            self._odrv.axis0.requested_state = oenum.AxisState.FULL_CALIBRATION_SEQUENCE
            self._odrv.axis1.requested_state = oenum.AxisState.FULL_CALIBRATION_SEQUENCE
            self.wait_for_idle()
        else:
            print("motors pre-calibrated, running reduced calibration sequence")
            # awis 0 and 1 motor calibration
            self._odrv.axis0.requested_state = oenum.AxisState.MOTOR_CALIBRATION
            self._odrv.axis1.requested_state = oenum.AxisState.MOTOR_CALIBRATION
            self.wait_for_idle()
            # awis 0 and 1 encoder search
            self._odrv.axis0.requested_state = oenum.AxisState.ENCODER_INDEX_SEARCH
            self.wait_for_idle()
            self._odrv.axis1.requested_state = oenum.AxisState.ENCODER_INDEX_SEARCH
            self.wait_for_idle()

        if homing:
            print("homing")
            # axis 0 homing
            self._odrv.axis0.requested_state = oenum.AxisState.HOMING
            self.wait_for_idle()
            # axis 1 homing
            self._odrv.axis1.requested_state = oenum.AxisState.HOMING
            self.wait_for_idle()

        # start closed loop control mode
        self._odrv.axis0.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL
        self._odrv.axis1.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL

        return 0

    def shutdown(self):
        if not self.use_robot:
            return

        self._odrv.axis0.requested_state = oenum.AxisState.IDLE
        self._odrv.axis1.requested_state = oenum.AxisState.IDLE

    def idle(self):
        if not self.use_robot:
            return

        self._odrv.axis0.requested_state = oenum.AxisState.IDLE
        self._odrv.axis1.requested_state = oenum.AxisState.IDLE

    def print_config(self):
        if not self.use_robot:
            return

        pprint(type(self._odrv.axis1.config))
        pprint(dir(self._odrv.axis1.config))

    def set_speed(self, x: float, y: float):
        if not self.use_robot:
            print(f"set_speed: {x}, {y}")
            return

        if not self._speed_mode:
            self._odrv.axis0.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL
            self._odrv.axis1.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL

            self._odrv.axis0.controller.config.control_mode = (
                oenum.ControlMode.VELOCITY_CONTROL
            )
            self._odrv.axis1.controller.config.control_mode = (
                oenum.ControlMode.VELOCITY_CONTROL
            )
            self.reset_modes()
            self._speed_mode = True

        self._odrv.axis0.controller.input_vel = x
        self._odrv.axis1.controller.input_vel = y

    def cartesian_move(
        self, x: float, y: float, vel: float, thresh: float = 0.008, wait: bool = True
    ):
        if not self.use_robot:
            print(f"cartesian_move: {x}, {y}, {vel}, {thresh}")
            return

        if x < self.AXIS_0_MIN_POS or x > self.AXIS_0_MAX_POS:
            raise ValueError(f"x={x} too big or too small")
        if y < self.AXIS_1_MIN_POS or y > self.AXIS_1_MAX_POS:
            raise ValueError(f"y={y} too big or too small")

        if not self._position_mode:
            self._odrv.axis0.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL
            self._odrv.axis1.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL

            self._odrv.axis0.controller.config.control_mode = (
                oenum.ControlMode.POSITION_CONTROL
            )
            self._odrv.axis1.controller.config.control_mode = (
                oenum.ControlMode.POSITION_CONTROL
            )

            self.reset_modes()
            self._position_mode = True
            print("changed modes")

        self._odrv.axis0.controller.config.vel_limit = vel
        self._odrv.axis1.controller.config.vel_limit = vel

        time.sleep(0.01)

        self._odrv.axis0.controller.input_pos = x
        self._odrv.axis1.controller.input_pos = y

        if wait:
            while (
                abs(self._odrv.axis0.encoder.pos_estimate - x) > thresh
                or abs(self._odrv.axis1.encoder.pos_estimate - y) > thresh
            ):
                time.sleep(0.001)

    def cartesian_move2(self, x: float, y: float, vel: float, thresh: float = 0.008):
        """
        Cartesian movement but the axes arrive at the same time
        """

        if not self.use_robot:
            print(f"cartesian_move2: {x}, {y}, {vel}, {thresh}")
            return

        if x < self.AXIS_0_MIN_POS or x > self.AXIS_0_MAX_POS:
            raise ValueError("x too big or too small")
        if y < self.AXIS_1_MIN_POS or y > self.AXIS_1_MAX_POS:
            raise ValueError("y too big or too small")

        if not self._position_mode:
            self._odrv.axis0.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL
            self._odrv.axis1.requested_state = oenum.AxisState.CLOSED_LOOP_CONTROL

            self._odrv.axis0.controller.config.control_mode = (
                oenum.ControlMode.POSITION_CONTROL
            )
            self._odrv.axis1.controller.config.control_mode = (
                oenum.ControlMode.POSITION_CONTROL
            )

            self.reset_modes()
            self._position_mode = True

        curr_x = self._odrv.axis0.encoder.pos_estimate
        curr_y = self._odrv.axis1.encoder.pos_estimate

        dx = x - curr_x
        dy = y - curr_y

        d = math.sqrt(dx**2 + dy**2)

        td = d / vel
        if td == 0:
            return

        vx = dx / td
        vy = dy / td

        self._odrv.axis0.controller.config.vel_limit = abs(vx)
        self._odrv.axis1.controller.config.vel_limit = abs(vy)

        # time.sleep(0.01)

        self._odrv.axis0.controller.input_pos = x
        self._odrv.axis1.controller.input_pos = y

        while (
            abs(self._odrv.axis0.encoder.pos_estimate - x) > thresh
            or abs(self._odrv.axis1.encoder.pos_estimate - y) > thresh
        ):
            time.sleep(0.001)

        self._odrv.axis0.controller.config.vel_limit = 5
        self._odrv.axis1.controller.config.vel_limit = 5

    def random_move(self, vel: float = 3, thresh: float = 0.008):
        rand_x = random.randint(self.AXIS_0_MIN_POS + 1, self.AXIS_0_MAX_POS - 1)
        rand_y = random.randint(self.AXIS_1_MIN_POS + 1, self.AXIS_1_MAX_POS - 1)

        self.cartesian_move2(rand_x, rand_y, vel, thresh)

    def error_watchdog(self):
        def dump_item(obj, path):
            for elem in path.split("."):
                if not hasattr(obj, elem):
                    return ""
                obj = getattr(obj, elem)

            return obj

        while True:
            ret = ""
            had_error = False

            if self._odrv.error:
                had_error = True
            ret += str(self._odrv.error) + "\n"  # odrive.enums.ODriveError

            for name, axis in [
                ("axis0", self._odrv.axis0),
                ("axis1", self._odrv.axis1),
            ]:
                axis_error = dump_item(axis, "error")
                axis_active_errors = dump_item(axis, "active_errors")
                axis_disarm_reason = dump_item(axis, "disarm_reason")
                axis_procedure_result = dump_item(axis, "procedure_result")
                axis_last_drv_fault = dump_item(axis, "last_drv_fault")

                non_errors = ["", 0, "0"]
                if axis_error not in non_errors:
                    had_error = True
                if axis_active_errors not in non_errors:
                    had_error = True
                if axis_disarm_reason not in non_errors:
                    had_error = True
                if axis_procedure_result not in non_errors:
                    had_error = True
                if axis_last_drv_fault not in non_errors:
                    had_error = True

                ret += f"  {name}: error={axis_error}\n"
                ret += f"  {name}: active_errors={axis_active_errors}\n"
                ret += f"  {name}: disarm_reason={axis_disarm_reason}\n"
                ret += f"  {name}: procedure_result={axis_procedure_result}\n"
                ret += f"  {name}: last_drv_fault={axis_last_drv_fault}\n"

            if (
                hasattr(self._odrv, "issues")
                and hasattr(self._odrv.issues, "length")
                and hasattr(self._odrv.issues, "get")
            ):
                issues = [
                    self._odrv.issues.get(i) for i in range(self._odrv.issues.length)
                ]
                if len(issues) == 0:
                    ret += "internal issues: none\n"
                else:
                    had_error = True
                    ret += f"internal issus: {str(len(issues))}\n"
                    ret += f"details for bug report: {str(issues)}\n"

            if had_error:
                print(ret)

            time.sleep(1)
