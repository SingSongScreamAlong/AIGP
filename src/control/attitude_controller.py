"""Attitude controller — converts desired velocity NED into T/R/P/Y commands.

This is the bridge between our existing navigator stack (which emits
VelocityNedYaw-shaped commands) and the DCL simulator (which accepts
Throttle, Roll, Pitch, Yaw stick inputs).

Architecture:
    Navigator.plan() → VelocityNedYaw (vn, ve, vd, yaw_deg)
        ↓
    AttitudeController.convert(desired_vel, current_state)
        ↓
    AttitudeCommand(throttle, roll_deg, pitch_deg, yaw_deg)
        ↓
    adapter.send_attitude(roll, pitch, yaw, thrust)

The controller uses a classical PID cascade:
    Outer loop: velocity error → desired attitude angles
    Altitude:   desired_vd error → throttle
    Yaw:        desired_yaw → yaw rate command

Coordinate conventions (NED, body-FRD):
    - Positive pitch → nose down → drone moves north (positive vn)
    - Positive roll  → right wing down → drone moves east (positive ve)
    - Throttle 0..1  → 0 = no thrust, 1 = max thrust
    - Yaw in degrees → heading reference

PID gains are tunable. Defaults are conservative starting points
designed for stability over speed — tune aggressively once we have
real DCL physics to test against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AttitudeCommand:
    """Output of the attitude controller — what gets sent to the adapter."""
    throttle: float     # 0.0 .. 1.0 (normalized thrust)
    roll_deg: float     # degrees, positive = right wing down
    pitch_deg: float    # degrees, positive = nose down (FRD convention)
    yaw_deg: float      # degrees, heading target


@dataclass
class PIDState:
    """Internal state for a single PID channel."""
    integral: float = 0.0
    prev_error: float = 0.0
    prev_output: float = 0.0


@dataclass
class PIDGains:
    """Gains for a single PID channel."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = float("inf")
    integral_limit: float = float("inf")


def _pid_step(
    gains: PIDGains, state: PIDState, error: float, dt: float
) -> float:
    """One PID tick. Returns control output and mutates state in-place."""
    # Proportional
    p_term = gains.kp * error

    # Integral with anti-windup clamp
    state.integral += error * dt
    state.integral = max(-gains.integral_limit,
                         min(gains.integral_limit, state.integral))
    i_term = gains.ki * state.integral

    # Derivative (on error, with zero-dt guard)
    if dt > 0:
        d_term = gains.kd * (error - state.prev_error) / dt
    else:
        d_term = 0.0

    output = p_term + i_term + d_term

    # Output clamp
    output = max(-gains.output_limit, min(gains.output_limit, output))

    state.prev_error = error
    state.prev_output = output
    return output


def _wrap_180(deg: float) -> float:
    """Wrap angle to [-180, 180) degrees."""
    deg = deg % 360.0
    if deg >= 180.0:
        deg -= 360.0
    return deg


class AttitudeController:
    """Converts desired velocity NED + yaw into T/R/P/Y commands.

    Usage:
        controller = AttitudeController()
        cmd = controller.convert(
            desired_vn=2.0, desired_ve=0.5, desired_vd=-0.1,
            desired_yaw_deg=45.0,
            current_vel_ned=(1.5, 0.3, 0.0),
            current_yaw_deg=40.0,
            dt=0.02,
        )
        # cmd.throttle, cmd.roll_deg, cmd.pitch_deg, cmd.yaw_deg
    """

    def __init__(
        self,
        # Velocity → attitude (outer loop)
        vel_north_gains: Optional[PIDGains] = None,
        vel_east_gains: Optional[PIDGains] = None,
        vel_down_gains: Optional[PIDGains] = None,
        yaw_gains: Optional[PIDGains] = None,
        # Attitude limits (safety)
        max_roll_deg: float = 35.0,
        max_pitch_deg: float = 35.0,
        # Throttle
        hover_throttle: float = 0.5,
        max_throttle: float = 1.0,
        min_throttle: float = 0.0,
    ):
        # Velocity-to-pitch PID (north velocity error → pitch angle)
        # Positive vn error → need nose down (positive pitch in FRD) to accelerate north
        self.vel_north_gains = vel_north_gains or PIDGains(
            kp=5.0, ki=0.5, kd=1.0,
            output_limit=max_pitch_deg,
            integral_limit=max_pitch_deg * 0.5,
        )

        # Velocity-to-roll PID (east velocity error → roll angle)
        # Positive ve error → need right roll (positive roll) to accelerate east
        self.vel_east_gains = vel_east_gains or PIDGains(
            kp=5.0, ki=0.5, kd=1.0,
            output_limit=max_roll_deg,
            integral_limit=max_roll_deg * 0.5,
        )

        # Vertical velocity → throttle offset PID
        # Positive vd means moving downward in NED; error = desired_vd - current_vd
        # Negative error (need to go up) → positive throttle delta
        self.vel_down_gains = vel_down_gains or PIDGains(
            kp=3.0, ki=1.0, kd=0.5,
            output_limit=0.4,          # max throttle offset from hover
            integral_limit=0.2,
        )

        # Yaw tracking PID (heading error → yaw rate/command)
        # Low kd to avoid derivative kick on first error step.
        self.yaw_gains = yaw_gains or PIDGains(
            kp=2.0, ki=0.1, kd=0.02,
            output_limit=90.0,         # max yaw correction in deg
            integral_limit=30.0,
        )

        self.max_roll_deg = max_roll_deg
        self.max_pitch_deg = max_pitch_deg
        self.hover_throttle = hover_throttle
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle

        # Per-channel PID states
        self._vn_state = PIDState()
        self._ve_state = PIDState()
        self._vd_state = PIDState()
        self._yaw_state = PIDState()

    def reset(self) -> None:
        """Clear all integrators and derivative state."""
        self._vn_state = PIDState()
        self._ve_state = PIDState()
        self._vd_state = PIDState()
        self._yaw_state = PIDState()

    def convert(
        self,
        desired_vn: float,
        desired_ve: float,
        desired_vd: float,
        desired_yaw_deg: float,
        current_vel_ned: Tuple[float, float, float],
        current_yaw_deg: float,
        dt: float,
    ) -> AttitudeCommand:
        """Convert a velocity NED command into an attitude command.

        Args:
            desired_vn/ve/vd: Target velocity in NED (m/s).
            desired_yaw_deg: Target heading (degrees).
            current_vel_ned: Current velocity (vn, ve, vd) in m/s.
            current_yaw_deg: Current heading (degrees).
            dt: Time step (seconds).

        Returns:
            AttitudeCommand with throttle/roll/pitch/yaw.
        """
        cvn, cve, cvd = current_vel_ned

        # ── Velocity errors (NED frame) ──────────────────────
        err_vn = desired_vn - cvn
        err_ve = desired_ve - cve
        err_vd = desired_vd - cvd

        # ── Rotate NED velocity errors into body frame ───────
        # The drone's body-forward axis is aligned with its yaw heading.
        # We need pitch/roll in the body frame, not NED.
        yaw_rad = math.radians(current_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Body-frame velocity errors:
        #   err_forward = err_vn * cos(yaw) + err_ve * sin(yaw)
        #   err_right   = -err_vn * sin(yaw) + err_ve * cos(yaw)
        err_forward = err_vn * cos_yaw + err_ve * sin_yaw
        err_right = -err_vn * sin_yaw + err_ve * cos_yaw

        # ── PID: forward error → pitch ───────────────────────
        # Forward error > 0 means we need to accelerate forward.
        # In the thrust model, positive pitch (nose down FRD) tilts
        # the thrust vector backward → negative forward accel. So
        # we NEGATE: forward demand → negative pitch (nose up tilts
        # thrust forward).
        pitch_deg = -_pid_step(
            self.vel_north_gains, self._vn_state, err_forward, dt
        )

        # ── PID: right error → roll ──────────────────────────
        # Right error > 0 means we need to accelerate rightward
        # → roll right wing down (positive roll)
        roll_deg = _pid_step(
            self.vel_east_gains, self._ve_state, err_right, dt
        )

        # ── PID: vertical velocity → throttle offset ─────────
        # err_vd < 0 means we want to move upward more (less positive vd
        # or more negative vd). Since in NED down is positive, wanting to
        # go up means desired_vd < current_vd → negative error.
        # Negative error → negative throttle_delta → but we NEGATE because
        # going up requires MORE throttle.
        throttle_delta = -_pid_step(
            self.vel_down_gains, self._vd_state, err_vd, dt
        )
        throttle = self.hover_throttle + throttle_delta
        throttle = max(self.min_throttle, min(self.max_throttle, throttle))

        # ── PID: yaw tracking ────────────────────────────────
        yaw_error = _wrap_180(desired_yaw_deg - current_yaw_deg)
        yaw_cmd = current_yaw_deg + _pid_step(
            self.yaw_gains, self._yaw_state, yaw_error, dt
        )

        # ── Clamp attitude angles ────────────────────────────
        roll_deg = max(-self.max_roll_deg, min(self.max_roll_deg, roll_deg))
        pitch_deg = max(-self.max_pitch_deg, min(self.max_pitch_deg, pitch_deg))

        return AttitudeCommand(
            throttle=throttle,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_cmd,
        )
