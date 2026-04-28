"""Tests for the attitude controller (velocity NED → T/R/P/Y).

Validates:
  1. Hover hold — zero velocity demand at hover produces stable output
  2. Forward flight — positive vn demand produces nose-down pitch
  3. Lateral flight — positive ve demand produces right roll
  4. Altitude — upward demand (negative vd) increases throttle
  5. Yaw tracking — heading error drives yaw command toward target
  6. Body-frame rotation — NED errors are correctly rotated for non-zero yaw
  7. Saturation — outputs stay within configured limits
  8. Reset — clearing state zeroes integrators
  9. Convergence — repeated ticks drive velocity error toward zero
"""

import math
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from control.attitude_controller import (
    AttitudeCommand,
    AttitudeController,
    PIDGains,
    PIDState,
    _pid_step,
    _wrap_180,
)


# ── Helpers ──────────────────────────────────────────────────────────

DT = 0.02  # 50 Hz


def hover_state():
    """Drone hovering: zero velocity, heading north."""
    return dict(
        current_vel_ned=(0.0, 0.0, 0.0),
        current_yaw_deg=0.0,
        dt=DT,
    )


# ── Unit: _wrap_180 ─────────────────────────────────────────────────

class TestWrap180:
    def test_zero(self):
        assert _wrap_180(0.0) == 0.0

    def test_positive_in_range(self):
        assert _wrap_180(90.0) == 90.0

    def test_negative_in_range(self):
        assert _wrap_180(-90.0) == pytest.approx(-90.0)

    def test_wrap_270(self):
        assert _wrap_180(270.0) == pytest.approx(-90.0)

    def test_wrap_negative_270(self):
        assert _wrap_180(-270.0) == pytest.approx(90.0)

    def test_wrap_360(self):
        assert _wrap_180(360.0) == pytest.approx(0.0)

    def test_wrap_540(self):
        assert _wrap_180(540.0) == pytest.approx(180.0) or _wrap_180(540.0) == pytest.approx(-180.0)


# ── Unit: _pid_step ─────────────────────────────────────────────────

class TestPIDStep:
    def test_proportional_only(self):
        gains = PIDGains(kp=2.0, ki=0.0, kd=0.0)
        state = PIDState()
        out = _pid_step(gains, state, error=5.0, dt=DT)
        assert out == pytest.approx(10.0)

    def test_integral_accumulates(self):
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0, integral_limit=100.0)
        state = PIDState()
        _pid_step(gains, state, error=10.0, dt=DT)
        assert state.integral == pytest.approx(10.0 * DT)
        _pid_step(gains, state, error=10.0, dt=DT)
        assert state.integral == pytest.approx(20.0 * DT)

    def test_integral_windup_clamp(self):
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.1)
        state = PIDState()
        for _ in range(1000):
            _pid_step(gains, state, error=100.0, dt=DT)
        assert state.integral == pytest.approx(0.1)

    def test_output_clamp(self):
        gains = PIDGains(kp=100.0, ki=0.0, kd=0.0, output_limit=5.0)
        state = PIDState()
        out = _pid_step(gains, state, error=10.0, dt=DT)
        assert out == pytest.approx(5.0)

    def test_derivative_term(self):
        gains = PIDGains(kp=0.0, ki=0.0, kd=1.0)
        state = PIDState()
        _pid_step(gains, state, error=0.0, dt=DT)   # seed prev_error
        out = _pid_step(gains, state, error=1.0, dt=DT)
        # d/dt(error) = (1.0 - 0.0) / 0.02 = 50.0
        assert out == pytest.approx(50.0)


# ── Integration: AttitudeController ─────────────────────────────────

class TestHover:
    """Hover: all demands zero, drone stationary → should hover stably."""

    def test_hover_throttle_near_nominal(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 0.0, 0.0, 0.0, **hover_state())
        assert cmd.throttle == pytest.approx(0.5, abs=0.05)

    def test_hover_pitch_near_zero(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 0.0, 0.0, 0.0, **hover_state())
        assert abs(cmd.pitch_deg) < 1.0

    def test_hover_roll_near_zero(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 0.0, 0.0, 0.0, **hover_state())
        assert abs(cmd.roll_deg) < 1.0


class TestForwardFlight:
    """Commanding forward velocity (positive vn) from hover."""

    def test_pitch_nose_down_for_forward(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(5.0, 0.0, 0.0, 0.0, **hover_state())
        # Need to accelerate north → pitch nose down (positive pitch)
        assert cmd.pitch_deg > 0.0

    def test_no_roll_for_pure_forward(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(5.0, 0.0, 0.0, 0.0, **hover_state())
        # No east demand → roll should stay near zero
        assert abs(cmd.roll_deg) < 1.0


class TestLateralFlight:
    """Commanding east velocity (positive ve) from hover."""

    def test_roll_right_for_east(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 5.0, 0.0, 0.0, **hover_state())
        # Need to accelerate east → roll right (positive roll)
        assert cmd.roll_deg > 0.0

    def test_no_pitch_for_pure_east(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 5.0, 0.0, 0.0, **hover_state())
        assert abs(cmd.pitch_deg) < 1.0


class TestAltitude:
    """Vertical velocity commands → throttle adjustment."""

    def test_climb_increases_throttle(self):
        ctrl = AttitudeController()
        # desired_vd = -1.0 (upward in NED), current_vd = 0.0
        cmd = ctrl.convert(0.0, 0.0, -1.0, 0.0, **hover_state())
        assert cmd.throttle > ctrl.hover_throttle

    def test_descend_decreases_throttle(self):
        ctrl = AttitudeController()
        # desired_vd = +1.0 (downward in NED), current_vd = 0.0
        cmd = ctrl.convert(0.0, 0.0, 1.0, 0.0, **hover_state())
        assert cmd.throttle < ctrl.hover_throttle

    def test_throttle_clamped(self):
        ctrl = AttitudeController()
        cmd = ctrl.convert(0.0, 0.0, -100.0, 0.0, **hover_state())
        assert cmd.throttle <= ctrl.max_throttle
        cmd = ctrl.convert(0.0, 0.0, 100.0, 0.0, **hover_state())
        assert cmd.throttle >= ctrl.min_throttle


class TestYawTracking:
    """Yaw heading commands."""

    def test_yaw_tracks_target(self):
        ctrl = AttitudeController()
        # Facing 0°, want to face 90°
        cmd = ctrl.convert(0.0, 0.0, 0.0, 90.0, **hover_state())
        # Yaw command should be moving toward 90
        assert cmd.yaw_deg > 0.0

    def test_yaw_shortest_path(self):
        ctrl = AttitudeController()
        # Facing 10°, want to face 350° → shortest path is -20° (turn left)
        cmd = ctrl.convert(
            0.0, 0.0, 0.0, 350.0,
            current_vel_ned=(0.0, 0.0, 0.0),
            current_yaw_deg=10.0,
            dt=DT,
        )
        # yaw_cmd should be < 10 (turning left toward 350)
        assert cmd.yaw_deg < 10.0


class TestBodyFrameRotation:
    """When the drone is yawed, NED errors should rotate into body frame."""

    def test_east_demand_at_yaw_90(self):
        """Drone facing east (yaw=90°). Demanding east velocity = body forward."""
        ctrl = AttitudeController()
        cmd = ctrl.convert(
            0.0, 5.0, 0.0, 90.0,
            current_vel_ned=(0.0, 0.0, 0.0),
            current_yaw_deg=90.0,
            dt=DT,
        )
        # East demand at yaw=90 is body-forward → expect pitch, not roll
        assert cmd.pitch_deg > 1.0
        assert abs(cmd.roll_deg) < 2.0

    def test_north_demand_at_yaw_90(self):
        """Drone facing east (yaw=90°). Demanding north velocity = body left."""
        ctrl = AttitudeController()
        cmd = ctrl.convert(
            5.0, 0.0, 0.0, 90.0,
            current_vel_ned=(0.0, 0.0, 0.0),
            current_yaw_deg=90.0,
            dt=DT,
        )
        # North demand at yaw=90 is body-left → expect negative roll (left roll)
        assert cmd.roll_deg < -1.0
        assert abs(cmd.pitch_deg) < 2.0


class TestSaturation:
    """Outputs must stay within configured limits."""

    def test_roll_limit(self):
        ctrl = AttitudeController(max_roll_deg=30.0)
        cmd = ctrl.convert(0.0, 100.0, 0.0, 0.0, **hover_state())
        assert abs(cmd.roll_deg) <= 30.0 + 0.01

    def test_pitch_limit(self):
        ctrl = AttitudeController(max_pitch_deg=25.0)
        cmd = ctrl.convert(100.0, 0.0, 0.0, 0.0, **hover_state())
        assert abs(cmd.pitch_deg) <= 25.0 + 0.01


class TestReset:
    """Resetting clears integrators."""

    def test_reset_zeroes_integrators(self):
        ctrl = AttitudeController()
        # Build up integrator state
        for _ in range(100):
            ctrl.convert(5.0, 5.0, -1.0, 90.0, **hover_state())
        ctrl.reset()
        assert ctrl._vn_state.integral == 0.0
        assert ctrl._ve_state.integral == 0.0
        assert ctrl._vd_state.integral == 0.0
        assert ctrl._yaw_state.integral == 0.0


class TestConvergence:
    """Simulated closed-loop: repeated ticks should drive error down."""

    def test_velocity_tracks_demand(self):
        """Simple 1D forward-flight convergence with mock plant dynamics."""
        ctrl = AttitudeController()
        # Simplified plant: pitch_deg → forward acceleration
        # a_forward ≈ g * tan(pitch_rad) ≈ g * pitch_rad for small angles
        g = 9.81
        vel_n = 0.0
        target_vn = 3.0
        yaw = 0.0

        for i in range(200):  # 4 seconds at 50 Hz
            cmd = ctrl.convert(
                target_vn, 0.0, 0.0, 0.0,
                current_vel_ned=(vel_n, 0.0, 0.0),
                current_yaw_deg=yaw,
                dt=DT,
            )
            # Simple plant model: forward accel from pitch
            accel = g * math.tan(math.radians(cmd.pitch_deg))
            vel_n += accel * DT

        # After 4 seconds, velocity should be close to target
        assert abs(vel_n - target_vn) < 1.0, f"vel_n={vel_n}, target={target_vn}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
