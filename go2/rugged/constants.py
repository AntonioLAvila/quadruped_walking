"""Constants specific to the rugged-terrain task.

Everything here is *derived* from ``go2.constants.GO2_ACTUATORS``, which stays the single
source of truth for actuator dynamics. Nothing actuator-related is hard-coded.

This module is stdlib-only for the same reason ``go2.constants`` is: the Drake sim-to-sim
check (``scripts/verify_rugged.py``) imports it from an environment that has pydrake but
neither mujoco nor mjlab.
"""

from __future__ import annotations

from dataclasses import dataclass

from go2.constants import FEET, GO2_ACTUATORS, JOINT_REGEX, JOINT_TYPES

##
# Control timing.
#
# 50 Hz, matching mjlab's own Go1 rough-terrain task and every Unitree deployment stack.
# The flat task's 200 Hz is a torque-control artefact; a PD position policy running at
# 200 Hz would be four times faster than anything that ships on hardware.
##


@dataclass(frozen=True)
class ControlTiming:
  sim_timestep: float
  decimation: int

  @property
  def ctrl_dt(self) -> float:
    return self.sim_timestep * self.decimation


POSITION_TIMING = ControlTiming(sim_timestep=0.005, decimation=4)  # 50 Hz

# Policy->motor bus latency, in seconds. Expressed as time rather than physics steps
# because this task's timestep differs from the flat task's.
COMMAND_DELAY_S = (0.0, 0.005)

##
# Observation history.
#
# The actor is blind, so history is what lets it infer terrain and contact state. This is
# THE knob to turn if the policy plateaus -- but changing it changes the ONNX input width,
# so it forces a retrain and a re-check of scripts/verify_rugged.py.
##

HISTORY_LENGTH = 5  # 5 frames @ 50 Hz = 100 ms

# Per-frame actor term widths, in the order env_cfg._actor_terms() declares them.
#
# THE LAYOUT CONTRACT. mjlab stacks history per term and concatenates afterwards, so the
# flat observation is TERM-MAJOR, oldest to newest:
#     [joint_pos t-4..t | joint_vel t-4..t | ang_vel | gravity | last_action | command]
# This is the opposite of legged_gym's time-major convention. Anything that rebuilds the
# observation outside mjlab -- the Drake check, and eventually the robot -- must match it
# exactly, and gets no error if it doesn't. ``scripts/check_obs_layout.py`` asserts it
# against the live environment.
ACTOR_TERM_WIDTHS = (
  ("joint_pos", 12),
  ("joint_vel", 12),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("last_action", 12),
  ("command", 3),
)
FRAME_DIM = sum(width for _, width in ACTOR_TERM_WIDTHS)
ACTOR_OBS_DIM = FRAME_DIM * HISTORY_LENGTH

##
# Actuators: PD position control.
##

# mjlab's convention for position-control action scale, used by every position-controlled
# asset in its zoo (see asset_zoo/robots/unitree_go1/go1_constants.py): an action of +/-1
# commands the position error that saturates 25% of the motor's torque limit.
#   hip/thigh: 0.25 * 23.5 / 20 = 0.294 rad,  calf: 0.25 * 45.0 / 40 = 0.281 rad
POSITION_ACTION_SCALE = {
  JOINT_REGEX[jt]: 0.25 * GO2_ACTUATORS[jt].effort_limit / GO2_ACTUATORS[jt].kp
  for jt in JOINT_TYPES
}
POSITION_ACTION_SCALE_FLAT = tuple(
  0.25 * GO2_ACTUATORS[jt].effort_limit / GO2_ACTUATORS[jt].kp
  for _ in FEET
  for jt in JOINT_TYPES
)

# Passive viscous damping when the joint is PD-position controlled.
#
# NOT GO2_ACTUATORS[jt].damping. mjlab writes ``viscous_damping`` onto <joint damping>,
# which stacks on top of the PD derivative gain rather than replacing it. Menagerie's 2.0
# is a stand-in for unmodelled friction under *torque* control; reusing it here would give
# 2.0 passive + kd active. For the hip (effective inertia ~0.03, critical damping ~1.55)
# that is zeta ~1.9 -- heavily over-damped, where the real robot at kd=1.0 sits near 0.65.
# A policy trained on over-damped joints oscillates on hardware.
POSITION_VISCOUS_DAMPING = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}

# Per-joint flats aligned to go2.constants.JOINT_NAMES, for the Drake control law.
JOINT_KP_FLAT = tuple(GO2_ACTUATORS[jt].kp for _ in FEET for jt in JOINT_TYPES)
JOINT_KD_FLAT = tuple(GO2_ACTUATORS[jt].kd for _ in FEET for jt in JOINT_TYPES)

##
# Velocity command envelope.
##

# Final bounds (vx m/s, vy m/s, wz rad/s). Reached via COMMAND_STAGES, not from step 0 --
# commanding 2.5 m/s on rugged terrain before the robot can walk is an exploration wall.
# It is also a *reward trap*: standing still banks the full `upright` + `pose` reward
# (~2.0/step) while an unreachable velocity command makes tracking hopeless, so the policy
# converges on standing and turning in place.
COMMAND_BOUNDS = (2.5, 1.0, 1.5)

# PPO rollout length. Lives here rather than in rl_cfg.py because COMMAND_STAGES needs it
# to convert iterations to env steps; rl_cfg.py imports it back.
NUM_STEPS_PER_ENV = 50

# Stage thresholds are compared against ``env.common_step_counter``, which counts
# **environment steps**, NOT training iterations -- it increments once per env.step().
# Writing iteration numbers here directly makes every stage fire ~50x too early.
_COMMAND_STAGE_ITERS = (
  (0, (1.5, 0.8, 1.2)),
  (800, (2.0, 1.0, 1.5)),
  (2000, COMMAND_BOUNDS),
)
COMMAND_STAGES = tuple(
  (iteration * NUM_STEPS_PER_ENV, bounds) for iteration, bounds in _COMMAND_STAGE_ITERS
)
# Fraction of resamples that command a full stop. Without this the jitter scheme reaches
# all-zero in ~0.5% of resamples, leaving `stand_still` and every command gate dead.
REL_STANDING_PROB = 0.1
COMMAND_RESAMPLE_TIME_S = 8.0

##
# Sensor names.
##

TERRAIN_SCAN_SENSOR = "terrain_scan"
FOOT_HEIGHT_SENSOR = "foot_height_scan"
BASE_HEIGHT_SENSOR = "base_height_scan"
THIGH_CONTACT_SENSOR = "thigh_ground_touch"
SHANK_CONTACT_SENSOR = "shank_ground_touch"
TRUNK_CONTACT_SENSOR = "trunk_ground_touch"

# Minimum base clearance above local terrain before the episode is a failure. The trunk
# collision box half-height is 0.057, so belly-down is ~0.06 while nominal stance is 0.27;
# 0.12 catches a collapse without firing on a deep crouch.
MIN_BASE_CLEARANCE = 0.12
