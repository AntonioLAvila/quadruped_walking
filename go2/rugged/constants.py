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

# Per-term symmetric uniform observation noise, i.e. U(-x, +x). Single source of truth:
# env_cfg.py wraps these in mjlab's UniformNoiseCfg, and scripts/verify_rugged.py applies
# them directly (it cannot import mjlab). Modelling sensor noise on the *deployment* side
# matters -- a policy that only works on clean observations is not deployable, and the
# training-time noise is the only reason it should be robust to real encoders and IMUs.
#
# last_action and command are exact: the robot knows what it commanded.
ACTOR_NOISE = {
  "joint_pos": 0.03,
  "joint_vel": 1.5,
  "base_ang_vel": 0.2,
  "projected_gravity": 0.05,
  "last_action": 0.0,
  "command": 0.0,
}
# Expanded to one scale per element of the 45-dim frame.
FRAME_NOISE_SCALE = tuple(
  scale for name, width in ACTOR_TERM_WIDTHS for scale in (ACTOR_NOISE[name],) * width
)

# Sensor pipeline latency, sampled per env per step from [min, max].
#
# UNITS: **control steps**, i.e. 20 ms each at this task's 50 Hz -- NOT physics steps.
# This differs from COMMAND_DELAY_S on the actuator side, which mjlab counts in physics
# timesteps (5 ms each). The two delays model different things and use different clocks:
#   observation delay = sensor -> onboard -> policy   (this one)
#   actuator delay    = policy -> bus -> motor board  (COMMAND_DELAY_S)
# 0-1 control steps is 0-20 ms, which brackets what a Go2 deployment stack actually sees;
# the mechanism cannot express sub-step latency.
#
# Applied only to *sensed* terms -- the same set that gets noise. last_action and command
# are known exactly and instantly: no sensor sits in either path.
ACTOR_DELAY_STEPS = (0, 1)
ACTOR_DELAYED_TERMS = tuple(name for name, scale in ACTOR_NOISE.items() if scale > 0.0)

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
# 2.0 passive + kd active, which is heavily over-damped.
#
# 0.1 is what the gait-switching project trained its 3 m/s policy on. The previous 0.5 is
# not obviously wrong for the real robot, but it is expensive at speed: a swinging leg
# reaches ~20 rad/s, where 0.5 burns 10 of the hip's 23.5 N*m just moving the joint. The
# true value is uncertain, which is what the (0.4, 2.0) ``joint_damping`` DR range is for.
POSITION_VISCOUS_DAMPING = {"hip": 0.1, "thigh": 0.1, "calf": 0.1}

# Per-joint flats aligned to go2.constants.JOINT_NAMES, for the Drake control law.
JOINT_KP_FLAT = tuple(GO2_ACTUATORS[jt].kp for _ in FEET for jt in JOINT_TYPES)
JOINT_KD_FLAT = tuple(GO2_ACTUATORS[jt].kd for _ in FEET for jt in JOINT_TYPES)

##
# Velocity command envelope.
#
# Two tiers, because the two things this policy is for pull in opposite directions:
#
#   cruise  up to CRUISE_SPEED, on every terrain.
#   sprint  up to the top of LIN_VEL_X_STAGES, only while the robot is over a sprint lane
#           (SPRINT_TERRAINS). ``rugged.mdp.SprintVelocityCommand`` enforces it.
#
# CRUISE_SPEED is measured, not guessed. A 6000-iteration run that commanded 2.5 m/s on
# every terrain showed 2.0 is where a blind policy stops improving on rough ground:
#
#   ramp to 2.0:  tracking 1.10 -> 0.76 -> recovered to 0.98, terrain 3.6 -> 4.4
#   ramp to 2.5:  tracking 0.98 -> 0.71 and flat for 3500 iterations, terrain stalled,
#                 falls doubled
#
# On flat ground the same robot tracks a 3.0 m/s envelope with no falls (gait-switching),
# so the limit is the terrain, not the robot -- hence lanes rather than one number.
##

CRUISE_SPEED = 2.0
REVERSE_SPEED = 1.0
LIN_VEL_Y = (-0.5, 0.5)
ANG_VEL_Z = (-1.0, 1.0)

# Sub-terrain names (keys of ``rugged.terrain``'s dict) that may be sprinted on.
SPRINT_TERRAINS = ("flat", "rough_light")
# A robot counts as on a lane only this far (m) inside it, so the command has dropped to
# cruise before the first foot lands on technical ground rather than after.
SPRINT_LANE_MARGIN = 1.0

# PPO rollout length. Lives here rather than in rl_cfg.py because the ramp below needs it
# to convert iterations to env steps; rl_cfg.py imports it back.
NUM_STEPS_PER_ENV = 50

# (training iteration, top forward speed in m/s).
#
# The last rungs are a probe, not a promise: nothing here knows how fast the robot can
# actually go. ``scripts/eval_rugged.py`` measures the speed reached at each command; if
# the top rungs are out of reach, lower them to what it reports, because an unreachable
# command is a reward trap (standing still banks ``upright`` + ``pose`` while tracking is
# hopeless either way).
_LIN_VEL_X_STAGE_ITERS = (
  (0, 1.5),
  (1000, 2.5),
  (2000, 3.0),
  (3000, 3.5),
  (4000, 4.0),
)
SPRINT_SPEED = _LIN_VEL_X_STAGE_ITERS[-1][1]

# mjlab's ``commands_vel`` compares against ``env.common_step_counter``, which counts
# **environment steps**, NOT training iterations. Writing iteration numbers into the
# curriculum directly makes every stage fire ~50x too early. Converted here, once.
LIN_VEL_X_STAGES = tuple(
  (iteration * NUM_STEPS_PER_ENV, (-REVERSE_SPEED, top))
  for iteration, top in _LIN_VEL_X_STAGE_ITERS
)

##
# Sensor names.
##

TERRAIN_SCAN_SENSOR = "terrain_scan"
FOOT_HEIGHT_SENSOR = "foot_height_scan"
LEG_CONTACT_SENSOR = "leg_ground_touch"
TRUNK_CONTACT_SENSOR = "trunk_ground_touch"
