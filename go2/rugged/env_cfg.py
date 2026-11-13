"""Rugged-terrain Go2 velocity task: PD position control, 50 Hz, blind actor + history.

One policy for two jobs: cross hills, slopes, jagged ground and shallow steps at up to
``CRUISE_SPEED``, and sprint as fast as the robot will go over open ground. It is the
gait-switching project's flat task (mjlab built-ins, energy-driven gait selection) with
terrain underneath it.

What makes it deployable, and what to keep true when editing:

* **The actor is blind.** Terrain scans feed the critic and the rewards only, so the
  exported policy needs no elevation map. ``HISTORY_LENGTH`` frames of proprioception are
  what let it infer the ground.
* **Every height term is terrain-relative**, read from raycast sensors. Absolute world z
  means nothing on generated terrain.
* **Speed above cruise is only ever commanded over a sprint lane.** See
  ``rugged.mdp.SprintVelocityCommand``.

Nothing outside ``go2/rugged/`` is modified; terrain-agnostic terms come from ``go2.mdp``.

    uv run scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import (
  push_by_setting_velocity,
  randomize_terrain,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.envs.mdp.observations import (
  base_ang_vel,
  base_lin_vel,
  generated_commands,
  height_scan,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.envs.mdp.rewards import (
  action_rate_l2,
  is_terminated,
  joint_pos_limits,
  joint_torques_l2,
)
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp.curriculums import commands_vel, terrain_levels_vel
from mjlab.tasks.velocity.mdp.observations import (
  foot_air_time,
  foot_contact,
  foot_contact_forces,
  foot_height,
)
from mjlab.tasks.velocity.mdp.rewards import (
  body_angular_velocity_penalty,
  feet_clearance,
  feet_slip,
  self_collision_cost,
  upright,
  variable_posture,
)
from mjlab.tasks.velocity.mdp.terminations import illegal_contact, out_of_terrain_bounds
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go2 import mdp as go2_mdp
from go2 import robot as flat_robot
from go2.constants import BASE_BODY, FEET_CONTACT_SENSOR, FOOT_GEOMS, FOOT_SITES
from go2.rugged import mdp as rugged_mdp
from go2.rugged import robot as rugged_robot
from go2.rugged.constants import (
  ACTOR_DELAY_STEPS,
  ACTOR_NOISE,
  ANG_VEL_Z,
  FOOT_HEIGHT_SENSOR,
  HISTORY_LENGTH,
  LEG_CONTACT_SENSOR,
  LIN_VEL_X_STAGES,
  LIN_VEL_Y,
  POSITION_ACTION_SCALE,
  POSITION_TIMING,
  TERRAIN_SCAN_SENSOR,
  TRUNK_CONTACT_SENSOR,
)
from go2.rugged.rl_cfg import DEFAULT_NUM_ENVS, go2_rugged_ppo_runner_cfg
from go2.rugged.terrain import make_rugged_terrain_cfg

TASK_ID = "Mjlab-Velocity-Rugged-Unitree-Go2"
COMMAND_NAME = "twist"

# Swing height the foot *site* should reach above the ground under it. The site is the
# foot sphere's centre (radius 0.022), so this is ~8 cm of daylight -- the tallest step in
# the terrain mix.
FOOT_CLEARANCE_TARGET = 0.10

# Everything that randomizes the physics or disturbs the robot; popped together in `play`.
# pd_gains and effort_limits only exist under position actuators (they raise TypeError on
# the flat task's torque <motor>).
_DR_EVENT_KEYS = (
  "base_inertial",
  "foot_friction",
  "foot_friction_torsion_roll",
  "joint_frictionloss",
  "joint_damping",
  "joint_armature",
  "encoder_bias",
  "pd_gains",
  "effort_limits",
  "push",
)

# Built from ACTOR_NOISE so the Drake verification applies exactly the same magnitudes.
_NOISE = {
  name: Unoise(n_min=-scale, n_max=scale)
  for name, scale in ACTOR_NOISE.items()
  if scale > 0.0
}


def _sensed(func, name: str, delay: bool) -> ObservationTermCfg:
  """An actor term read off a real sensor, so it gets noise and pipeline latency.

  mjlab's per-term pipeline is compute -> noise -> clip -> scale -> delay -> history, so
  what enters the history buffer is what the robot's sensors reported -- late and noisy.
  """
  lo, hi = ACTOR_DELAY_STEPS if delay else (0, 0)
  return ObservationTermCfg(
    func=func, noise=_NOISE[name], delay_min_lag=lo, delay_max_lag=hi
  )


def _actor_terms(delay: bool = True) -> dict[str, ObservationTermCfg]:
  """What the real Go2 can measure: 45 dims per frame, stacked ``HISTORY_LENGTH`` deep.

  No height scan, and no ``base_lin_vel`` (there is no reliable estimator for it on the
  real robot). Stacking is TERM-MAJOR, oldest to newest:
    [joint_pos t-4..t | joint_vel t-4..t | ang_vel | gravity | last_action | command]
  That is the opposite of legged_gym's time-major stacking, and any deployment-side buffer
  must match it exactly. ``scripts/check_obs_layout.py`` asserts it.

  Args:
    delay: apply sensor latency. Off for deterministic evaluation and for the layout
      check, which cannot reconstruct a randomly-lagged signal from the live state.
  """
  return {
    "joint_pos": _sensed(joint_pos_rel, "joint_pos", delay),
    "joint_vel": _sensed(joint_vel_rel, "joint_vel", delay),
    "base_ang_vel": _sensed(base_ang_vel, "base_ang_vel", delay),
    "projected_gravity": _sensed(projected_gravity, "projected_gravity", delay),
    # No noise and no delay: the robot knows exactly what it commanded, instantly.
    "last_action": ObservationTermCfg(func=last_action),
    "command": ObservationTermCfg(
      func=generated_commands, params={"command_name": COMMAND_NAME}
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """Clean single-frame state, plus what the actor cannot measure (283 dims).

  The last two are what make the value function terrain-aware while the policy stays
  blind. Do not move them to the actor.
  """
  feet = {"sensor_name": FEET_CONTACT_SENSOR}
  return {
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=projected_gravity),
    "joint_pos": ObservationTermCfg(func=joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
    "command": ObservationTermCfg(
      func=generated_commands, params={"command_name": COMMAND_NAME}
    ),
    "actuator_force": ObservationTermCfg(func=go2_mdp.actuator_force),
    "feet_lin_vel": ObservationTermCfg(
      func=go2_mdp.feet_lin_vel,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
    ),
    "foot_contact": ObservationTermCfg(func=foot_contact, params=feet),
    "foot_air_time": ObservationTermCfg(func=foot_air_time, params=feet),
    "foot_contact_forces": ObservationTermCfg(func=foot_contact_forces, params=feet),
    "height_scan": ObservationTermCfg(
      func=height_scan, params={"sensor_name": TERRAIN_SCAN_SENSOR}
    ),
    "foot_height": ObservationTermCfg(
      func=foot_height, params={"sensor_name": FOOT_HEIGHT_SENSOR}
    ),
  }


def _command(play: bool) -> UniformVelocityCommandCfg:
  """mjlab's uniform velocity command; in training, with the sprint-lane rule on top.

  ``play`` uses the plain built-in with the full range: its terrain grid is sampled
  randomly rather than laid out in columns, so there are no lanes to gate on, and the
  viewer's joystick is the operator.

  ``heading_command`` is here for the terrain curriculum, which promotes on distance
  travelled from the spawn point: a robot holding a yaw rate walks a circle and gets
  nowhere however well it tracks. Heading envs turn to a target and then go straight.
  """
  cls = UniformVelocityCommandCfg if play else rugged_mdp.SprintVelocityCommandCfg
  return cls(
    entity_name="robot",
    resampling_time_range=(8.0, 16.0),
    rel_standing_envs=0.1,
    rel_forward_envs=0.5,
    heading_command=True,
    rel_heading_envs=0.3,
    heading_control_stiffness=0.5,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=LIN_VEL_X_STAGES[-1 if play else 0][1],
      lin_vel_y=LIN_VEL_Y,
      ang_vel_z=ANG_VEL_Z,
      heading=(-math.pi, math.pi),
    ),
  )


def _rewards() -> dict[str, RewardTermCfg]:
  """Weights are per second; the reward manager multiplies by dt.

  gait-switching's set and weights, with the flat-ground terms swapped for
  terrain-relative ones and one penalty added for dragging legs across the ground.
  """
  foot_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)
  base = SceneEntityCfg("robot", body_names=(BASE_BODY,))
  gated = {
    "sensor_name": FEET_CONTACT_SENSOR,
    "command_name": COMMAND_NAME,
    "command_threshold": 0.05,
  }
  return {
    # go2.mdp's trackers, not mjlab's track_linear/angular_velocity: those fold body-z
    # velocity and roll/pitch rate into the same kernel, so a bounding gait would lose
    # its tracking reward for pitching -- and bounding is what the energy term is meant to
    # be free to pick at speed.
    "tracking_lin_vel": RewardTermCfg(
      func=go2_mdp.tracking_lin_vel,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.4},
    ),
    # The kernel above goes flat once the robot is ~1 m/s short of its command; this does
    # not. Weighted to outbid what standing still collects for free (yaw tracking, pose
    # and upright, about 2/s) net of what running costs, even when the command is far out
    # of reach. See rugged.mdp.forward_progress.
    "progress": RewardTermCfg(
      func=rugged_mdp.forward_progress, weight=2.0, params={"command_name": COMMAND_NAME}
    ),
    # Half the linear weight: a robot that gives up on a fast command and stands still
    # still earns this in full whenever the yaw command is zero.
    "tracking_ang_vel": RewardTermCfg(
      func=go2_mdp.tracking_ang_vel,
      weight=1.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.25},
    ),
    # Gait selection comes from these two. Mechanical power is hundreds of watts, so the
    # old -2e-4 cost about 2% of the tracking reward and could not decide anything.
    "energy": RewardTermCfg(func=go2_mdp.energy, weight=-1e-3),
    "torques": RewardTermCfg(func=joint_torques_l2, weight=-1e-4),
    # The terms below all shape motion, so they are kept light enough not to pick the gait
    # themselves. Action rate grows with stride frequency squared, which would otherwise
    # tax fast gaits; pitch and bounce are inherent to bounding and galloping.
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.1),
    "lin_vel_z": RewardTermCfg(func=rugged_mdp.lin_vel_z_b, weight=-0.25),
    "ang_vel_xy": RewardTermCfg(
      func=body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": base}
    ),
    # Attitude relative to the ground under the robot, not to gravity. A gravity-relative
    # term scores sin^2(24 deg) on the steepest slope here and charges the robot for
    # standing on it correctly. Kept at half mjlab's weight because it is paid in full for
    # standing still, and every such term makes giving up on a fast command cheaper.
    "upright": RewardTermCfg(
      func=upright,
      weight=0.5,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": base,
        "terrain_sensor_names": (TERRAIN_SCAN_SENSOR,),
      },
    ),
    "pose": RewardTermCfg(
      func=variable_posture,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": COMMAND_NAME,
        "std_standing": {".*_(hip|thigh)_joint": 0.05, ".*_calf_joint": 0.1},
        "std_walking": {".*_(hip|thigh)_joint": 0.3, ".*_calf_joint": 0.6},
        "std_running": {".*_hip_joint": 0.3, ".*_thigh_joint": 0.6, ".*_calf_joint": 0.9},
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # Twice gait-switching's weight: on a plane a low swing costs nothing, here it is how
    # a blind robot trips.
    "feet_clearance": RewardTermCfg(
      func=feet_clearance,
      weight=-0.2,
      params={
        "target_height": FOOT_CLEARANCE_TARGET,
        "height_sensor_name": FOOT_HEIGHT_SENSOR,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.05,
        "asset_cfg": foot_sites,
      },
    ),
    "feet_slip": RewardTermCfg(
      func=feet_slip, weight=-0.1, params={**gated, "asset_cfg": foot_sites}
    ),
    # Small on purpose: enough to punish tapping, too small to set the swing time.
    "feet_air_time": RewardTermCfg(
      func=rugged_mdp.feet_air_time, weight=0.25, params={**gated, "target_time": 0.2}
    ),
    "stand_still": RewardTermCfg(
      func=go2_mdp.stand_still, weight=-1.0, params={"command_name": COMMAND_NAME}
    ),
    "joint_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
    # Thighs and shanks on the ground are penalized, not terminal: on jagged terrain a Go2
    # brushes its legs constantly. Only the trunk terminates.
    "leg_collision": RewardTermCfg(
      func=self_collision_cost, weight=-0.1, params={"sensor_name": LEG_CONTACT_SENSOR}
    ),
    # dt-scaled like everything else, so -25 is -0.5 per fall.
    "termination": RewardTermCfg(func=is_terminated, weight=-25.0),
  }


def _events() -> dict[str, EventTermCfg]:
  feet = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  base = SceneEntityCfg("robot", body_names=(BASE_BODY,))
  all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  return {
    "base_inertial": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={"asset_cfg": base, "alpha_range": (-0.1, 0.1), "t_range": (-0.02, 0.02)},
    ),
    "foot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": feet,
        "operation": "abs",
        "axes": [0],
        "ranges": (0.4, 1.2),
        "shared_random": False,
      },
    ),
    "foot_friction_torsion_roll": EventTermCfg(
      func=dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": feet,
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1, 2],
        "ranges": {1: (5e-3, 5e-2), 2: (1e-3, 2e-2)},
        "shared_random": False,
      },
    ),
    "joint_frictionloss": EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.5, 1.5)},
    ),
    # Wide on purpose: the right passive damping under position control is genuinely
    # uncertain (see POSITION_VISCOUS_DAMPING), and this range spans the disagreement.
    "joint_damping": EventTermCfg(
      func=dr.joint_damping,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.4, 2.0)},
    ),
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.5, 2.0)},
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": all_joints, "bias_range": (-0.02, 0.02)},
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains,
      mode="startup",
      params={
        "asset_cfg": all_joints,
        "kp_range": (0.8, 1.25),
        "kd_range": (0.75, 1.3),
        "operation": "scale",
      },
    ),
    "effort_limits": EventTermCfg(
      func=dr.effort_limits,
      mode="startup",
      params={
        "asset_cfg": all_joints,
        "effort_limit_range": (0.85, 1.0),
        "operation": "scale",
      },
    ),
    "push": EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(13.0, 20.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        # An explicit z lift, so the robot does not spawn inside a bump.
        "pose_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (0.05, 0.15),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": all_joints,
      },
    ),
  }


def make_go2_rugged_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the task config.

  ``play`` strips noise, latency, DR and the push, makes episodes ~infinite, opens the
  full speed range and samples terrain uniformly instead of by curriculum level.
  """
  cfg = ManagerBasedRlEnvCfg(
    decimation=POSITION_TIMING.decimation,
    episode_length_s=20.0,
    scene=SceneCfg(
      num_envs=DEFAULT_NUM_ENVS,
      extent=2.0,
      terrain=make_rugged_terrain_cfg(play=play),
      # Command latency is DR too, but it lives on the actuator cfg rather than in
      # `events`, so it is switched here rather than popped below.
      entities={"robot": rugged_robot.get_go2_robot_cfg(command_delay=not play)},
      sensors=(
        flat_robot.get_feet_contact_sensor_cfg(),
        rugged_robot.get_terrain_scan_sensor_cfg(),
        rugged_robot.get_foot_height_sensor_cfg(),
        *rugged_robot.get_body_contact_sensor_cfgs(),
      ),
    ),
    # nan_policy="sanitize" is the backstop for heightfield contacts occasionally NaN-ing
    # the warp solver: rsl_rl aborts training on a single non-finite value, and mjlab's
    # nan_detection cannot see derived quantities like raycast heights.
    observations={
      "actor": ObservationGroupCfg(
        terms=_actor_terms(delay=not play),
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=HISTORY_LENGTH,
        nan_policy="sanitize",
      ),
      "critic": ObservationGroupCfg(
        terms=_critic_terms(),
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
      ),
    },
    # The key must be "joint_pos": mjlab's ONNX metadata exporter asserts on it.
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=POSITION_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={COMMAND_NAME: _command(play)},
    rewards=_rewards(),
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      # Looser than gait-switching's 60 deg: this is measured against gravity, the
      # steepest slope is 24 deg, and transient tilt on rubble reaches 45-50.
      "fell_over": TerminationTermCfg(
        func=bad_orientation, params={"limit_angle": math.radians(75.0)}
      ),
      # The terrain-safe "it collapsed" test. root_height_below_minimum compares world z
      # against a constant, which fires at once in any dip and never on a rise.
      "illegal_contact": TerminationTermCfg(
        func=illegal_contact,
        params={"sensor_name": TRUNK_CONTACT_SENSOR, "force_threshold": 20.0},
      ),
      # Truncation, not failure: leaving the grid is not the policy's fault.
      "out_of_bounds": TerminationTermCfg(
        func=out_of_terrain_bounds, params={"margin": 0.5}, time_out=True
      ),
      "state_nan": TerminationTermCfg(func=nan_detection),
    },
    events=_events(),
    curriculum={
      # Promotes a robot that got more than half a patch from its spawn point, demotes
      # one that covered under half the distance its command asked for.
      "terrain_levels": CurriculumTermCfg(
        func=terrain_levels_vel, params={"command_name": COMMAND_NAME}
      ),
      # Steps are env steps, not iterations; LIN_VEL_X_STAGES is already converted.
      "command_vel": CurriculumTermCfg(
        func=commands_vel,
        params={
          "command_name": COMMAND_NAME,
          "velocity_stages": [
            {"step": step, "lin_vel_x": lin_vel_x} for step, lin_vel_x in LIN_VEL_X_STAGES
          ],
        },
      ),
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=POSITION_TIMING.sim_timestep,
        cone="elliptic",
        # With many simultaneous terrain contacts a high impratio makes the constraint
        # system ill-conditioned. mjlab's rough tasks use 10.
        impratio=10.0,
        iterations=10,
        ls_iterations=20,
        # Box and heightfield terrain need real continuous-collision iterations.
        ccd_iterations=500,
      ),
      # A flat plane needs njmax=300; terrain generates far more simultaneous contacts.
      njmax=1500,
      nconmax=35,
      contact_sensor_maxmatch=500,
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=BASE_BODY,
      distance=2.0,
      elevation=-10.0,
    ),
  )

  if play:
    cfg.episode_length_s = int(1e9)
    for key in _DR_EVENT_KEYS:
      cfg.events.pop(key)
    cfg.terminations.pop("out_of_bounds")
    # The curriculum owns terrain levels and types in training; without it something
    # else has to place the robots.
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=randomize_terrain, mode="reset", params={}
    )

  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=make_go2_rugged_env_cfg(play=False),
  play_env_cfg=make_go2_rugged_env_cfg(play=True),
  rl_cfg=go2_rugged_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
