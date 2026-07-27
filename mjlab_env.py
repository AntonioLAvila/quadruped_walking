"""
mjlab is manager-based and config-driven: instead of an ``MjxEnv`` subclass with
``reset``/``step``, the environment is a ``ManagerBasedRlEnvCfg`` assembled from
observation/reward/termination/event/command/action *terms*. This module builds that
config for the Go2 (torque control, flat terrain, asymmetric 45/120 observations) and
registers the task so it can be trained with mjlab's CLI:

    python train_go2.py Mjlab-Velocity-Flat-Unitree-Go2

Robot constants live in ``go2_constants.py``, the mjlab scene/entity builders in
``go2_robot.py``, and the custom MDP terms in ``go2_mdp.py``. The original
``env.py``/``train.py``/``configs.py`` are left untouched (brax reference).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_root_state_uniform
from mjlab.envs.mdp.observations import (
  base_ang_vel,
  base_lin_vel,
  builtin_sensor,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.envs.mdp.rewards import action_rate_l2, flat_orientation_l2, is_terminated
from mjlab.envs.mdp.terminations import (
  bad_orientation,
  root_height_below_minimum,
  time_out,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import go2_mdp
from go2_constants import (
  BASE_BODY,
  BASE_MASS,
  DEFAULT_HEIGHT,
  FEET_CONTACT_SENSOR,
  FEET_MAX_HEIGHT,
  FOOT_GEOMS,
  FOOT_SITES,
  GO2_ACTION_SCALE,
)
from go2_robot import get_feet_contact_sensor_cfg, get_go2_robot_cfg
from go2_rl_cfg import DEFAULT_NUM_ENVS, go2_ppo_runner_cfg

TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go2"
COMMAND_NAME = "twist"

# Domain-randomization event keys (disabled together in deterministic `play` eval).
_DR_EVENT_KEYS = (
  "base_inertial",
  "foot_friction",
  "foot_friction_torsion_roll",
  "joint_frictionloss",
  "joint_damping",
  "joint_armature",
  "encoder_bias",
)

# env.py noise_config (level 1.0): per-signal uniform noise scales.
_NOISE = {
  "joint_pos": Unoise(n_min=-0.03, n_max=0.03),
  "joint_vel": Unoise(n_min=-1.5, n_max=1.5),
  "base_lin_vel": Unoise(n_min=-0.1, n_max=0.1),
  "base_ang_vel": Unoise(n_min=-0.2, n_max=0.2),
  "projected_gravity": Unoise(n_min=-0.05, n_max=0.05),
}


def _actor_terms() -> dict[str, ObservationTermCfg]:
  """The 45-dim policy observation (fresh cfg objects each call)."""
  return {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, noise=_NOISE["joint_pos"]),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, noise=_NOISE["joint_vel"]),
    # Base *linear* velocity is not reliably observable on the real Go2 (no reliable
    # state estimate for it), so the policy is trained blind to it. The critic still
    # sees the true value via "base_lin_vel_priv". Angular velocity stays: it is a
    # direct IMU gyro reading on hardware.
    # "base_lin_vel": ObservationTermCfg(func=base_lin_vel, noise=_NOISE["base_lin_vel"]),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel, noise=_NOISE["base_ang_vel"]),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=_NOISE["projected_gravity"]
    ),
    "last_action": ObservationTermCfg(func=last_action),
    "command": ObservationTermCfg(
      func=generated_commands, params={"command_name": COMMAND_NAME}
    ),
  }


def _privileged_terms() -> dict[str, ObservationTermCfg]:
  """The extra clean terms appended after the noisy actor block (75 dims)."""
  robot_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)
  base = SceneEntityCfg("robot", body_names=BASE_BODY)
  return {
    "joint_pos_priv": ObservationTermCfg(func=joint_pos_rel),
    "joint_vel_priv": ObservationTermCfg(func=joint_vel_rel),
    "base_lin_vel_priv": ObservationTermCfg(func=base_lin_vel),
    "base_ang_vel_priv": ObservationTermCfg(func=base_ang_vel),
    "projected_gravity_priv": ObservationTermCfg(func=projected_gravity),
    "accelerometer": ObservationTermCfg(
      func=builtin_sensor, params={"sensor_name": "robot/accelerometer"}
    ),
    "base_ang_vel_w": ObservationTermCfg(func=go2_mdp.base_ang_vel_w),
    "feet_lin_vel": ObservationTermCfg(
      func=go2_mdp.feet_lin_vel, params={"asset_cfg": robot_sites}
    ),
    "actuator_force": ObservationTermCfg(func=go2_mdp.actuator_force),
    "kick_force": ObservationTermCfg(func=go2_mdp.kick_force, params={"asset_cfg": base}),
    "foot_contact": ObservationTermCfg(
      func=go2_mdp.foot_contact, params={"sensor_name": FEET_CONTACT_SENSOR}
    ),
    "foot_air_time": ObservationTermCfg(
      func=go2_mdp.foot_air_time, params={"sensor_name": FEET_CONTACT_SENSOR}
    ),
    "is_being_kicked": ObservationTermCfg(
      func=go2_mdp.is_being_kicked, params={"asset_cfg": base}
    ),
  }


def make_go2_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Go2 velocity-tracking config (flat terrain, torque control)."""
  robot_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)

  # --- Scene: Go2 on a flat plane + a feet contact sensor. ---
  # Actuator command latency is DR too, but it lives on the actuator cfg rather
  # than in `events`, so it is disabled here rather than popped in the play block.
  scene = SceneCfg(
    num_envs=DEFAULT_NUM_ENVS,
    extent=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_go2_robot_cfg(command_delay=not play)},
    sensors=(get_feet_contact_sensor_cfg(),),
  )

  # --- Observations: actor (45, noisy) + critic (120 = noisy 45 ++ clean 75). ---
  observations = {
    "actor": ObservationGroupCfg(
      terms=_actor_terms(), concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms={**_actor_terms(), **_privileged_terms()},
      concatenate_terms=True,
      enable_corruption=True,  # noisy first-45; appended terms have no noise cfg.
    ),
  }

  # --- Action: direct torque (env.py applied action * scale as motor ctrl). ---
  actions = {
    "joint_effort": JointEffortActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=GO2_ACTION_SCALE
    )
  }

  # --- Command: faithful velocity command with jitter + exponential resampling. ---
  commands = {
    COMMAND_NAME: go2_mdp.Go2VelocityCommandCfg(
      bounds=(1.5, 0.8, 1.2), probs=(0.9, 0.25, 0.5), mean_resample_time_s=5.0
    )
  }

  # --- Rewards: weight == env.py scale (RewardManager applies the * dt). ---
  rewards = {
    "healthy": RewardTermCfg(func=go2_mdp.healthy, weight=1e-3),
    "tracking_lin_vel": RewardTermCfg(
      func=go2_mdp.tracking_lin_vel,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=go2_mdp.tracking_ang_vel,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(func=go2_mdp.lin_vel_z, weight=-0.5),
    "ang_vel_xy": RewardTermCfg(func=go2_mdp.ang_vel_xy, weight=-0.5),
    "orientation": RewardTermCfg(func=flat_orientation_l2, weight=-5.0),
    "dof_pos_limits": RewardTermCfg(func=go2_mdp.dof_pos_limits, weight=-1.0),
    "pose": RewardTermCfg(
      func=go2_mdp.pose,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "weight": {".*_hip_joint": 1.0, ".*_thigh_joint": 0.8, ".*_calf_joint": 0.1},
      },
    ),
    "termination": RewardTermCfg(func=is_terminated, weight=-1.0),
    "stand_still": RewardTermCfg(
      func=go2_mdp.stand_still,
      weight=-1.0,
      params={"command_name": COMMAND_NAME}
    ),
    "torques": RewardTermCfg(func=go2_mdp.torques, weight=-1e-4),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-5.0),
    "energy": RewardTermCfg(func=go2_mdp.energy, weight=-2e-4),
    "feet_clearance": RewardTermCfg(
      func=go2_mdp.feet_clearance,
      weight=-0.2,
      params={"asset_cfg": robot_sites, "max_foot_height": FEET_MAX_HEIGHT},
    ),
    "feet_height": RewardTermCfg(
      func=go2_mdp.feet_height,
      weight=-0.2,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "asset_cfg": robot_sites,
        "max_foot_height": FEET_MAX_HEIGHT,
        "command_threshold": 0.01,
      },
    ),
    "feet_slip": RewardTermCfg(
      func=go2_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "asset_cfg": robot_sites,
        "command_threshold": 0.01,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=go2_mdp.feet_air_time,
      weight=1.0,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.01,
      },
    ),
  }

  # --- Terminations: height + tilt (failures); time-out (truncation). ---
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=bad_orientation, params={"limit_angle": math.pi / 4}  # body_z_axis_z < 0.707
    ),
    "low_height": TerminationTermCfg(
      func=root_height_below_minimum, params={"minimum_height": DEFAULT_HEIGHT - 0.025}
    ),
  }

  # --- Events: physical domain randomization + reset randomization + the kick. ---
  #
  # The observation noise (above) and the kick (below) already model sensing noise
  # and external disturbances. The terms here close the remaining sim-to-real gap:
  # the *physical parameters* that are never known exactly on hardware. They are all
  # ``startup`` events, so each parallel env is built once with its own sampled
  # dynamics and the policy must learn a controller robust across the whole range.
  feet = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  base = SceneEntityCfg("robot", body_names=BASE_BODY)
  all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  events = {
    # Base mass / inertia / CoM: payload and build-to-build mass uncertainty.
    # pseudo_inertia perturbs mass, inertia, ipos and iquat *consistently* (so the
    # inertia tensor tracks the mass change). alpha is a log mass-density scale, so
    # mass scales by exp(2*alpha) ~= [0.82, 1.22]; t shifts the CoM by +/- 2 cm.
    "base_inertial": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={"asset_cfg": base, "alpha_range": (-0.1, 0.1), "t_range": (-0.02, 0.02)},
    ),
    # Foot-ground tangential friction (per foot, so legs can see different grip).
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
    # Torsional / rolling friction (the foot contacts are condim 6, so these matter).
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
    # Joint dry friction (frictionloss), viscous damping, and reflected rotor inertia
    # (armature): the dominant unmodeled actuator/transmission dynamics. Scaled per
    # joint about the MJCF defaults (frictionloss 0.2, damping 2.0, armature 0.01).
    "joint_frictionloss": EventTermCfg(
      func=dr.joint_friction,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.5, 1.5)},
    ),
    "joint_damping": EventTermCfg(
      func=dr.joint_damping,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.75, 1.25)},
    ),
    # Armature is the least well known of these. Menagerie ships a flat 0.01 for
    # every joint; Unitree's own RL config uses 0.02 for the knee -- a 2x spread on
    # the value we cannot measure. GO2_ACTUATORS takes Unitree's numbers, and this
    # range is widened to (0.5, 2.0) so the whole span of that disagreement stays
    # in distribution rather than betting the policy on one end of it.
    "joint_armature": EventTermCfg(
      func=dr.joint_armature,
      mode="startup",
      params={"asset_cfg": all_joints, "operation": "scale", "ranges": (0.5, 2.0)},
    ),
    # Constant per-joint encoder offset: real joint encoders are imperfectly zeroed.
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": all_joints, "bias_range": (-0.02, 0.02)},
    ),
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
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
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "kick": EventTermCfg(
      func=go2_mdp.Go2KickEvent,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
        "base_mass": BASE_MASS,
        "wait_range_s": (0.05, 0.2),
        "duration_range_s": (0.05, 0.2),
        "magnitude_range": (0.0, 3.0),
      },
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=2,  # step_dt = 0.0025 * 2 = 0.005 (env.py ctrl_dt)
    episode_length_s=20.0,
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.0025, cone="elliptic", impratio=100.0),
      njmax=300,
      nconmax=None,
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
    # Deterministic eval: no observation noise, no kicks, nominal physics (no DR),
    # ~infinite episode.
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.events.pop("kick", None)
    for key in _DR_EVENT_KEYS:
      cfg.events.pop(key, None)

  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=make_go2_velocity_env_cfg(play=False),
  play_env_cfg=make_go2_velocity_env_cfg(play=True),
  rl_cfg=go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
