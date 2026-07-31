"""Rugged-terrain Go2 velocity task: PD position control, 50 Hz, blind actor + history.

A robustness upgrade over the flat task -- hills, slopes, jagged ground and shallow steps
-- deliberately stopping short of anything needing foothold planning.

Three things differ structurally from ``go2.flat.env_cfg``:

* **PD position control at 50 Hz** instead of direct torque at 200 Hz. Every Unitree RL
  stack and every published rough-terrain result uses position control; the real robot's
  motor boards close a ~1 kHz PD loop that torque control throws away.
* **The actor is blind but has memory.** Terrain scans go to the critic only, so the
  deployed policy needs no elevation map. ``HISTORY_LENGTH`` frames of proprioception are
  what let it infer terrain and contact state.
* **Every height-based term is terrain-relative**, read from raycast sensors rather than
  absolute world z, which is meaningless on generated terrain.

Nothing in ``go2/`` outside this package is modified; terrain-agnostic terms are imported
from ``go2.mdp`` so the flat task cannot regress.

    python scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import (
  randomize_terrain,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.envs.mdp.observations import (
  base_ang_vel,
  base_lin_vel,
  builtin_sensor,
  generated_commands,
  height_scan,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.envs.mdp.rewards import action_rate_l2, is_terminated
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
from mjlab.tasks.velocity.mdp.observations import foot_height
from mjlab.tasks.velocity.mdp.rewards import (
  feet_clearance,
  self_collision_cost,
  soft_landing,
  upright,
)
from mjlab.tasks.velocity.mdp.terminations import illegal_contact, out_of_terrain_bounds
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go2 import mdp as go2_mdp
from go2.constants import (
  BASE_BODY,
  BASE_MASS,
  FEET_CONTACT_SENSOR,
  FEET_MAX_HEIGHT,
  FOOT_GEOMS,
  FOOT_SITES,
)
from go2.rugged import mdp as rugged_mdp
from go2.rugged import robot as rugged_robot
from go2.rugged.constants import (
  ACTOR_NOISE,
  BASE_HEIGHT_SENSOR,
  COMMAND_RESAMPLE_TIME_S,
  COMMAND_STAGES,
  FOOT_HEIGHT_SENSOR,
  HISTORY_LENGTH,
  MIN_BASE_CLEARANCE,
  POSITION_ACTION_SCALE,
  POSITION_TIMING,
  REL_STANDING_PROB,
  SHANK_CONTACT_SENSOR,
  TERRAIN_SCAN_SENSOR,
  THIGH_CONTACT_SENSOR,
  TRUNK_CONTACT_SENSOR,
)
from go2.rugged.rl_cfg import DEFAULT_NUM_ENVS, go2_rugged_ppo_runner_cfg
from go2.rugged.terrain import make_rugged_terrain_cfg

TASK_ID = "Mjlab-Velocity-Rugged-Unitree-Go2"
COMMAND_NAME = "twist"

# Domain-randomization event keys, disabled together for deterministic `play` eval.
# Two more than the flat task: pd_gains and effort_limits only exist under position
# actuators (they raise TypeError on a torque <motor>).
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
)

# Built from ACTOR_NOISE so the Drake verification applies exactly the same magnitudes;
# see the note there. Terms with scale 0 get no noise cfg at all.
_NOISE = {
  name: Unoise(n_min=-scale, n_max=scale)
  for name, scale in ACTOR_NOISE.items()
  if scale > 0.0
}


def _actor_terms(history_length: int) -> dict[str, ObservationTermCfg]:
  """The blind policy observation: 45 dims per frame, stacked ``history_length`` deep.

  Same six terms as the flat actor -- no height scan, and no ``base_lin_vel`` (there is no
  reliable estimator for it on the real Go2). History is set **per term**, not on the
  group: a group-level ``history_length`` overrides every term, which would stack the
  187-ray terrain scan on the critic too.

  Resulting layout is TERM-MAJOR, oldest to newest:
    [joint_pos t-4..t | joint_vel t-4..t | ang_vel | gravity | last_action | command]
  That is the opposite of legged_gym's time-major stacking, and any deployment-side
  buffer must match it exactly. ``scripts/verify_rugged.py`` asserts this.
  """
  h = history_length
  return {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, noise=_NOISE["joint_pos"], history_length=h
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, noise=_NOISE["joint_vel"], history_length=h
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel, noise=_NOISE["base_ang_vel"], history_length=h
    ),
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity, noise=_NOISE["projected_gravity"], history_length=h
    ),
    "last_action": ObservationTermCfg(func=last_action, history_length=h),
    "command": ObservationTermCfg(
      func=generated_commands, params={"command_name": COMMAND_NAME}, history_length=h
    ),
  }


def _privileged_terms() -> dict[str, ObservationTermCfg]:
  """Clean, single-frame terms the critic gets and the actor never sees (270 dims).

  The flat task's 75 privileged dims plus 195 of terrain information: the full 187-ray
  height scan, per-foot clearance, base clearance, and the local surface normal. This is
  what makes the value function terrain-aware without the policy needing an elevation map.
  """
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
    # --- terrain, critic-only ---
    "height_scan": ObservationTermCfg(
      func=height_scan, params={"sensor_name": TERRAIN_SCAN_SENSOR}
    ),
    "foot_clearance_priv": ObservationTermCfg(
      func=foot_height, params={"sensor_name": FOOT_HEIGHT_SENSOR}
    ),
    "base_clearance": ObservationTermCfg(
      func=rugged_mdp.base_clearance, params={"sensor_name": BASE_HEIGHT_SENSOR}
    ),
    "terrain_normal": ObservationTermCfg(
      func=rugged_mdp.terrain_normal_b,
      params={"sensor_names": (TERRAIN_SCAN_SENSOR,)},
    ),
  }


def make_go2_rugged_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the rugged-terrain Go2 config (generated terrain, PD position control)."""
  robot_sites = SceneEntityCfg("robot", site_names=FOOT_SITES)

  # --- Scene ---
  # Command latency is DR too, but it lives on the actuator cfg rather than in `events`,
  # so it is disabled here rather than popped in the play block below.
  scene = SceneCfg(
    num_envs=DEFAULT_NUM_ENVS,
    extent=2.0,
    terrain=make_rugged_terrain_cfg(play=play),
    entities={"robot": rugged_robot.get_go2_robot_cfg(command_delay=not play)},
    sensors=(
      rugged_robot.get_feet_contact_sensor_cfg(),
      rugged_robot.get_terrain_scan_sensor_cfg(),
      rugged_robot.get_foot_height_sensor_cfg(),
      rugged_robot.get_base_height_sensor_cfg(),
      *rugged_robot.get_body_contact_sensor_cfgs(),
    ),
  )

  # --- Observations: actor (45 x H, noisy) + critic (actor block ++ 270 clean) ---
  # nan_policy="sanitize" is the backstop for heightfield contacts occasionally NaN-ing
  # the warp solver: rsl_rl aborts training on a single non-finite value, and mjlab's
  # nan_detection cannot see derived quantities like raycast heights.
  observations = {
    "actor": ObservationGroupCfg(
      terms=_actor_terms(HISTORY_LENGTH),
      concatenate_terms=True,
      enable_corruption=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={**_actor_terms(HISTORY_LENGTH), **_privileged_terms()},
      concatenate_terms=True,
      enable_corruption=True,
      nan_policy="sanitize",
    ),
  }

  # --- Action: PD position target. The key MUST be "joint_pos": mjlab's ONNX metadata
  # exporter asserts on both the name and the action type. ---
  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=POSITION_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  commands = {
    COMMAND_NAME: rugged_mdp.RuggedVelocityCommandCfg(
      bounds=COMMAND_STAGES[0][1],  # ramped up by the command_bounds curriculum
      probs=(0.9, 0.25, 0.5),
      mean_resample_time_s=COMMAND_RESAMPLE_TIME_S,
      rel_standing_prob=REL_STANDING_PROB,
    )
  }

  # --- Rewards ---
  rewards = {
    # Unchanged from flat, except sigma: exp(-err^2/0.25) scores only 0.1 for a 0.75 m/s
    # error, which is punishing once the command envelope reaches 2.5 m/s.
    "tracking_lin_vel": RewardTermCfg(
      func=go2_mdp.tracking_lin_vel,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.4},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=go2_mdp.tracking_ang_vel,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.25},
    ),
    "dof_pos_limits": RewardTermCfg(func=go2_mdp.dof_pos_limits, weight=-1.0),
    "torques": RewardTermCfg(func=go2_mdp.torques, weight=-1e-4),
    "energy": RewardTermCfg(func=go2_mdp.energy, weight=-2e-4),
    "feet_air_time": RewardTermCfg(
      func=go2_mdp.feet_air_time,
      weight=1.0,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.05,
      },
    ),
    "feet_slip": RewardTermCfg(
      # go2.mdp's version, not mjlab's: it aligns foot-site ids to the contact sensor's
      # primary order, where mjlab's assumes the two orderings already agree.
      func=go2_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "asset_cfg": robot_sites,
        "command_threshold": 0.05,
      },
    ),
    # Rescaled for 50 Hz. Rewards are dt-scaled, so every *state-based* weight carries
    # over unchanged -- but action_rate is a *difference*, so its per-second cost scales
    # as 1/dt^2. 200 -> 50 Hz is 16x: -5.0/16 ~= -0.31.
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.25),
    # Body frame, and softened. World-frame v_z charges for legitimately climbing a
    # slope; at a realistic rough-terrain |w_xy| ~ 1 rad/s the old -0.5 cost a quarter of
    # the tracking reward simply for crossing bumps.
    "lin_vel_z": RewardTermCfg(func=rugged_mdp.lin_vel_z_b, weight=-0.25),
    "ang_vel_xy": RewardTermCfg(func=rugged_mdp.ang_vel_xy_b, weight=-0.05),
    # Terrain-relative attitude, replacing flat_orientation_l2 at -5.0. That term scores
    # sin^2(24 deg) = 0.165 on a slope -- -0.83/s, about 40% of the maximum tracking
    # reward, charged for standing *correctly* on a hill.
    "upright": RewardTermCfg(
      func=upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
        "terrain_sensor_names": (TERRAIN_SCAN_SENSOR,),
      },
    ),
    # Replaces `healthy` (absolute z / DEFAULT_HEIGHT, which saturates dead on raised
    # terrain). One-sided, so climbing is never punished.
    "base_height": RewardTermCfg(
      func=rugged_mdp.base_height_above_terrain,
      weight=-2.0,
      params={"sensor_name": BASE_HEIGHT_SENSOR},
    ),
    # Terrain-relative foot terms. The foot site sits at the foot sphere's centre, so a
    # planted foot reads ~0.022 -- the same number the flat task's absolute-z version
    # sees, which is why FEET_MAX_HEIGHT carries over untouched.
    "feet_clearance": RewardTermCfg(
      func=feet_clearance,
      weight=-0.2,
      params={
        "target_height": FEET_MAX_HEIGHT,
        "height_sensor_name": FOOT_HEIGHT_SENSOR,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.05,
        "asset_cfg": robot_sites,
      },
    ),
    "feet_swing_height": RewardTermCfg(
      func=rugged_mdp.feet_swing_height,  # subclassed to add the missing reset hook
      weight=-0.2,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "height_sensor_name": FOOT_HEIGHT_SENSOR,
        "target_height": FEET_MAX_HEIGHT,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.05,
      },
    ),
    "soft_landing": RewardTermCfg(
      func=soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": FEET_CONTACT_SENSOR,
        "command_name": COMMAND_NAME,
        "command_threshold": 0.05,
      },
    ),
    # Leg and body contacts are penalized, not terminal -- on jagged ground a Go2 brushes
    # its legs constantly. Only the trunk terminates (see terminations below).
    "thigh_collision": RewardTermCfg(
      func=self_collision_cost, weight=-0.1, params={"sensor_name": THIGH_CONTACT_SENSOR}
    ),
    "shank_collision": RewardTermCfg(
      func=self_collision_cost, weight=-0.1, params={"sensor_name": SHANK_CONTACT_SENSOR}
    ),
    "trunk_collision": RewardTermCfg(
      func=self_collision_cost, weight=-0.25, params={"sensor_name": TRUNK_CONTACT_SENSOR}
    ),
    "pose": RewardTermCfg(
      func=go2_mdp.pose,
      weight=1.0,
      # Halved from the flat task: hips and thighs must range wider to find footing on
      # uneven ground, and some splay is actively good for stability on a slope.
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "weight": {".*_hip_joint": 0.5, ".*_thigh_joint": 0.4, ".*_calf_joint": 0.05},
      },
    ),
    "stand_still": RewardTermCfg(
      func=go2_mdp.stand_still,
      weight=-1.0,
      params={"command_name": COMMAND_NAME},
    ),
    # Rewards are scaled by dt, so at 50 Hz a weight of -1.0 costs only -0.02 per fall
    # against roughly +40 earned per 20 s episode -- far too cheap to discourage falling,
    # and more so now that two of the flat task's three early terminations are relaxed.
    "termination": RewardTermCfg(func=is_terminated, weight=-25.0),
  }

  # --- Terminations ---
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    # Relaxed from pi/4. Max terrain slope here is 24 deg and transient tilt on rubble
    # legitimately reaches 45-50; mjlab drops this term entirely for Go1 rough, but a
    # loose version is a useful safety net that speeds early learning.
    "fell_over": TerminationTermCfg(
      func=bad_orientation, params={"limit_angle": math.radians(75.0)}
    ),
    # Terrain-relative replacement for the flat task's root_height_below_minimum.
    "low_clearance": TerminationTermCfg(
      func=rugged_mdp.base_clearance_below_minimum,
      params={"sensor_name": BASE_HEIGHT_SENSOR, "minimum_height": MIN_BASE_CLEARANCE},
    ),
    # Truncation, not failure: leaving the grid is not the policy's fault.
    "out_of_bounds": TerminationTermCfg(
      func=out_of_terrain_bounds, params={"margin": 0.5}, time_out=True
    ),
    "illegal_contact": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": TRUNK_CONTACT_SENSOR, "force_threshold": 20.0},
    ),
    "state_nan": TerminationTermCfg(func=nan_detection),
  }

  # --- Events ---
  feet = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  base = SceneEntityCfg("robot", body_names=BASE_BODY)
  all_joints = SceneEntityCfg("robot", joint_names=(".*",))
  events = {
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
    # Widened from the flat task's (0.75, 1.25). Under position control the passive
    # damping we ship (0.5) stacks with the PD derivative gain, and the "right" passive
    # value is genuinely uncertain -- Menagerie says 2.0, which would be badly
    # over-damped here. This range spans the disagreement.
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
    # Only available under position actuators -- both raise TypeError on a torque <motor>,
    # which is why the flat task has never been able to randomize motor strength.
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
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        # Tighter xy than the flat task, and an explicit z lift so the robot does not
        # spawn inside a bump on generated terrain.
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
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": all_joints,
      },
    ),
    "kick": EventTermCfg(
      func=rugged_mdp.RuggedKickEvent,  # subclassed to add the missing reset hook
      mode="step",
      params={
        "asset_cfg": base,
        "base_mass": BASE_MASS,
        "wait_range_s": (0.05, 0.2),
        "duration_range_s": (0.05, 0.2),
        "magnitude_range": (0.0, 3.0),
      },
    ),
  }

  # --- Curriculum ---
  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=rugged_mdp.terrain_levels_survival, params={"min_progress_frac": 0.25}
    ),
    "command_bounds": CurriculumTermCfg(
      func=rugged_mdp.command_bounds_stages,
      params={"command_name": COMMAND_NAME, "stages": COMMAND_STAGES},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=POSITION_TIMING.decimation,
    episode_length_s=20.0,
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    curriculum=curriculum,
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=POSITION_TIMING.sim_timestep,
        cone="elliptic",
        # Down from the flat task's 100: with many simultaneous terrain contacts a high
        # impratio makes the constraint system ill-conditioned. mjlab's rough tasks use 10.
        impratio=10.0,
        iterations=10,
        ls_iterations=20,
        # Box and heightfield terrain need real continuous-collision iterations.
        ccd_iterations=500,
      ),
      # The flat task's njmax=300 is a flat-plane budget; terrain generates far more
      # simultaneous contacts.
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
    # Deterministic eval: no observation noise, no kicks, nominal physics, ~infinite
    # episode, and terrain sampled uniformly instead of by curriculum level.
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.events.pop("kick", None)
    for key in _DR_EVENT_KEYS:
      cfg.events.pop(key, None)
    cfg.terminations.pop("out_of_bounds", None)
    cfg.curriculum = {}
    # Curriculum owns terrain_levels/terrain_types otherwise; the two are mutually
    # exclusive by design.
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
