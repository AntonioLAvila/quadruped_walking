"""MDP terms that exist only for the rugged-terrain task.

Everything terrain-agnostic is imported from ``go2.mdp`` instead of being duplicated --
this module holds only what genuinely does not exist yet:

* terrain-relative replacements for the flat task's absolute-world-z height terms,
* body-frame velocity penalties (the flat ones are world-frame and charge for climbing),
* two critic observations sourced from the terrain sensors,
* a survival-based terrain curriculum and a command-envelope ramp,
* subclasses that add the ``reset`` hooks mjlab and ``go2.mdp`` are both missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.rewards import feet_swing_height as _feet_swing_height
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply_inverse

from go2.constants import DEFAULT_HEIGHT
from go2.mdp import Go2KickEvent, Go2VelocityCommand, Go2VelocityCommandCfg

_ROBOT = SceneEntityCfg("robot")


##
# Terrain-relative height terms.
#
# The flat task reads absolute world z for base height, foot clearance and the low-height
# termination. On generated terrain that is meaningless: a robot standing correctly on a
# 0.3 m rise reads 0.57, and one in a dip reads 0.1. These read clearance above the ground
# *directly underneath*, from a TerrainHeightSensor.
##


def _clearance(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Base clearance above local terrain, [B]."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"{sensor_name} must be a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights[:, 0]


def base_height_above_terrain(
  env: ManagerBasedRlEnv, sensor_name: str, target_height: float = DEFAULT_HEIGHT
) -> torch.Tensor:
  """Penalize crouching below the nominal stance height, [B].

  Deliberately **one-sided**: only clearance *below* target is penalized. A symmetric
  term would charge the robot for the extra ride height it legitimately needs when
  stepping over rubble or cresting a rise, which is exactly the behaviour we want.
  """
  deficit = torch.clamp(target_height - _clearance(env, sensor_name), min=0.0)
  return torch.square(deficit)


def base_clearance_below_minimum(
  env: ManagerBasedRlEnv, sensor_name: str, minimum_height: float
) -> torch.Tensor:
  """Terminate when the base is dragging on the ground, [B] bool.

  Replaces ``root_height_below_minimum``, which compares absolute world z against a
  constant and is unusable on generated terrain -- it fires instantly on any patch below
  the threshold and can never fire on a patch above it.
  """
  return _clearance(env, sensor_name) < minimum_height


##
# Body-frame velocity penalties.
#
# The flat task penalizes world-frame v_z and world-frame omega_xy. On a slope, climbing
# at 1 m/s up a 0.45 grade has a legitimate world v_z of ~0.4 m/s, so the world-frame term
# charges for doing the task. Body frame measures actual bouncing and wobble instead.
##


def lin_vel_z_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


##
# Privileged (critic-only) terrain observations.
##


def base_clearance(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Base clearance above terrain, [B, 1]."""
  return _clearance(env, sensor_name).unsqueeze(-1)


def terrain_normal_b(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Local terrain surface normal in the body frame, [B, 3].

  Tells the critic which way "downhill" is without the actor ever seeing it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  normal_w = terrain_normal_from_sensors(env, sensor_names)
  return quat_apply_inverse(asset.data.root_link_quat_w, normal_w)


##
# Curriculum.
##


def terrain_levels_survival(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  min_progress_frac: float = 0.25,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> dict[str, torch.Tensor]:
  """Promote on survival + progress, demote on failure.

  Deliberately NOT mjlab's ``terrain_levels_vel``. That demotes when net displacement is
  below ``|cmd_xy| * episode_length_s * 0.5``, which assumes a *persistent* command.
  ``Go2VelocityCommand`` re-jitters on an exponential schedule with yaw rates up to
  1.5 rad/s, so net displacement is closer to a random walk while that threshold would
  demand ~10 m -- every env would demote to level 0 and stay there.

  Safe to read ``termination_manager`` here: ``_reset_idx`` runs the curriculum *before*
  ``scene.reset`` and the reset events, so this still sees the finished episode.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None
  asset: Entity = env.scene[asset_cfg.name]

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )
  # `terminated` excludes time_out/truncation, so running out the clock is not a failure.
  fell = env.termination_manager.terminated[env_ids]
  move_up = (~fell) & (distance > generator.size[0] * min_progress_frac)
  move_down = fell
  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  out: dict[str, torch.Tensor] = {"mean": levels.mean(), "max": levels.max()}
  # Under curriculum=True each column is one sub-terrain, so column index -> name. These
  # per-column means are the primary training dashboard: a column stuck at 0 is too hard,
  # one pinned at max is too easy.
  names = list(generator.sub_terrains)
  origins = terrain.terrain_origins
  if origins is not None and origins.shape[1] == len(names):
    for i, name in enumerate(names):
      mask = terrain.terrain_types == i
      if mask.any():
        out[name] = levels[mask].mean()
  return out


def command_bounds_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[tuple[int, tuple[float, float, float]], ...],
) -> dict[str, torch.Tensor]:
  """Ramp the velocity command envelope with training progress.

  mjlab's ``commands_vel`` cannot be reused: it casts to ``UniformVelocityCommandCfg`` and
  writes ``cfg.ranges``, which ``Go2VelocityCommandCfg`` does not have.
  """
  del env_ids  # Unused; the ramp is global, not per-env.
  term = env.command_manager.get_term(command_name)
  assert term is not None
  for step, bounds in stages:
    if env.common_step_counter >= step:
      term.cfg.bounds = bounds
      term.set_bounds(bounds)
  current = term.cfg.bounds
  return {
    "vx": torch.tensor(current[0]),
    "vy": torch.tensor(current[1]),
    "wz": torch.tensor(current[2]),
  }


##
# Command term with an explicit standing probability.
##


class RuggedVelocityCommand(Go2VelocityCommand):
  """``Go2VelocityCommand`` that sometimes commands a full stop.

  The inherited jitter scheme reaches all-zero on roughly 0.5% of resamples, which leaves
  ``stand_still`` and every ``command_threshold`` gate effectively dead. Standing still on
  a slope without sliding is a behaviour worth training, so make it explicit.
  """

  def set_bounds(self, bounds: tuple[float, float, float]) -> None:
    """Update the sampling envelope in place (used by ``command_bounds_stages``)."""
    self._bounds = torch.tensor(bounds, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    prob = getattr(self.cfg, "rel_standing_prob", 0.0)
    if prob > 0.0:
      standing = torch.rand(len(env_ids), device=self.device) < prob
      self._command[env_ids] = torch.where(
        standing.unsqueeze(1), torch.zeros_like(self._command[env_ids]),
        self._command[env_ids],
      )


@dataclass(kw_only=True)
class RuggedVelocityCommandCfg(Go2VelocityCommandCfg):
  rel_standing_prob: float = 0.1
  """Fraction of resamples that command a full stop."""

  def build(self, env: ManagerBasedRlEnv) -> RuggedVelocityCommand:
    return RuggedVelocityCommand(self, env)


##
# Reset hooks.
#
# The EventManager/RewardManager only call `reset` when the term object actually has the
# attribute, so adding one is sufficient. Both parent classes are missing it, which means
# a freshly reset env inherits the previous episode's state.
##


class feet_swing_height(_feet_swing_height):
  """mjlab's term plus the missing reset.

  Without this, ``peak_heights`` carries the previous episode's swing peak into the new
  one's first landing and scores it as a tracking error.
  """

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_heights[env_ids] = 0.0


class RuggedKickEvent(Go2KickEvent):
  """``Go2KickEvent`` plus the missing reset.

  Without this a freshly reset robot can be mid-kick, or inherit a cooldown, from whatever
  the previous episode was doing.
  """

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)  # type: ignore[assignment]
    n = self._num_envs if isinstance(env_ids, slice) else len(env_ids)
    self._active[env_ids] = False
    self._elapsed[env_ids] = 0.0
    self._duration[env_ids] = 0.0
    self._mag[env_ids] = 0.0
    self._dir[env_ids] = 0.0
    self._wait[env_ids] = self._sample(self._wait_range, n)
