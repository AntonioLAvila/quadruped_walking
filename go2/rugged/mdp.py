"""MDP terms that exist only for the rugged-terrain task.

Three things, because that is all mjlab and ``go2.mdp`` do not already provide:

* ``SprintVelocityCommand`` -- mjlab's uniform velocity command, with forward speed above
  cruise allowed only over the sprint lanes.
* ``feet_air_time`` -- the first-contact air-time reward, reading the right sensor field.
* ``lin_vel_z_b`` -- vertical bounce in the body frame, so climbing a slope is free.

Everything else in ``env_cfg`` is a built-in or comes from ``go2.mdp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

from go2.rugged.constants import CRUISE_SPEED, SPRINT_LANE_MARGIN, SPRINT_TERRAINS

_ROBOT = SceneEntityCfg("robot")


##
# Command.
##


class SprintVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command whose top forward speed depends on the ground underfoot.

  Over a sprint lane the sampled ``vx`` is issued as is. Anywhere else its forward part is
  scaled down so the top of the range lands on ``cruise_speed``. The rule is re-evaluated
  every step from the robot's **current** position: a robot that runs off a lane sees its
  command fall to cruise, and one that wanders onto a lane gets its sprint back.

  Two things this deliberately does not do:

  * Gate on the *spawn* column. At 3.5 m/s a robot is off its 8 m patch in about a second,
    so most of a spawn-gated sprint would be run on the neighbours.
  * Read position inside ``_resample_command``. On the reset path that runs before mjlab's
    ``forward()``, so body positions still belong to the episode that just ended.
    ``_update_command`` runs after it.

  Scaling rather than clamping keeps the off-lane distribution uniform; a clamp would pile
  a third of all technical-terrain commands onto exactly ``cruise_speed``.
  """

  cfg: SprintVelocityCommandCfg

  def __init__(self, cfg: SprintVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    origins = terrain.terrain_origins  # [rows, cols, 3]
    assert origins is not None and origins.shape[1] == len(names), (
      "sprint lanes need curriculum=True terrain (one column per sub-terrain); "
      "use the plain UniformVelocityCommandCfg otherwise"
    )
    missing = set(cfg.sprint_terrains) - set(names)
    assert not missing, f"sprint terrains {missing} are not sub-terrains: {names}"
    # Columns are laid out along world y (rows, i.e. difficulty, along x).
    assert torch.allclose(origins[0, :, 0], origins[0, :1, 0]), "columns are not along y"
    technical = [i for i, name in enumerate(names) if name not in cfg.sprint_terrains]
    self._technical_y = origins[0, technical, 1]  # [T] column centre lines
    self._keep_out = generator.size[1] / 2 + cfg.lane_margin
    self._vx_sampled = torch.zeros(self.num_envs, device=self.device)

  def on_lane(self) -> torch.Tensor:
    """Whether each robot is at least ``lane_margin`` clear of technical terrain, [B]."""
    y = self.robot.data.root_link_pos_w[:, 1]
    gap = (y[:, None] - self._technical_y[None, :]).abs().min(dim=1).values
    return gap > self._keep_out

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._vx_sampled[env_ids] = self.vel_command_b[env_ids, 0]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    top = self.cfg.ranges.lin_vel_x[1]  # ramped by the commands_vel curriculum
    scale = min(1.0, self.cfg.cruise_speed / top)
    vx = self._vx_sampled
    capped = torch.where(vx > 0.0, vx * scale, vx)
    self.vel_command_b[:, 0] = torch.where(self.on_lane(), vx, capped)
    super()._update_command(env_ids)  # heading control, then zeroes standing envs


@dataclass(kw_only=True)
class SprintVelocityCommandCfg(UniformVelocityCommandCfg):
  cruise_speed: float = CRUISE_SPEED
  """Top forward speed (m/s) off the sprint lanes."""
  sprint_terrains: tuple[str, ...] = SPRINT_TERRAINS
  """Sub-terrain names on which the full ``ranges.lin_vel_x`` applies."""
  lane_margin: float = SPRINT_LANE_MARGIN
  """How far (m) inside a lane a robot must be before it counts as on it."""

  def build(self, env: ManagerBasedRlEnv) -> SprintVelocityCommand:
    return SprintVelocityCommand(self, env)


##
# Rewards.
##


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  target_time: float = 0.2,
) -> torch.Tensor:
  """sum((air_time - target_time) * first_contact), gated by command activity.

  Reads ``last_air_time``. The sensor has already zeroed ``current_air_time`` by the step
  a foot lands, so ``go2.mdp.feet_air_time`` (which reads that) scores ``-target_time`` on
  every touchdown regardless of how long the swing was -- a footfall penalty, not an
  air-time reward. Not mjlab's ``feet_air_time`` either: that one pays per step for any
  foot currently in the air, which rewards hovering a leg.
  """
  sensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)  # [B, 4]
  air_time = sensor.data.last_air_time.clamp(max=0.5)
  rew = torch.sum((air_time - target_time) * first_contact.float(), dim=1)
  command = env.command_manager.get_command(command_name)
  return rew * (torch.norm(command, dim=1) > command_threshold).float()


def lin_vel_z_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Squared body-frame vertical velocity, [B].

  Body frame, not world: climbing a 24 deg slope at 2 m/s has a legitimate world v_z of
  0.8 m/s, and a world-frame term would charge for doing the task.
  """
  return torch.square(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 2])
