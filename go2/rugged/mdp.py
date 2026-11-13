"""MDP terms that exist only for the rugged-terrain task.

Three things, because that is all mjlab and ``go2.mdp`` do not already provide:

* ``forward_progress`` -- pays for commanded speed made good, so an out-of-reach command
  means "run flat out" instead of "stand still".
* ``feet_air_time`` -- the first-contact air-time reward, reading the right sensor field.
* ``lin_vel_z_b`` -- vertical bounce in the body frame, so climbing a slope is free.

Everything else in ``env_cfg``, the velocity command included, is a built-in or comes from
``go2.mdp``.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_ROBOT = SceneEntityCfg("robot")


##
# Rewards.
##


def forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Share of the commanded planar speed the robot is actually making good, in [0, 1].

  This is what keeps a command the robot cannot reach from being a reason to stand still.
  The tracking kernel ``exp(-err^2 / sigma)`` is worth nothing, and has no gradient, once
  the robot is more than about 1 m/s short of its command -- so under a 3.5 m/s command a
  robot that tops out at 2.7 earns the same tracking reward running flat out as it does
  standing, and standing is free. A run with a 4.0 m/s ladder did exactly that: 2.5 m/s
  tracked at 99%, then every command from 3.0 up produced a robot standing still.

  The ladder now stops at 3.0, which the robot reaches on flat ground, but the same
  command is issued on stairs and rubble, where it does not -- so the trap is still there
  to fall into and this term still closes it.

  Linear in speed along the commanded direction and capped at the command, so going
  faster always pays, overshoot never does, and precision is still the kernel's job.
  """
  command = env.command_manager.get_command(command_name)[:, :2]
  speed = torch.norm(command, dim=1)
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  made_good = torch.sum(velocity * command, dim=1) / speed.clamp(min=command_threshold)
  share = (made_good / speed.clamp(min=command_threshold)).clamp(0.0, 1.0)
  return share * (speed > command_threshold).float()


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
