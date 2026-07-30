from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from go2.constants import (
  BASE_BODY,
  BASE_MASS,
  DEFAULT_HEIGHT,
  FEET,
  FEET_CONTACT_SENSOR,
  FOOT_GEOMS,
  FOOT_SITES,
  FEET_MIN_HEIGHT
)

from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class Go2VelocityCommandCfg(CommandTermCfg):
  """Config for the Go2 velocity command term.

  Reproduces env.py: command [vx, vy, wz] sampled uniformly in [-bounds, bounds] on a
  fresh episode, then jittered per-axis at each resample; resample interval is
  Exponential with mean ``mean_resample_time_s`` seconds.
  """

  resampling_time_range: tuple[float, float] = (5.0, 5.0)  # unused (timing overridden).
  bounds: tuple[float, float, float] = (1.5, 0.8, 1.2)
  probs: tuple[float, float, float] = (0.9, 0.25, 0.5)
  mean_resample_time_s: float = 5.0

  def build(self, env: ManagerBasedRlEnv) -> "Go2VelocityCommand":
    return Go2VelocityCommand(self, env)


class Go2VelocityCommand(CommandTerm):
  cfg: Go2VelocityCommandCfg

  def __init__(self, cfg: Go2VelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self._bounds = torch.tensor(cfg.bounds, device=self.device)
    self._probs = torch.tensor(cfg.probs, device=self.device)
    # A "fresh" env samples a plain uniform command; otherwise it jitters the previous
    # command. Reset on every episode reset (matches env.py's reset -> U(-b, b)).
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._fresh[env_ids] = True
    return super().reset(env_ids)

  def _resample(self, env_ids: torch.Tensor) -> None:
    # Override base uniform timing with Exponential(mean=mean_resample_time_s).
    if len(env_ids) == 0:
      return
    self.time_left[env_ids] = (
      torch.empty(len(env_ids), device=self.device).exponential_()
      * self.cfg.mean_resample_time_s
    )
    self._resample_command(env_ids)
    self.command_counter[env_ids] += 1

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    y = (2.0 * torch.rand(n, 3, device=self.device) - 1.0) * self._bounds
    z = (torch.rand(n, 3, device=self.device) < self._probs).float()
    w = (torch.rand(n, 3, device=self.device) < 0.5).float()
    x_k = self._command[env_ids]
    x_jitter = x_k - w * (x_k - y * z)  # env.py _sample_command
    fresh = self._fresh[env_ids].unsqueeze(1)
    self._command[env_ids] = torch.where(fresh, y, x_jitter)
    self._fresh[env_ids] = False

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


def _command_norm(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """L2 norm of the full [vx, vy, wz] command (matches env.py's jp.linalg.norm)."""
  return torch.norm(env.command_manager.get_command(command_name), dim=1)


# =====================================================================================
# Observations (privileged extras; the actor terms all reuse built-in mdp functions).
# =====================================================================================


def feet_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """World-frame foot site linear velocities, flattened -> (B, 12)."""
  asset = env.scene[asset_cfg.name]
  vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]  # (B, 4, 3)
  return vel.reshape(vel.shape[0], -1)


def base_ang_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """World-frame base angular velocity (env.py's global_angvel) -> (B, 3)."""
  return env.scene[asset_cfg.name].data.root_link_ang_vel_w


def actuator_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Joint-space actuator force (env.py's actuator_force) -> (B, 12)."""
  asset = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
  """Per-foot binary contact -> (B, 4)."""
  return (env.scene[sensor_name].data.found > 0).float()


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str = FEET_CONTACT_SENSOR) -> torch.Tensor:
  """Per-foot air time -> (B, 4)."""
  return env.scene[sensor_name].data.current_air_time


def kick_force(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """External force applied to the base (env.py's xfrc_applied[torso, :3]) -> (B, 3)."""
  asset = env.scene[asset_cfg.name]
  f = asset.data.body_external_force[:, asset_cfg.body_ids, :]  # (B, 1, 3)
  return f.reshape(f.shape[0], -1)


def is_being_kicked(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Whether a kick force is currently applied to the base -> (B, 1)."""
  asset = env.scene[asset_cfg.name]
  f = asset.data.body_external_force[:, asset_cfg.body_ids, :]  # (B, 1, 3)
  return (torch.norm(f, dim=-1) > 0.0).float()  # (B, 1)


# =====================================================================================
# Rewards (weight = env.py scale; RewardManager applies the * dt).
# =====================================================================================


def healthy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
  return torch.clamp(z / DEFAULT_HEIGHT, 0.0, 1.0)


def tracking_lin_vel(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.25, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  v = asset.data.root_link_lin_vel_b
  err = torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1)
  return torch.exp(-err / sigma)


def tracking_ang_vel(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.25, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  w = asset.data.root_link_ang_vel_b
  err = torch.square(cmd[:, 2] - w[:, 2])
  return torch.exp(-err / sigma)


def lin_vel_z(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return torch.square(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2])


def ang_vel_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  w = env.scene[asset_cfg.name].data.root_link_ang_vel_w
  return torch.sum(torch.square(w[:, :2]), dim=1)


def stand_still(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  inactive = (_command_norm(env, command_name) < 0.01).float()
  return torch.sum(torch.abs(q - q0), dim=1) * inactive


def torques(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.sum(torch.square(tau), dim=1)) + torch.sum(torch.abs(tau), dim=1)


def energy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(qd) * torch.abs(tau), dim=1)


def feet_clearance(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, max_foot_height: float = 0.1
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (B, 4)
  vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (B, 4, 2)
  vel_norm = torch.norm(vel_xy, dim=-1)
  delta = torch.abs(foot_z - max_foot_height)
  return torch.sum(delta * vel_norm, dim=1)


def dof_pos_limits(
  env: ManagerBasedRlEnv, soft_factor: float = 0.95, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Soft joint-limit penalty matching env.py (limits scaled by 0.95 about zero)."""
  asset = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, :]  # (B, J, 2)
  soft_lo = limits[..., 0] * soft_factor
  soft_hi = limits[..., 1] * soft_factor
  lower = (soft_lo - q).clamp(min=0.0)
  upper = (q - soft_hi).clamp(min=0.0)
  return torch.sum(lower + upper, dim=1)


class pose:
  """exp(-sum(weight * (q - q0)^2)) with per-joint weights resolved by name."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset = env.scene[cfg.params["asset_cfg"].name]
    self.default_joint_pos = asset.data.default_joint_pos
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, weight = resolve_matching_names_values(cfg.params["weight"], joint_names)
    self.weight = torch.tensor(weight, device=env.device, dtype=torch.float32)

  def __call__(self, env: ManagerBasedRlEnv, weight, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    del weight  # Resolved into a tensor in __init__.
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.sum(torch.square(q - q0) * self.weight, dim=1))


def _aligned_site_ids(asset, contact_sensor, device) -> torch.Tensor:
  """Foot site ids ordered to match the contact sensor's primary (foot) order."""
  try:
    primary_names = list(contact_sensor.primary_names)
  except AttributeError:
    primary_names = list(FOOT_GEOMS)
  site_ids = []
  for pname in primary_names:
    leg = next((f for f in FEET if f in pname), None)
    if leg is None:
      raise ValueError(f"Cannot map contact primary '{pname}' to a foot site.")
    ids, _ = asset.find_sites(f"{leg}_site")
    site_ids.append(int(ids[0]))
  return torch.tensor(site_ids, device=device, dtype=torch.long)


class feet_slip:
  """sum(foot_xy_vel^2 * in_contact) gated by command activity."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset = env.scene[cfg.params["asset_cfg"].name]
    sensor = env.scene[cfg.params["sensor_name"]]
    self.site_ids = _aligned_site_ids(asset, sensor, env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.01,
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()  # (B, 4), primary order
    vel_xy = asset.data.site_lin_vel_w[:, self.site_ids, :2]  # (B, 4, 2), aligned
    vel_sq = torch.sum(torch.square(vel_xy), dim=-1)  # (B, 4)
    active = (_command_norm(env, command_name) > command_threshold).float()
    return torch.sum(vel_sq * in_contact, dim=1) * active


class feet_height:
  """sum((swing_peak / target - 1)^2 * first_contact) gated by command activity."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset = env.scene[cfg.params["asset_cfg"].name]
    sensor = env.scene[cfg.params["sensor_name"]]
    self.site_ids = _aligned_site_ids(asset, sensor, env.device)
    self.peak = torch.zeros(env.num_envs, len(self.site_ids), device=env.device)
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    max_foot_height: float = 0.1,
    command_threshold: float = 0.01,
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    foot_z = asset.data.site_pos_w[:, self.site_ids, 2]  # (B, 4)
    in_air = sensor.data.found == 0  # (B, 4)
    self.peak = torch.where(in_air, torch.maximum(self.peak, foot_z), self.peak)
    first_contact = sensor.compute_first_contact(dt=self.step_dt)  # (B, 4) bool
    error = self.peak / max_foot_height - 1.0
    active = (_command_norm(env, command_name) > command_threshold).float()
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    self.peak = torch.where(first_contact, torch.zeros_like(self.peak), self.peak)
    return cost


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  target_time: float = 0.25
) -> torch.Tensor:
  """sum((air_time - target_time) * first_contact) gated by command activity."""
  sensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time  # (B, 4)
  first_contact = sensor.compute_first_contact(dt=env.step_dt)  # (B, 4) bool
  rew = torch.sum((air_time.clamp(max=0.5) - target_time) * first_contact.float(), dim=1)
  active = (_command_norm(env, command_name) > command_threshold).float()
  return rew * active


# =====================================================================================
# Event: stochastic kick (functional reconstruction of env.py's _handle_kick).
# =====================================================================================


class Go2KickEvent:
  """Periodic external-force kick on the base, applied over a sampled duration.

  Faithful to the *intent* of env.py's ``_handle_kick``: after a random wait, apply a
  horizontal force with a half-sine envelope ``0.5*sin(pi * t/T)`` scaled by the base
  mass and a sampled magnitude, in a random heading, then resample and repeat.

  NOTE: env.py's original ``_handle_kick`` is effectively a no-op (``kick_dir`` is never
  assigned, so the applied force is always zero, and the step counters are tangled dead
  code). This implements the evident intent so the kick is a real disturbance and the
  privileged kick observations carry signal. Use ``mode="step"``.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._device = env.device
    self._num_envs = env.num_envs
    self._step_dt = env.step_dt
    self._asset = env.scene[p["asset_cfg"].name]
    self._body_ids = p["asset_cfg"].body_ids  # local base body id(s)
    self._mass = float(p.get("base_mass", BASE_MASS))
    self._wait_range = p.get("wait_range_s", (0.05, 0.2))
    self._duration_range = p.get("duration_range_s", (0.05, 0.2))
    self._magnitude_range = p.get("magnitude_range", (0.0, 3.0))

    z = lambda: torch.zeros(self._num_envs, device=self._device)  # noqa: E731
    self._active = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
    self._elapsed = z()
    self._duration = z()
    self._mag = z()
    self._dir = torch.zeros(self._num_envs, 3, device=self._device)
    self._wait = self._sample(self._wait_range, self._num_envs)

  def _sample(self, rng: tuple[float, float], n: int) -> torch.Tensor:
    lo, hi = rng
    return torch.rand(n, device=self._device) * (hi - lo) + lo

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    del env, env_ids  # step events run on all envs.
    dt = self._step_dt
    force = torch.zeros(self._num_envs, 3, device=self._device)

    # Advance active kicks and compute their force.
    self._elapsed[self._active] += dt
    expired = self._active & (self._elapsed >= self._duration)
    ongoing = self._active & ~expired
    if ongoing.any():
      frac = self._elapsed[ongoing] / self._duration[ongoing]  # in (0, 1)
      u = 0.5 * torch.sin(torch.pi * frac)
      fmag = u * self._mass * self._mag[ongoing] / self._duration[ongoing]
      force[ongoing] = fmag.unsqueeze(1) * self._dir[ongoing]

    # Expire finished kicks; start their next cooldown.
    if expired.any():
      self._active[expired] = False
      self._wait[expired] = self._sample(self._wait_range, int(expired.sum()))

    # Count down waiting envs and trigger new kicks.
    waiting = ~self._active
    self._wait[waiting] -= dt
    trigger = waiting & (self._wait <= 0.0)
    if trigger.any():
      n = int(trigger.sum())
      self._active[trigger] = True
      self._elapsed[trigger] = 0.0
      self._duration[trigger] = self._sample(self._duration_range, n)
      self._mag[trigger] = self._sample(self._magnitude_range, n)
      angle = torch.rand(n, device=self._device) * 2.0 * torch.pi - torch.pi
      self._dir[trigger] = torch.stack(
        [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=1
      )

    # Write the wrench to the base for all envs (zeros where not kicking).
    torque = torch.zeros_like(force)
    self._asset.write_external_wrench_to_sim(
      force.unsqueeze(1), torque.unsqueeze(1), body_ids=self._body_ids
    )
