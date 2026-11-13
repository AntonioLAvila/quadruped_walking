"""Measure what a trained rugged policy can actually do: how fast, and over what.

    uv run scripts/eval_rugged.py [path/to/model_N.pt] [--reps 8] [--device cuda:0]

Training logs cannot answer either question. ``Episode_Reward/tracking_lin_vel`` and
``Metrics/twist/error_vel_xy`` average over every command and every terrain, so a policy
that tracks 1 m/s perfectly and ignores 3 m/s looks the same as one that is mediocre at
both. This runs fixed commands on fixed terrain cells and reports each one separately.

Two tables, from one batch of environments with noise, latency, pushes and domain
randomization all off:

**Speed.** Robots start at one end of each open-ground column facing down it and are
commanded a constant forward speed. Reported: the speed they settle at, how many fell, and
how often the gait asks a motor for torque outside the real motor's torque-speed envelope.
The simulated actuators deliver full torque at any joint speed; the real ones do not, so a
gait that leans on that corner of the envelope is one the hardware will not reproduce.

**Terrain.** One group of robots per (sub-terrain, difficulty level, speed), started at the
centre of the patch facing +x. A robot succeeds if it reaches the patch edge without
falling, averaging at least half its commanded speed. Pyramid terrains have their platform
at the centre, so ``slope``/``stairs_low`` are the descent and the ``_inv`` columns are the
climb.
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import _torch_linalg  # noqa: E402,F401  (must precede any GPU torch.linalg call)
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg  # noqa: E402
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from go2.rugged import env_cfg  # noqa: E402
from go2.rugged.constants import POSITION_TIMING, TOP_SPEED  # noqa: E402
from go2.rugged.rl_cfg import go2_rugged_ppo_runner_cfg  # noqa: E402
from go2.rugged.terrain import make_rugged_terrain_cfg  # noqa: E402

# The two columns a robot can hold a speed down the whole length of (64 m each).
SPEED_TERRAINS = ("flat", "rough_light")
SPEED_COMMANDS = tuple(1.0 + 0.5 * i for i in range(int((TOP_SPEED - 1.0) / 0.5) + 1))
# The whole command range is issued on every terrain in training, so all of it is scored.
TERRAIN_COMMANDS = (1.0, 2.0, TOP_SPEED)

# The run is scored over this window: long enough after the start for the robot to have
# finished accelerating, short enough that the fastest command stays on the 64 m column.
SPEED_SETTLE_S = 3.0
SPEED_END_S = 8.0
# Reaching this fraction of the half-patch counts as crossing it (mjlab's
# ``terrain_edge_reached`` uses the same number).
EDGE_FRACTION = 0.95
# A crossing only counts if the robot averaged at least this fraction of its command.
MIN_SPEED_FRACTION = 0.5

# The real Go2 motor envelope, per joint type: (stall torque N*m, no-load speed rad/s).
# Datasheet figures; the knee is geared 1.5x further, hence more torque and less speed.
# Modelled as a straight line between the two, which is the standard DC-motor
# approximation and somewhat pessimistic in the middle of the range.
MOTOR_ENVELOPE = {"hip": (23.7, 30.1), "thigh": (23.7, 30.1), "calf": (45.43, 15.70)}


def latest_checkpoint() -> Path:
  runs = sorted((REPO_ROOT / "logs/rsl_rl/go2_rugged").glob("*"))
  for run in reversed(runs):
    found = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if found:
      return found[-1]
  raise FileNotFoundError("no checkpoint found under logs/rsl_rl/go2_rugged")


def make_cfg(num_envs: int):
  """The play config, on the training terrain grid, with commands under script control."""
  cfg = env_cfg.make_go2_rugged_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  # Play mode samples a small random grid; the cells here are addressed by (level, type),
  # which needs the training layout of one column per sub-terrain.
  cfg.scene.terrain = make_rugged_terrain_cfg(play=False)
  cfg.events.pop("randomize_terrain")
  # Everyone faces +x: the speed runs go down their column, and the rest cross their patch
  # along the difficulty axis rather than into a neighbouring terrain type.
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  # A command term that never resamples and never edits what the loop below writes.
  cfg.commands[env_cfg.COMMAND_NAME] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    ),
  )
  return cfg


def main() -> None:
  parser = argparse.ArgumentParser(description="Per-speed, per-terrain rugged policy eval.")
  parser.add_argument("checkpoint", nargs="?", type=Path, help="default: latest run")
  parser.add_argument("--reps", type=int, default=8, help="robots per terrain cell")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()
  checkpoint = args.checkpoint or latest_checkpoint()
  device = args.device
  start = time.perf_counter()

  names = list(make_rugged_terrain_cfg().terrain_generator.sub_terrains)
  num_levels = make_rugged_terrain_cfg().terrain_generator.num_rows
  speed_cols = [names.index(n) for n in SPEED_TERRAINS]
  speed_reps = 2 * args.reps

  # One row per env: (kind, column, level, commanded vx). Speed-run levels are filled in
  # once the terrain exists, because which row is the -x end of a column is a layout detail.
  plan: list[tuple[int, int, int, float]] = []
  for col in speed_cols:
    for vx in SPEED_COMMANDS:
      plan += [(0, col, -1, vx)] * speed_reps
  for col in range(len(names)):
    for level in range(num_levels):
      for vx in TERRAIN_COMMANDS:
        plan += [(1, col, level, vx)] * args.reps
  num_envs = len(plan)

  env = ManagerBasedRlEnv(make_cfg(num_envs), device=device)
  terrain = env.scene.terrain
  origins = terrain.terrain_origins  # [rows, cols, 3]
  patch = terrain.cfg.terrain_generator.size[0]

  kind = torch.tensor([p[0] for p in plan], device=device)
  col = torch.tensor([p[1] for p in plan], device=device)
  level = torch.tensor([p[2] for p in plan], device=device)
  cmd_vx = torch.tensor([p[3] for p in plan], device=device)
  is_speed_run = kind == 0
  level[is_speed_run] = origins[:, col[is_speed_run], 0].argmin(dim=0)  # the -x end of the column

  # Place every env on its cell before the wrapper's reset spawns the robots.
  terrain.terrain_levels[:] = level
  terrain.terrain_types[:] = col
  terrain.env_origins[:] = origins[level, col]

  wrapped = RslRlVecEnvWrapper(env)
  runner = VelocityOnPolicyRunner(wrapped, asdict(go2_rugged_ppo_runner_cfg()), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  command = env.command_manager.get_term(env_cfg.COMMAND_NAME)
  joint_types = [name.split("_")[1] for name in robot.joint_names]
  stall = torch.tensor([MOTOR_ENVELOPE[t][0] for t in joint_types], device=device)
  no_load = torch.tensor([MOTOR_ENVELOPE[t][1] for t in joint_types], device=device)

  dt = POSITION_TIMING.ctrl_dt
  edge = EDGE_FRACTION * patch / 2
  deadline = edge / (MIN_SPEED_FRACTION * cmd_vx)  # [B] seconds allowed to reach the edge
  horizon = max(SPEED_END_S, float(deadline[~is_speed_run].max()))

  fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
  crossed_at = torch.full((num_envs,), float("inf"), device=device)
  speed_sum = torch.zeros(num_envs, device=device)
  speed_n = torch.zeros(num_envs, device=device)
  outside = torch.zeros(num_envs, device=device)
  peak_qd = torch.zeros(num_envs, len(joint_types), device=device)

  # The robots stand for one step with a zero command, then get a step change to theirs --
  # the same thing a command resample looks like in training.
  wrapped.step(torch.zeros(num_envs, 12, device=device))
  command.vel_command_b[:, 0] = cmd_vx
  obs = wrapped.get_observations()

  with torch.no_grad():
    for step in range(1, int(horizon / dt) + 1):
      t = step * dt
      obs, _, dones, _ = wrapped.step(policy(obs))
      # An auto-reset zeroes the command; restore it. Envs that reset have already had
      # their outcome recorded, so what they do afterwards is ignored.
      command.vel_command_b[:, 0] = cmd_vx

      alive = ~fell
      # Episodes never time out here, so any `done` is a fall. A terrain robot that has
      # already crossed its patch is finished; a speed run is on the clock until the end.
      fell |= dones.bool() & alive & (is_speed_run | ~torch.isfinite(crossed_at))
      alive = ~fell

      offset = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs()
      at_edge = offset.max(dim=1).values > edge
      crossed_at = torch.where(
        at_edge & alive & ~torch.isfinite(crossed_at),
        torch.full_like(crossed_at, t),
        crossed_at,
      )

      if SPEED_SETTLE_S <= t <= SPEED_END_S:
        scored = is_speed_run & alive
        speed_sum += robot.data.root_link_lin_vel_b[:, 0] * scored
        speed_n += scored
        qd = robot.data.joint_vel
        tau = robot.data.qfrc_actuator
        # Motoring quadrant only: a real motor can brake harder than it can drive.
        limit = stall * (1.0 - qd.abs() / no_load).clamp(min=0.0)
        over = (tau * qd > 0.0) & (tau.abs() > limit)
        outside += over.float().mean(dim=1) * scored
        peak_qd = torch.where(scored[:, None], torch.maximum(peak_qd, qd.abs()), peak_qd)

  print(f"\ncheckpoint: {checkpoint}")
  print(f"{num_envs} envs, {horizon:.1f} s each, {time.perf_counter() - start:.0f} s wall\n")

  print(f"SPEED  (steady state over {SPEED_SETTLE_S:.0f}-{SPEED_END_S:.0f} s, "
        f"{speed_reps} robots per row)")
  print("  'outside' = share of joint-steps asking for torque beyond the real motor's"
        " torque-speed line")
  for speed_col in speed_cols:
    print(f"\n  {names[speed_col]}")
    print("    cmd m/s  reached  ratio   fell   outside   peak hip/thigh/calf rad/s")
    for vx in SPEED_COMMANDS:
      m = is_speed_run & (col == speed_col) & (cmd_vx == vx)
      scored = m & (speed_n > 0)
      reached = (speed_sum[scored] / speed_n[scored]).mean().item() if scored.any() else 0.0
      out = (outside[scored] / speed_n[scored]).mean().item() if scored.any() else 0.0
      peaks = [
        peak_qd[m][:, [i for i, jt in enumerate(joint_types) if jt == want]].max().item()
        for want in ("hip", "thigh", "calf")
      ]
      print(
        f"    {vx:7.1f}  {reached:7.2f}  {reached / vx:5.0%}  {fell[m].float().mean():5.0%}"
        f"   {out:6.1%}   {peaks[0]:5.1f} / {peaks[1]:5.1f} / {peaks[2]:5.1f}"
      )

  ok = torch.isfinite(crossed_at) & (crossed_at <= deadline)
  for vx in TERRAIN_COMMANDS:
    print(f"\nTERRAIN  success crossing half a patch at {vx:.1f} m/s  ({args.reps} robots per cell)")
    print("  " + " " * 16 + "".join(f"  L{lv}" for lv in range(num_levels)) + "    all")
    for c, name in enumerate(names):
      row = (kind == 1) & (col == c) & (cmd_vx == vx)
      cells = "".join(
        f"{ok[row & (level == lv)].float().mean().item():5.0%}" for lv in range(num_levels)
      )
      print(f"  {name:<16}{cells}  {ok[row].float().mean().item():5.0%}")
  env.close()


if __name__ == "__main__":
  main()
