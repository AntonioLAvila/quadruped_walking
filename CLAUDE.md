# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sim-to-real RL training for a Unitree Go2 quadruped doing **torque-controlled** velocity-tracking
locomotion. Training is a faithful port to **mjlab** (manager-based, on mujoco-warp) + **rsl_rl**
PPO. A trained policy is exported to ONNX and validated in a **separate pydrake sim**
(`verification.py`) as a sim-to-sim check. (The README mentions Brax; the repo has since moved to
mjlab — there are no Brax/MJX files left in tree.)

## Commands

```bash
# Train (defaults live in go2_rl_cfg.py: 8192 envs, 1000 iters). Needs CUDA in practice.
python train_go2.py Mjlab-Velocity-Flat-Unitree-Go2
# ...with overrides (tyro CLI over the whole cfg tree):
python train_go2.py Mjlab-Velocity-Flat-Unitree-Go2 --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 --agent.run-name my_run

# Play / evaluate a checkpoint (mjlab viewer). Must import mjlab_env first to register the task.
bash eval.sh   # or the python -c one-liner it wraps, editing the --checkpoint-file path

# Sim-to-sim verification of an exported ONNX policy in Drake + meshcat.
python verification.py   # edit ONNX_POLICY_PATH inside first

# Sanity-check the (upstream) robot MJCF still matches what this repo assumes. Asserts
# topology/mass/names/margin/ctrlrange, then prints nu/nq/nv/nbody/nsensor.
python go2_robot.py
```

There is no test suite, linter, or build step. The env runs/instantiates on CPU (warp compiles CPU
kernels), which is how a quick local `ManagerBasedRlEnv(cfg, device="cpu")` smoke test works;
training wants CUDA. Checkpoints and the exported `model.onnx` land in
`logs/rsl_rl/<experiment_name>/<run_name>/` (experiment name is `go2_velocity`).

## Architecture

mjlab is **config-driven**: there is no `Go2Env` with `step`/`reset`. You assemble a
`ManagerBasedRlEnvCfg` out of observation / reward / termination / event / command / action *terms*
and register it as a task. The pieces:

- **`train_go2.py`** — registry glue only. Importing `mjlab_env` registers the task
  `Mjlab-Velocity-Flat-Unitree-Go2`; then it delegates to `mjlab.scripts.train.main`. Kept
  import-light to avoid a self-import cycle.
- **`mjlab_env.py`** — the heart. `make_go2_velocity_env_cfg(play)` builds the whole cfg (scene,
  terrain, sensors, the actor/critic observation groups, rewards, terminations, events, command,
  torque action) and `register_mjlab_task(...)` registers train + play variants.
- **`go2_mdp.py`** — all custom MDP term implementations (velocity command, rewards, privileged
  observations, the stochastic kick event, terrain-relative terminations, NaN-safety helpers).
- **`go2_robot.py`** — mjlab scene/entity/sensor *builders* (need `mujoco`/`mjlab`): the Go2
  `EntityCfg`, the feet contact sensor, and the terrain-clearance raycast sensors.
- **`go2_constants.py`** — pure physical constants **and the `GO2_ACTUATORS` table**, stdlib-only on
  purpose (see firewall below).
- **`go2_rl_cfg.py`** — the rsl_rl PPO cfg (MLP sizes, PPO hyperparams, obs normalization).
- **`verification.py`** — standalone pydrake diagram: loads the URDF, wires an
  `ObservationExtractor` → ONNX `NNPolicy` → torque `Gain` control loop.

### Cross-file contracts (the non-obvious parts)

**`go2_constants.py` is a dependency firewall.** It must import only the stdlib. Both worlds import
it: the mjlab side (`go2_robot.py`, `go2_mdp.py`, `mjlab_env.py`, which need mujoco/warp) and the
Drake side (`verification.py`, which has pydrake but *neither* mujoco nor mjlab). Putting a
`mujoco`/`mjlab` import in `go2_constants.py` breaks `verification.py`. mjlab-specific builders go in
`go2_robot.py` instead.

**`GO2_ACTUATORS` is the single source of truth for actuator dynamics, and the MJCF is not
vendored.** `go2_constants.GO2_ACTUATORS` holds per-joint-type `effort_limit` / `armature` /
`damping` / `frictionloss` / `kp` / `kd`. Both simulators configure themselves *from* it, so they
cannot drift: `go2_robot._actuator_cfgs()` passes `armature`/`frictionloss`/`viscous_damping` to
three `XmlActuatorCfg` groups (mjlab treats a non-`None` value as an override of the XML, so the
model's own numbers never matter), and `verification.py` feeds the same table to Drake's
`set_default_rotor_inertia`, `set_default_damping`, and the torque `Saturation`. **Never hard-code an
actuator parameter at a call site.** Values follow Unitree's own mjlab config
([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)), not Menagerie's generic
defaults — notably the knee armature is `0.02`, twice Menagerie's flat `0.01`.

`go2_robot.get_spec()` builds the model from upstream `robot_descriptions.go2_mj_description` and
applies exactly four deltas (margin 0, four foot sites, the `accelerometer` sensor, effort limits
from the table). Because the model now tracks upstream, `go2_robot.check_spec()` asserts topology,
total mass, geom/site/sensor names, margin, and ctrlrange — run `python go2_robot.py` after any
`robot_descriptions` bump.

**Asymmetric actor/critic observations (45 vs 120).** The actor group is 45-dim and **blind**
(joint pos/vel, base *angular* vel, projected gravity, last action, command) — that is the only
thing exported to ONNX and the only thing deployable on hardware. Base **linear** velocity is
deliberately commented out of `_actor_terms()`: there is no reliable estimate of it on the real Go2,
so the policy must not depend on it. It stays in the critic as `base_lin_vel_priv`. The critic group
is the actor's 45 plus 75 dims of **privileged** terms (true velocities, contact/air-time, kick
force). Defined in `_actor_terms()` / `_privileged_terms()` in `mjlab_env.py`; consumed by the
`obs_groups` default in `go2_rl_cfg.py`. If you add/remove/reorder an actor term you **must** mirror
the exact same layout in `verification.py`'s `ObservationExtractor` (it hand-builds a 42-dim vector +
3-dim command = 45 in that order), or the deployed policy silently gets garbage input.

**Torque (direct-effort) control.** Action is `JointEffortActionCfg` scaled by `GO2_ACTION_SCALE`
(derived from `GO2_ACTUATORS`, i.e. per-joint torque limits `[23.5, 23.5, 45.0]`): action ≈ [-1, 1]
× scale = joint torque.
`verification.py` reproduces this with a `Gain(k=ACTION_SCALE)` on the ONNX output. Because it is a
direct-torque `<motor>`/`XmlActuator`, PD/position-actuator DR (e.g. `dr.effort_limits`,
`dr.pd_gains`) does **not** apply here.

**Robustness / sim-to-real features** (all in service of the real Go2):
- *Physical domain randomization* — startup events via `mjlab.envs.mdp.dr`: base mass/inertia/CoM,
  per-foot friction (tangential + torsional/rolling), joint frictionloss/damping/armature, encoder
  bias. Keys are collected in `_DR_EVENT_KEYS` and popped in `play` mode.
- *Randomized terrain* — `make_go2_terrain_cfg()` (flat / rough / pyramid slopes / waves), with a
  `randomize_terrain` reset event reassigning each env to a new patch each episode.
- *Terrain-relative heights* — the height-based reward/termination terms (`healthy`,
  `feet_clearance`, `feet_height`, `low_height`) read raycast **clearance** via
  `go2_mdp._clearance()`, not absolute world z, so they stay valid on slopes. Do not revert these to
  `root_link_pos_w[:, 2]`.
- *Kick* — `go2_mdp.Go2KickEvent` applies a real half-sine external force (the original Brax kick
  was a no-op; this is the deliberate functional version).

**NaN safety — do not remove.** At scale, heightfield contacts occasionally NaN the warp solver, and
rsl_rl aborts training on a single non-finite obs/reward. mjlab's `nan_detection` only inspects
qpos/qvel/qacc/qacc_warmstart/sensordata — it is **blind** to derived quantities (raycast heights,
`qfrc_actuator`, site velocities) that the critic reads. Three layers defend against this:
`go2_mdp._clearance()` (sanitize+clamp at the height source), the `state_nan` termination (physics
*and* derived-state check, plus `reset_qacc_warmstart` so a reset env comes back clean), and the
`finite()` wrapper applied to every obs and reward term in `mjlab_env.py`. `finite()` must preserve
class-ness for mjlab's class-based terms (`pose`, `feet_slip`, `feet_height`) because managers use
`inspect.isclass` to decide whether to instantiate them.

**`play=True` mode** strips observation noise, the kick, and all DR, shrinks the terrain grid, and
makes episodes ~infinite — for deterministic evaluation. Pass it when building an eval cfg.

### Timing

`sim.timestep = 0.0025`, `decimation = 2` ⇒ control dt `0.005 s` (`CTRL_DT` in `go2_constants.py`,
also used to drive `verification.py`'s discrete update rate). Keep these three in sync.
