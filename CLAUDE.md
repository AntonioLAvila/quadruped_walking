# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sim-to-real RL training for a Unitree Go2 quadruped doing velocity-tracking locomotion in **mjlab**
(manager-based, on mujoco-warp) + **rsl_rl** PPO. Trained policies are exported to ONNX and validated
in a **separate pydrake sim** as a sim-to-sim check. There are no Brax/MJX files left in tree.

**Two independent tasks.** They share `go2/{constants,robot,mdp}.py` but nothing else, and neither
task's config imports the other's:

| | `Mjlab-Velocity-Flat-Unitree-Go2` | `Mjlab-Velocity-Rugged-Unitree-Go2` |
|---|---|---|
| control | direct torque (`XmlActuatorCfg`) | PD position (`BuiltinPositionActuatorCfg`) |
| rate | 200 Hz (0.0025 × 2) | 50 Hz (0.005 × 4) |
| terrain | flat plane | generated, 12-column curriculum |
| actor obs | 45, single frame | 225 = 45 × 5 frames of history |
| critic obs | 120 | 495 |
| experiment | `go2_velocity` | `go2_rugged` |

## Commands

Run from the repo root — mjlab resolves `logs/` relative to the working directory.

```bash
# Train. Defaults live in each task's rl_cfg.py. Needs CUDA in practice.
python scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2
python scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2 --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 --agent.run-name my_run

# Play / evaluate a checkpoint in the mjlab viewer.
python scripts/play.py Mjlab-Velocity-Rugged-Unitree-Go2 \
    --agent trained --checkpoint-file logs/rsl_rl/go2_rugged/latest/model_999.pt --num-envs 1

# Sim-to-sim verification of an exported ONNX policy in Drake + meshcat (flat ground only).
python scripts/verify_flat.py
python scripts/verify_rugged.py [path/to/model.onnx]

# Sanity-check the (upstream) robot MJCF still matches what this repo assumes. Asserts
# topology/mass/names/margin/ctrlrange for both control modes.
python scripts/check_robot.py

# Assert the rugged actor observation is packed the way the deployment code assumes.
# Run this before trusting scripts/verify_rugged.py or any hardware port.
python scripts/check_obs_layout.py

# Generate the rugged terrain, print its cost, optionally open the viewer.
python scripts/view_terrain.py [--view] [--play]
```

There is no test suite, linter, or build step; the `scripts/check_*.py` files are the closest thing.
Everything instantiates and steps on CPU (warp compiles CPU kernels), which is how the smoke tests
work; training wants CUDA. Checkpoints and the exported ONNX land in
`logs/rsl_rl/<experiment_name>/<run_name>/`.

## Architecture

mjlab is **config-driven**: there is no `Go2Env` with `step`/`reset`. You assemble a
`ManagerBasedRlEnvCfg` out of observation / reward / termination / event / command / action *terms*
and register it as a task.

```
go2/
  constants.py      physical constants + the GO2_ACTUATORS table. STDLIB ONLY (see firewall).
  robot.py          MJCF spec builder, torque EntityCfg, feet contact sensor.
  mdp.py            custom MDP terms: velocity command, rewards, privileged obs, kick event.
  flat/env_cfg.py   the flat task. make_go2_velocity_env_cfg(play) + register_mjlab_task.
  flat/rl_cfg.py    its PPO cfg.
  rugged/           the rugged task, entirely self-contained (see below).
scripts/            train, play, verify_flat, verify_rugged, check_robot, check_obs_layout,
                    view_terrain. Not a package -- each puts the repo root on sys.path.
```

**Every `__init__.py` is empty, deliberately.** Task registration happens at module scope in each
`env_cfg.py`, not in `__init__.py`, so that `import go2.constants` stays free of mujoco/mjlab. See
the firewall below.

`go2/rugged/` mirrors that structure — `constants.py` (also stdlib-only), `robot.py`, `terrain.py`,
`mdp.py`, `env_cfg.py`, `rl_cfg.py` — and **modifies nothing outside itself**. It reuses the flat
task by importing it: `rugged/robot.get_spec()` wraps `go2.robot.get_spec()`, and `rugged/env_cfg`
imports every terrain-agnostic reward and observation from `go2.mdp`. `rugged/mdp.py` holds only what
does not otherwise exist.

### Cross-file contracts (the non-obvious parts)

**`go2/constants.py` is a dependency firewall.** It must import only the stdlib. Both worlds import
it: the mjlab side (`go2/robot.py`, `go2/mdp.py`, the two `env_cfg.py`, which need mujoco/warp) and the
Drake side (`scripts/verify_*.py`, which have pydrake but *neither* mujoco nor mjlab). Putting a
`mujoco`/`mjlab` import in `go2/constants.py` breaks the Drake scripts. mjlab-specific builders go in
`go2/robot.py` instead. `go2/rugged/constants.py` is stdlib-only for the same reason, and every
`__init__.py` stays empty so that importing a constants module never pulls in mjlab.

**`GO2_ACTUATORS` is the single source of truth for actuator dynamics, and the MJCF is not
vendored.** `go2.constants.GO2_ACTUATORS` holds per-joint-type `effort_limit` / `armature` /
`damping` / `frictionloss` / `kp` / `kd`. Both simulators configure themselves *from* it, so they
cannot drift: `go2/robot.py::_actuator_cfgs()` passes `armature`/`frictionloss`/`viscous_damping` to
three `XmlActuatorCfg` groups (mjlab treats a non-`None` value as an override of the XML, so the
model's own numbers never matter), and `scripts/verify_*.py` feed the same table to Drake's
`set_default_rotor_inertia`, `set_default_damping`, and the torque `Saturation`. **Never hard-code an
actuator parameter at a call site.** Values follow Unitree's own mjlab config
([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)), not Menagerie's generic
defaults — notably the knee armature is `0.02`, twice Menagerie's flat `0.01`.

`go2.robot.get_spec()` builds the model from upstream `robot_descriptions.go2_mj_description` and
applies exactly four deltas (margin 0, four foot sites, the `accelerometer` sensor, effort limits
from the table). Because the model now tracks upstream, `check_spec()` asserts topology,
total mass, geom/site/sensor names, margin, and ctrlrange — run `python scripts/check_robot.py` after any
`robot_descriptions` bump.

**Asymmetric actor/critic observations (flat task: 45 vs 120).** The actor group is 45-dim and **blind**
(joint pos/vel, base *angular* vel, projected gravity, last action, command) — that is the only
thing exported to ONNX and the only thing deployable on hardware. Base **linear** velocity is
deliberately commented out of `_actor_terms()`: there is no reliable estimate of it on the real Go2,
so the policy must not depend on it. It stays in the critic as `base_lin_vel_priv`. The critic group
is the actor's 45 plus 75 dims of **privileged** terms (true velocities, contact/air-time, kick
force). Defined in `_actor_terms()` / `_privileged_terms()` in `go2/flat/env_cfg.py`; consumed by the
`obs_groups` default in `go2/flat/rl_cfg.py`. If you add/remove/reorder an actor term you **must** mirror
the exact same layout in `scripts/verify_flat.py`'s `ObservationExtractor` (it hand-builds a 42-dim vector +
3-dim command = 45 in that order), or the deployed policy silently gets garbage input.

**Torque (direct-effort) control — the flat task only.** Action is `JointEffortActionCfg` scaled by `GO2_ACTION_SCALE`
(derived from `GO2_ACTUATORS`, i.e. per-joint torque limits `[23.5, 23.5, 45.0]`): action ≈ [-1, 1]
× scale = joint torque.
`scripts/verify_flat.py` reproduces this with a `Gain(k=ACTION_SCALE)` on the ONNX output. Because it is a
direct-torque `<motor>`/`XmlActuator`, PD/position-actuator DR (e.g. `dr.effort_limits`,
`dr.pd_gains`) does **not** apply here.

**Robustness / sim-to-real features** (both tasks unless noted):
- *Physical domain randomization* — startup events via `mjlab.envs.mdp.dr`: base mass/inertia/CoM,
  per-foot friction (tangential + torsional/rolling), joint frictionloss/damping/armature, encoder
  bias. Keys are collected in `_DR_EVENT_KEYS` and popped in `play` mode. The rugged task adds
  `dr.pd_gains` and `dr.effort_limits`, which **only work under position actuators** — both raise
  `TypeError` on a torque `<motor>`, which is why the flat task cannot randomize motor strength.
- *Actuator command delay* — modelled on the actuator cfg (not as an event), so it is disabled via
  `get_go2_robot_cfg(command_delay=...)` rather than popped from `events`.
- *Kick* — `go2_mdp.Go2KickEvent` applies a real half-sine external force (the original Brax kick
  was a no-op; this is the deliberate functional version).

**`play=True` mode** strips observation noise, the kick, and all DR, and makes episodes ~infinite. On
the rugged task it additionally clears the curriculum, drops `out_of_bounds`, shrinks the terrain grid
and adds a `randomize_terrain` reset event (the curriculum otherwise owns terrain levels/types, so the
two are mutually exclusive).

### The rugged task

`go2/rugged/` is a **PD position-control, 50 Hz** task on generated terrain. Beyond the config, four
things are worth knowing before touching it.

**`rugged/robot.get_spec()` adds two deltas on top of `go2.robot.get_spec()`, and both are load-bearing.**
It names the 19 anonymous group-3 collision geoms (upstream names only the four feet, so contact
sensors cannot otherwise address the thighs/shanks/trunk — and matching by *body* is not a workaround,
since the foot geom is a child of the calf body). And it **deletes the `<actuator>` block**:
`BuiltinPositionActuatorCfg` *adds* `<position>` elements without removing Menagerie's 12 `<motor>`s,
which would give `nu=24`, a bogus keyframe torque bias on the orphans, and a corrupted joint→ctrl map
in mjlab's ONNX metadata exporter.

**The actor is blind; terrain scans are critic-only.** `terrain_scan` (187 rays), `foot_height_scan`
and `base_height_scan` feed the critic and the reward terms, never the policy. That is what makes the
exported policy deployable without an elevation map. Do not "helpfully" add a height scan to the actor.

**Every height term is terrain-relative.** `base_height_above_terrain`, `feet_clearance`,
`feet_swing_height` and the `low_clearance` termination read `TerrainHeightSensor` clearance, not
`root_link_pos_w[:, 2]`. The flat task's `root_height_below_minimum` compares absolute world z against
a constant and is *unusable* on generated terrain — it fires instantly on any patch below the
threshold and can never fire above it. Do not port it across.

**The terrain curriculum is custom for a reason.** `rugged/mdp.terrain_levels_survival` promotes on
survival plus progress. mjlab's `terrain_levels_vel` demotes when displacement is under
`‖cmd_xy‖ · episode_length_s · 0.5`, which assumes a *persistent* command; `Go2VelocityCommand`
re-jitters on an exponential schedule, so net displacement is closer to a random walk and every env
would pin at level 0.

**Observation history layout — the contract that breaks silently.** The actor obs is
`ACTOR_TERM_WIDTHS × HISTORY_LENGTH`, laid out **term-major, oldest→newest**:
`[joint_pos t-4..t | joint_vel | ang_vel | gravity | last_action | command]`. This is the *opposite*
of legged_gym's time-major stacking. Anything rebuilding the observation outside mjlab —
`scripts/verify_rugged.py`, and eventually the robot — must match exactly and gets no error if it
doesn't. `scripts/check_obs_layout.py` asserts it against the live env with `atol=0`; run it after any
actor-term change. `HISTORY_LENGTH` is the single knob, but changing it changes the ONNX input width
and forces a retrain.

Two upstream bugs are worked around by subclassing in `rugged/mdp.py`: mjlab's `feet_swing_height` and
`go2_mdp.Go2KickEvent` both lack a `reset` method, so their per-env state leaks across episode resets.
The flat task still has both.

**`COMMAND_BOUNDS` is capped at 2.0 m/s for a measured reason — don't raise it casually.** A full
6000-iteration run at 2.5 m/s established the boundary: ramping to 2.0 cost tracking 1.10 → 0.76 and
it recovered to 0.98 while terrain kept climbing 3.6 → 4.4, but ramping to 2.5 cost 0.98 → 0.71 and it
never recovered, with terrain flat (+0.09 over 3500 iterations) and falls doubled. Past 2.0 the policy
spends its capacity chasing an unreachable command instead of improving on terrain. Going faster needs
exteroception or a longer history, not a bigger number.

Related trap: `command_bounds_stages` compares against `env.common_step_counter`, which counts
**env steps, not iterations**. `_COMMAND_STAGE_ITERS` is written in iterations and converted in one
place; writing raw iteration numbers into `COMMAND_STAGES` makes the ramp fire ~50× early, which
collapses the policy onto standing still (standing banks the full `upright` + `pose` reward while an
unreachable command makes tracking hopeless either way).

### Timing

The two tasks run at different rates, and each owns its own constants. Keep sim timestep, decimation
and the Drake script's discrete update rate in sync **within** a task; they need not match across.

| task | source | timestep | decimation | control dt |
|---|---|---|---|---|
| flat | `CTRL_DT` in `go2/constants.py` | 0.0025 | 2 | 0.005 s (200 Hz) |
| rugged | `POSITION_TIMING` in `go2/rugged/constants.py` | 0.005 | 4 | 0.02 s (50 Hz) |

`go2/constants.py`'s `SIM_TIMESTEP` / `DECIMATION` / `CTRL_DT` describe the **flat** task only;
`scripts/verify_flat.py` reads them. The rugged task and `scripts/verify_rugged.py` read
`POSITION_TIMING` instead. 50 Hz matches mjlab's own rough-terrain reference and every Unitree
deployment stack — a PD position policy at 200 Hz would be four times faster than anything that ships.
