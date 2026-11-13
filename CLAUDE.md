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
| terrain | flat plane | generated, 9-column curriculum |
| actor obs | 45, single frame | 225 = 45 × 5 frames of history |
| critic obs | 120 | 283 |
| top speed | 1.5 m/s | 3.0 m/s, on every terrain |
| experiment | `go2_velocity` | `go2_rugged` |

## Commands

**The environment is uv-managed.** `pyproject.toml` + `uv.lock` pin every dependency and
`.python-version` pins CPython 3.12; there is no `requirements.txt` and no venv to activate.
`uv run` syncs `.venv/` from the lockfile before each command, so it is the only supported way to
invoke anything here — a bare `python scripts/...` will not find `mjlab`. Use `uv add` / `uv remove`
to change dependencies (never `pip install`, which `uv run` would revert on the next sync), and
`uv sync` after pulling a lockfile change.

**mjlab, mujoco and mujoco-warp move in lockstep — bump them together or not at all.** mjlab pins
its siblings tightly (1.6.0 requires `mujoco~=3.11.0` *and* `mujoco-warp~=3.11.0`), while
`warp-lang` is declared unbounded as `>=1.14.0` by both and is *not* actually version-agnostic:
warp-lang 1.16.0 against mujoco-warp **3.8.1** fails to codegen the kernels
(`Referencing undefined symbol: xmat`, out of `mujoco_warp/_src/sensor.py::_frame_axis`). It pairs
fine with mujoco-warp 3.11.0, which is what the lockfile holds. The failure mode is worth
remembering because it is invisible to the cheap check: `check_robot.py` only builds specs and still
passes, while `check_obs_layout.py` instantiates an env and dies. **Run both after any bump.**

mjlab also breaks API across minors. 1.6.0 changed `CommandTerm._update_command` from `(self)` to
`(self, env_ids)`; `Go2VelocityCommand` in `go2/mdp.py` carries the new signature and mjlab raises a
`TypeError` at env construction for the old one.

Run from the repo root — mjlab resolves `logs/` relative to the working directory.

```bash
# Train. Defaults live in each task's rl_cfg.py. Needs CUDA in practice.
uv run scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2
uv run scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2 --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 --agent.run-name my_run

# Play / evaluate a checkpoint in the mjlab viewer.
uv run scripts/play.py Mjlab-Velocity-Rugged-Unitree-Go2 \
    --agent trained --checkpoint-file logs/rsl_rl/go2_rugged/latest/model_999.pt --num-envs 1

# Sim-to-sim verification of an exported ONNX policy in Drake + meshcat (flat ground only).
uv run scripts/verify_flat.py
uv run scripts/verify_rugged.py [path/to/model.onnx] [--cmd 3.0 0 0]

# Measure a rugged checkpoint: speed actually reached per forward command, and success rate
# per (terrain type, difficulty level). Headless. This is the only place either is reported.
uv run scripts/eval_rugged.py [path/to/model_N.pt]

# Re-tune the rugged task's PPO hyperparameters (15-run Bayesian sweep on wandb).
uv run wandb sweep sweeps/ppo.yaml

# Sanity-check the (upstream) robot MJCF still matches what this repo assumes. Asserts
# topology/mass/names/margin/ctrlrange for both control modes.
uv run scripts/check_robot.py

# Assert the rugged actor observation is packed the way the deployment code assumes.
# Run this before trusting scripts/verify_rugged.py or any hardware port.
uv run scripts/check_obs_layout.py

# Generate the rugged terrain, print its cost, optionally open the viewer.
uv run scripts/view_terrain.py [--view] [--play]
```

**Plain `uv run` is enough; there is no wrapper script.** torch opens
`libtorch_cuda_linalg.so` lazily and by bare filename, and when the loader credits that call to
a library whose RPATH does not cover `torch/lib` the lookup falls through to `LD_LIBRARY_PATH`
(a sourced ROS workspace) and fails with "cannot open shared object file". `scripts/_torch_linalg.py`
opens the library once by absolute path, after which the loader matches later requests by SONAME
and never searches. `train.py`, `play.py` and `eval_rugged.py` import it first; **any new GPU
entry point must too.** It replaced `bin/uv-run`, which patched `LD_LIBRARY_PATH` from outside.

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
go2_mjcf/           git submodule: the robot MJCF + meshes (see below).
scripts/            train, play, verify_flat, verify_rugged, eval_rugged, check_robot,
                    check_obs_layout, view_terrain. Not a package -- each puts the repo root
                    on sys.path.
sweeps/ppo.yaml     wandb sweep over the rugged task's PPO hyperparameters.
```

A fresh clone needs `git submodule update --init` (and `uv sync`), or nothing that touches the
robot will import.

**Every `__init__.py` is empty, deliberately.** Task registration happens at module scope in each
`env_cfg.py`, not in `__init__.py`, so that `import go2.constants` stays free of mujoco/mjlab. See
the firewall below.

`go2/rugged/` mirrors that structure — `constants.py` (also stdlib-only), `robot.py`, `terrain.py`,
`mdp.py`, `env_cfg.py`, `rl_cfg.py` — and **modifies nothing outside itself**. It reuses the flat
task by importing it: `rugged/robot.get_spec()` wraps `go2.robot.get_spec()`, and `rugged/env_cfg`
takes its terms from mjlab's built-ins first and `go2.mdp` second. `rugged/mdp.py` holds the three
things neither provides: the `forward_progress` reward, a correct
`feet_air_time`, and a body-frame `lin_vel_z`. Before adding a fourth, check `mjlab.envs.mdp` and
`mjlab.tasks.velocity.mdp`.

### Cross-file contracts (the non-obvious parts)

**`go2/constants.py` is a dependency firewall.** It must import only the stdlib. Both worlds import
it: the mjlab side (`go2/robot.py`, `go2/mdp.py`, the two `env_cfg.py`, which need mujoco/warp) and the
Drake side (`scripts/verify_*.py`, which have pydrake but *neither* mujoco nor mjlab). Putting a
`mujoco`/`mjlab` import in `go2/constants.py` breaks the Drake scripts. mjlab-specific builders go in
`go2/robot.py` instead. `go2/rugged/constants.py` is stdlib-only for the same reason, and every
`__init__.py` stays empty so that importing a constants module never pulls in mjlab.

This is why `GO2_MJCF_PATH` lives in `go2/constants.py` and is built with `pathlib` — both sides load
the same MJCF now, and each parses it with its own parser (mujoco's on one side, Drake's on the
other). The Drake scripts still import no mujoco.

**`GO2_ACTUATORS` is the single source of truth for actuator dynamics.**
`go2.constants.GO2_ACTUATORS` holds per-joint-type `effort_limit` / `armature` / `damping` /
`frictionloss` / `kp` / `kd`. Both simulators configure themselves *from* it, so they cannot drift:
`go2/robot.py::get_spec()` writes the joint dynamics and effort limits onto the spec, and
`scripts/verify_*.py` feed the same table to Drake's `set_default_rotor_inertia`,
`set_default_damping`, and the torque `Saturation`. **Never hard-code an actuator parameter at a call
site.** Values follow Unitree's own mjlab config
([unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)), not Menagerie's generic
defaults — notably the knee armature is `0.02`, twice Menagerie's flat `0.01`.

**`XmlActuatorCfg` silently ignores `armature` / `frictionloss` / `viscous_damping`.** `ActuatorCfg`
documents all three as "None preserves the XML value", but only the PD/builtin actuator paths call
mjlab's joint-dynamics writers; `XmlActuator.edit_spec` just wraps the existing `<motor>` (verified
against mjlab 1.6.0). So the *flat* task's joint dynamics come from whatever `get_spec()` leaves on
the joints — which is why `get_spec()` applies `GO2_ACTUATORS` itself in `_apply_joint_dynamics()`,
and why `check_spec()` asserts the compiled `dof_armature`/`dof_damping`/`dof_frictionloss`. The
values `_actuator_cfgs()` passes are documentation only. The *rugged* task's
`BuiltinPositionActuatorCfg` really does override, so it is unaffected either way.

**The MJCF lives in the `go2_mjcf` submodule** — a pinned, edited copy of Menagerie's `unitree_go2`,
shared with a separate trajectory-optimization project. Its base is byte-identical to what
`robot_descriptions` pulls (same topology, inertias, joint ranges; total mass 15.206408 kg). The
edits it ships, which this repo therefore no longer applies: flattened `<default>` classes with every
joint/motor attribute inlined, `margin` 0, `condim` 3 on non-foot collision geoms, capsule (not
cylinder) calf geoms, all 23 collision geoms named, the four `<leg>_site` foot sites, a 30-entry
`<sensor>` block, and `home`/`tuck` keyframes.

*The flattening is load-bearing — do not "tidy" the XML back into nested `<default>` classes.* Drake
merges a default class with its **immediate parent only**, so upstream's depth-2
`front_hip`/`back_hip` classes silently parsed in Drake with axis `(0,0,1)` instead of `(0,1,0)` and
zero armature — no warning, 29× error in the thigh mass matrix. `check_spec()` asserts the four thigh
joints' axis for exactly this reason.

`go2.robot.get_spec()` loads it and applies two deltas (joint dynamics, effort limits on both
`ctrlrange` and `forcerange`) plus deleting the keyframes. `check_spec()` asserts topology, total
mass, geom/site/sensor names, foot-site offset, margin, thigh axis, ctrl/force range and joint
dynamics — run `uv run scripts/check_robot.py` after any submodule bump.

**Both simulators now load this one file.** `scripts/verify_{flat,rugged}.py` used to build the Drake
plant from `robot_descriptions`' **URDF**; they parse the same MJCF instead, so sim-to-sim compares
one model against itself rather than two descriptions that happen to agree. `robot_descriptions` is
gone from the dependency list — nothing imports it any more. Consequences worth knowing:

- Drake creates the 12 `JointActuator`s itself from the `<motor>` elements, so neither script adds
  them by hand. The actuation input port is ordered by `JointActuatorIndex`, which follows XML
  order; both scripts **assert** that equals `JOINT_NAMES` order, because a mismatch would just
  permute the legs and still produce a plausible-looking gait.
- Drake reads `armature` (already matching `GO2_ACTUATORS`) and `damping` (0.05, which does **not**
  match) off the XML. Both scripts override from the table, as before.
- Effort limits come from `<motor forcerange>` = the datasheet peaks 23.7/45.43, not the table's
  derated 23.5/45.0. Harmless: Drake's `effort_limit` is advisory on the actuation port, and both
  scripts already clamp explicitly from the table.
- Drake warns about `<site>`, `<sensor>`, `<keyframe>`, elliptic cone, and `condim=6` (it substitutes
  3). All expected — MuJoCo-only refinements that don't change the rigid-body model.
- Collision geometry improved: the URDF gave Drake unnamed shapes at its default µ=1.0, the MJCF
  gives 23 named geoms at µ=0.6 body / 0.8 feet, matching MuJoCo.

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
- *Physical domain randomization* — startup events via `mjlab.envs.mdp.dr`: base mass/inertia/CoM
  (trunk mass ×0.82–1.22 on flat; ×0.8–1.35 on rugged, set by `BASE_MASS_SCALE` and skewed heavy
  for payload),
  per-foot friction (tangential + torsional/rolling), joint frictionloss/damping/armature, encoder
  bias. Keys are collected in `_DR_EVENT_KEYS` and popped in `play` mode. The rugged task adds
  `dr.pd_gains` and `dr.effort_limits`, which **only work under position actuators** — both raise
  `TypeError` on a torque `<motor>`, which is why the flat task cannot randomize motor strength.
- *Actuator command delay* — modelled on the actuator cfg (not as an event), so it is disabled via
  `get_go2_robot_cfg(command_delay=...)` rather than popped from `events`.
- *Observation delay* (rugged only) — sensor→policy latency on the four **sensed** actor terms, set
  per `ObservationTermCfg` rather than as an event, so `play` builds the terms with
  `_actor_terms(delay=False)` instead of popping anything. **Its unit is control steps (20 ms
  at 50 Hz), while the actuator delay's unit is physics steps (5 ms)** — they model different links
  in the chain and use different clocks. `last_action` and `command` get neither noise nor delay:
  no sensor sits in either path.
- *Disturbance* — flat: `go2_mdp.Go2KickEvent` applies a real half-sine external force (the
  original Brax kick was a no-op; this is the deliberate functional version). Rugged: mjlab's
  built-in `push_by_setting_velocity` on an interval.

**`play=True` mode** strips observation noise, the kick/push, and all DR, and makes episodes
~infinite. On the rugged task it additionally clears the curriculum, drops `out_of_bounds`, shrinks
the terrain grid, adds a `randomize_terrain` reset event (the curriculum otherwise owns terrain
levels/types, so the two are mutually exclusive), and opens the command's full speed range at once
instead of waiting for the curriculum.

### The rugged task

`go2/rugged/` is a **PD position-control, 50 Hz** task on generated terrain: gait-switching's flat
task (mjlab built-ins, energy-driven gait selection) with terrain put under it. One blind policy,
one command envelope (forward speed up to `TOP_SPEED`) on every terrain.

**`rugged/robot.get_spec()` adds one delta on top of `go2.robot.get_spec()`: it deletes the
`<actuator>` block.** `BuiltinPositionActuatorCfg` *adds* `<position>` elements without removing the
MJCF's 12 `<motor>`s, which would give `nu=24`, a bogus keyframe torque bias on the orphans, and a
corrupted joint→ctrl map in mjlab's ONNX metadata exporter.

Contact sensors address collision geoms by the names `go2_mjcf` gives them
(`TRUNK_COLLISION_GEOMS` = `torso_box`/`head_cyl`/`head_sphere`; `LEG_COLLISION_GEOMS` =
`<leg>_thigh_col`, `<leg>_calf_upper`/`_lower`). Matching by *body* is not a workaround: the foot
geom is a child of the calf body, so a calf-body match fires on every footstep.

**The actor is blind; terrain scans are critic-only.** `terrain_scan` (187 rays) and
`foot_height_scan` feed the critic and the reward terms, never the policy. That is what makes the
exported policy deployable without an elevation map. Do not "helpfully" add a height scan to the actor.

**Every height term is terrain-relative.** `feet_clearance` reads `TerrainHeightSensor` clearance,
`upright` scores attitude against the fitted terrain normal rather than gravity, and "the robot
collapsed" is the trunk `illegal_contact` termination. The flat task's `root_height_below_minimum`
compares absolute world z against a constant and is *unusable* on generated terrain — it fires
instantly on any patch below the threshold and can never fire above it. Do not port it across.

**One speed envelope on every terrain, and it is gait-switching's.** Forward speed ramps
0 → 1.5, 1000 → 2.5, 2000 → 3.0 m/s (`_LIN_VEL_X_STAGE_ITERS`; `TOP_SPEED` is its last rung), with
lateral ±0.5 and yaw ±1.0. Two things were tried and removed, and neither should come back without
a reason the first attempt did not have:

- *A ladder to 4.0 m/s.* It worked by the numbers — 3.36 m/s reached at a 3.5 command, 2.87 at 3.0,
  no falls — and produced a gait that hops at every speed. Asking for more than the robot has makes
  the optimizer buy speed with flight time, and one policy carries that gait down to 1 m/s.
  **`eval_rugged.py` cannot see this**: its tables looked good for that run. Watch a checkpoint in
  the viewer before believing it.
- *Sprint lanes* (`SprintVelocityCommand`: speeds above 2.0 m/s only while over the `flat` and
  `rough_light` columns). They existed to make the 4.0 ladder survivable on rough ground and went
  with it. The command is now mjlab's `UniformVelocityCommandCfg`, unmodified.

What that leaves open: 3.0 m/s is also commanded on stairs and rubble, where a blind robot will not
reach it. An earlier version of this task that commanded 2.5 m/s on every terrain saw tracking fall
0.98 → 0.71 and terrain progress stall — but that was under a curriculum that promoted on survival
alone and without the `progress` reward, so it is a warning rather than a prediction.
`terrain_levels_vel` demotes a robot that covers under half its commanded distance, so expect
terrain levels to settle lower than they would under a slower command.

**The `progress` reward is what makes an out-of-reach command safe — do not remove it or fold it
into tracking.** The tracking kernel `exp(-err²/0.4)` is worth nothing, and has no gradient, once the
robot is ~1 m/s short of its command. Under a 3.5 m/s command a robot that tops out at 2.7 therefore
earns the same tracking reward flat out as standing, and standing collects yaw tracking + `pose` +
`upright` for free. A run with the 4.0 ladder and no `progress` term did exactly that: 2.5 m/s
tracked at 99% by iteration 2400, then every command from 3.0 up produced a robot standing still,
with zero falls. `rugged.mdp.forward_progress` pays linearly for commanded speed made good, so an
out-of-reach command means "go as fast as you can". With the ladder at 3.0 the trap is no longer on
flat ground, but it is still there on every terrain the robot cannot cross at 3.0.

**The training curves do not show this failure.** Through that whole run
`Episode_Reward/tracking_lin_vel` sat at ~1.5/2.0 and reward looked healthy, because the commands
that failed were ~9% of envs; the only hint was `Metrics/twist/error_vel_xy` creeping from 0.18 to 0.39 as the
ladder rose. Run `eval_rugged.py` on a checkpoint before believing a run.

Related trap: mjlab's `commands_vel` compares against `env.common_step_counter`, which counts
**env steps, not iterations**. `_LIN_VEL_X_STAGE_ITERS` is written in iterations and converted in one
place; writing raw iteration numbers into the curriculum makes the ramp fire ~50× early.

**The simulated motors are ideal.** `BuiltinPositionActuatorCfg` delivers the full torque limit at
any joint speed; the real motors lose torque as they spin up (about 30 rad/s no-load at hip/thigh,
15.7 at the knee). That gap is widest at the top of the speed range. `eval_rugged.py`'s `outside`
column reports how often the gait leans on torque the hardware does not have — read it before
believing a top speed. Training against the curve means `DcMotorActuatorCfg`, and that breaks
mjlab's ONNX metadata exporter, which reads PD gains off `gainprm`/`biasprm` (1 and 0 for a
`<motor>`).

**Both curricula are mjlab's.** `terrain_levels_vel` promotes a robot that got more than half a
patch from its spawn and demotes one that covered under half the distance its command asked for.
It needs the command to produce net displacement, which is why `heading_command` is on: a robot
holding a yaw rate walks a circle and is demoted however well it tracks. The previous custom rule
promoted on survival alone, and tracking ended at 0.80/2.0 — robots were promoted onto terrain they
could stand on but not walk across. Expect mean level to drop to ~0 in the first few hundred
iterations (nothing travels yet) and climb from there.

**Observation history layout — the contract that breaks silently.** The actor obs is
`ACTOR_TERM_WIDTHS × HISTORY_LENGTH`, laid out **term-major, oldest→newest**:
`[joint_pos t-4..t | joint_vel | ang_vel | gravity | last_action | command]`. This is the *opposite*
of legged_gym's time-major stacking. Anything rebuilding the observation outside mjlab —
`scripts/verify_rugged.py`, and eventually the robot — must match exactly and gets no error if it
doesn't. `scripts/check_obs_layout.py` asserts it against the live env with `atol=0`; run it after any
actor-term change. `HISTORY_LENGTH` is the single knob, but changing it changes the ONNX input width
and forces a retrain. History is set on the actor *group*; mjlab still keeps one buffer per term, so
the layout is the same as setting it per term.

**`go2.mdp.feet_air_time` is a footfall penalty, and the flat task still uses it.** It reads
`current_air_time`, which mjlab's contact sensor has already zeroed on the step a foot lands, so it
scores `-target_time` per touchdown however long the swing was. `rugged.mdp.feet_air_time` reads
`last_air_time` instead. The flat task is left as is on purpose: fixing it changes that task's reward.

**The PPO hyperparameters were tuned on flat ground.** `rugged/rl_cfg.py` carries the winner of
gait-switching's sweep. `Train/mean_reward` is the *stochastic* rollout's reward, so its low
`entropy_coef` wins partly by shrinking action noise; treat the values as a starting point and
cross-check any re-sweep with `eval_rugged.py`.

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
