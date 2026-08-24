# quadruped_walking

Sim-to-real reinforcement learning for a Unitree Go2, training velocity-tracking locomotion in
[mjlab](https://github.com/mujocolab/mjlab) (manager-based, on mujoco-warp) with
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO. Started as an MIT 6.7960 class project;
originally built on Brax/MJX, since ported to mjlab.

Trained policies are exported to ONNX and validated in a separate pydrake simulation as a
soft check.

## Layout

```
go2/
  constants.py    physical constants + the GO2_ACTUATORS table (stdlib-only on purpose)
  robot.py        MJCF spec, entity and sensor builders
  mdp.py          custom command / reward / observation / event terms
  flat/           task 1: flat ground, direct torque control, 200 Hz
  rugged/         task 2: rugged terrain, PD position control, 50 Hz
go2_mjcf/         submodule: the robot MJCF
scripts/          train / play / verify / check entry points
```

The MJCF comes from the [`go2_mjcf`](https://github.com/AntonioLAvila/go2_mjcf) submodule — a
pinned, edited copy of Menagerie's `unitree_go2`, shared with a separate trajectory-optimization
project so both agree on one robot. `go2.robot.get_spec()` loads it and applies two deltas from
`GO2_ACTUATORS` (joint dynamics and effort limits). The Drake verification scripts parse the same
file, so the sim-to-sim check compares one model against itself.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml` + `uv.lock` pin the
whole environment, Python included.

```bash
git clone --recurse-submodules git@github.com:<you>/quadruped_walking.git
# or, in an existing checkout:
git submodule update --init

uv sync    # creates .venv/ from the lockfile (downloads CUDA torch -- several GB the first time)
```

There is no need to activate the venv: `uv run` uses it, and re-syncs it if the lockfile moved.

## Commands

Run from the repo root; mjlab resolves `logs/` relative to the working directory.

```bash
# Train (defaults live in each task's rl_cfg.py). Needs CUDA in practice.
uv run scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2
uv run scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2 --env.scene.num-envs 4096

# Evaluate a checkpoint in the mjlab viewer.
uv run scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
    --agent trained --checkpoint-file logs/rsl_rl/go2_velocity/latest/model_999.pt --num-envs 1

# Sim-to-sim verification of an exported ONNX policy in Drake + meshcat.
uv run scripts/verify_flat.py

# Check the submodule MJCF still matches what this repo assumes (run after any bump).
uv run scripts/check_robot.py
```
*Note that the flat policy is direct torque control, and the rugged policy is joint position control
with the loop being closed by the Go2's motor drivers.

Checkpoints and the exported `model.onnx` land in `logs/rsl_rl/<experiment_name>/<run_name>/`.


https://github.com/user-attachments/assets/b08e5424-e194-40e9-b84c-480a516b7fd8
