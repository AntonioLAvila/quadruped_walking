# quadruped_walking

Sim-to-real reinforcement learning for a Unitree Go2, training velocity-tracking locomotion in
[mjlab](https://github.com/mujocolab/mjlab) (manager-based, on mujoco-warp) with
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO. Started as an MIT 6.7960 class project;
originally built on Brax/MJX, since ported to mjlab.

Trained policies are exported to ONNX and validated in a **separate pydrake simulation** as a
sim-to-sim check before anything reaches hardware.

## Layout

```
go2/
  constants.py    physical constants + the GO2_ACTUATORS table (stdlib-only on purpose)
  robot.py        MJCF spec, entity and sensor builders
  mdp.py          custom command / reward / observation / event terms
  flat/           task 1: flat ground, direct torque control, 200 Hz
  rugged/         task 2: rugged terrain, PD position control, 50 Hz
scripts/          train / play / verify / check entry points
```

The MJCF is **not vendored** — `go2.robot.get_spec()` builds it from upstream
`robot_descriptions.go2_mj_description` and applies a short list of deltas.

## Commands

Run from the repo root; mjlab resolves `logs/` relative to the working directory.

```bash
# Train (defaults live in each task's rl_cfg.py). Needs CUDA in practice.
python scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2
python scripts/train.py Mjlab-Velocity-Rugged-Unitree-Go2 --env.scene.num-envs 4096

# Evaluate a checkpoint in the mjlab viewer.
python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
    --agent trained --checkpoint-file logs/rsl_rl/go2_velocity/latest/model_999.pt --num-envs 1

# Sim-to-sim verification of an exported ONNX policy in Drake + meshcat.
python scripts/verify_flat.py

# Check the upstream MJCF still matches what this repo assumes.
python scripts/check_robot.py
```

Checkpoints and the exported `model.onnx` land in `logs/rsl_rl/<experiment_name>/<run_name>/`.

<video src="https://github.com/user-attachments/assets/bf309b27-882c-4e81-8b6b-6a6bd5c9bced" controls></video>
