"""Launcher that registers the Go2 task, then delegates to mjlab's training CLI.

mjlab's ``train`` entry point only imports ``mjlab.tasks``; importing ``mjlab_env`` here
registers ``Mjlab-Velocity-Flat-Unitree-Go2`` into the same registry before mjlab lists
the available tasks. This is registry glue only.

Usage:
    python train_go2.py Mjlab-Velocity-Flat-Unitree-Go2
    python train_go2.py Mjlab-Velocity-Flat-Unitree-Go2 --env.scene.num-envs 4096 \
        --agent.max-iterations 10000 --agent.run-name go2_v1

To evaluate a trained checkpoint, import this module first so the task is registered:
    python -c "import mjlab_env; from mjlab.scripts.play import main; main()" \
        Mjlab-Velocity-Flat-Unitree-Go2 --agent trained --checkpoint-file <path/to/model.pt>
"""

import mjlab_env  # noqa: F401  (registers Mjlab-Velocity-Flat-Unitree-Go2 on import)
from mjlab.scripts.train import main

DEFAULT_NUM_ENVS = 8192
DEFAULT_NUM_IT = 10_000
DEFAULT_NUM_MINIBATCH = 4
NUM_ROLLOUT = 32
NUM_SAVES = 10

if __name__ == "__main__":
  main()
