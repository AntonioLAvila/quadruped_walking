"""Launcher that registers this repo's Go2 tasks, then delegates to mjlab's training CLI.

mjlab's ``train`` entry point only imports ``mjlab.tasks``; importing the task modules here
registers ours into the same registry before mjlab lists the available tasks. This is
registry glue only -- default env/iteration counts live in each task's ``rl_cfg.py``.

Run from the repo root (mjlab resolves ``logs/`` relative to the working directory):

    python scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2
    python scripts/train.py Mjlab-Velocity-Flat-Unitree-Go2 --env.scene.num-envs 4096 \
        --agent.max-iterations 10000 --agent.run-name go2_v1
"""

import sys
from pathlib import Path

# scripts/ is not a package, so put the repo root on the path for ``go2.*``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2.flat import env_cfg as _flat  # noqa: E402,F401  (registers the flat task)
from go2.rugged import env_cfg as _rugged  # noqa: E402,F401  (registers the rugged task)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
