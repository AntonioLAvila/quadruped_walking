"""Evaluate a trained checkpoint in the mjlab viewer. Replaces the old ``eval.sh``.

Registers this repo's tasks, then delegates to mjlab's play CLI. Run from the repo root:

    python scripts/play.py Mjlab-Velocity-Flat-Unitree-Go2 \
        --agent trained --checkpoint-file logs/rsl_rl/go2_velocity/latest/model_999.pt --num-envs 1

Every argument after the task id is passed straight through to ``mjlab.scripts.play``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2.flat import env_cfg  # noqa: E402,F401  (registers the flat task on import)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
