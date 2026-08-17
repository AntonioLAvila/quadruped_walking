"""Sanity-check the Go2 MJCF still matches what this repo assumes.

``go2.robot.get_spec()`` loads the ``go2_mjcf`` submodule, which pins the model by SHA --
but that model is shared with a separate trajectory-optimization project, so a bump made
for that project can still change inertias, topology, geom names or joint dynamics here.
Run this after any submodule bump:

    git submodule update --remote go2_mjcf && python scripts/check_robot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2 import robot  # noqa: E402


def main() -> None:
  m = robot.check_spec()
  print("flat / torque:")
  print(f"  nu={m.nu} nq={m.nq} nv={m.nv} nbody={m.nbody} nsensor={m.nsensor}")
  print(f"  mass={m.body_subtreemass[1]:.6f} kg  margin={m.geom_margin.max()}")

  try:
    from go2.rugged import robot as rugged_robot
  except ImportError:
    return
  mr = rugged_robot.check_spec()
  print("rugged / position:")
  print(f"  nu={mr.nu} nq={mr.nq} nv={mr.nv} nbody={mr.nbody} nsensor={mr.nsensor}")
  print(f"  mass={mr.body_subtreemass[1]:.6f} kg  margin={mr.geom_margin.max()}")


if __name__ == "__main__":
  main()
