"""Generate the rugged terrain and report its cost; optionally open it in the viewer.

    python scripts/view_terrain.py            # stats only
    python scripts/view_terrain.py --view     # also launch the MuJoCo viewer

Worth eyeballing before a long training run -- this is where a too-steep slope or
too-jagged cobble field is obvious, and where you find out what terrain generation costs
per env build.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
from mjlab.terrains import TerrainEntity  # noqa: E402

from go2.rugged.terrain import make_rugged_terrain_cfg  # noqa: E402


def main() -> None:
  play = "--play" in sys.argv
  cfg = make_rugged_terrain_cfg(play=play)
  generator = cfg.terrain_generator
  assert generator is not None

  names = list(generator.sub_terrains)
  cols = len(names) if generator.curriculum else generator.num_cols
  print(f"mode={'play' if play else 'train'}  curriculum={generator.curriculum}")
  print(f"grid: {generator.num_rows} rows x {cols} cols of {generator.size} m")
  print(f"columns: {', '.join(names)}")

  start = time.perf_counter()
  entity = TerrainEntity(cfg, device="cpu")
  model = entity.spec.compile()
  elapsed = time.perf_counter() - start

  print(f"generated in {elapsed:.1f}s -> ngeom={model.ngeom} nhfield={model.nhfield}")
  if model.nhfield:
    total = sum(int(model.hfield_nrow[i]) * int(model.hfield_ncol[i])
                for i in range(model.nhfield))
    print(f"  heightfield cells: {total:,}")

  if "--view" in sys.argv:
    import mujoco.viewer

    mujoco.viewer.launch(model, mujoco.MjData(model))


if __name__ == "__main__":
  main()
