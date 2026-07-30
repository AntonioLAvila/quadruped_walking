"""Rugged terrain: hills, slopes, jagged ground and shallow steps.

Scope is deliberately bounded. This is a robustness upgrade over flat ground, not a
parkour course: the policy is blind and should never need to plan a sequence of footholds.
So every sub-terrain here is *continuously traversable* -- nothing with a ``floor_depth``
(pits), no stepping stones, no beams, and no stairs above ankle height.

Run ``python scripts/view_terrain.py`` to look at the mix before training on it.
"""

from __future__ import annotations

from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
  box_random_grid,
  discrete_obstacles,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  perlin_noise,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  random_spread_boxes,
  wave_terrain,
)

# 8 m patches: a 20 s episode at the final 2.5 m/s command covers up to 50 m, so smaller
# patches would have the robot spending most of its life in the neighbours.
PATCH_SIZE = (8.0, 8.0)
NUM_LEVELS = 8


def _sub_terrains() -> dict[str, SubTerrainCfg]:
  """Fresh cfg objects per call.

  Deliberately not module-level constants and deliberately not ``dataclasses.replace`` of
  a shared cfg: ``replace`` is a shallow copy, and ``TerrainGenerator.__init__`` mutates
  ``sub_cfg.size`` on whatever objects it is handed. Two env cfgs built from a shared dict
  would alias each other.
  """
  return {
    # Keeps flat-ground competence in distribution.
    "flat": flat(proportion=0.08),
    # --- Jagged ground -------------------------------------------------------------
    # HfRandomUniformTerrainCfg does `del difficulty` -- it has NO curriculum. Hence two
    # fixed-severity columns rather than one that ramps. Their level curves will look
    # flat in the logs; that is expected, not a bug.
    "rough_light": random_rough(
      proportion=0.12, noise_range=(0.01, 0.05), noise_step=0.005, border_width=0.25
    ),
    "rough_coarse": random_rough(
      proportion=0.12, noise_range=(0.05, 0.12), noise_step=0.02, border_width=0.25
    ),
    # --- Hills: smooth, difficulty-scaled relief. The main "hills" driver. ----------
    "hills": perlin_noise(
      proportion=0.14,
      height_range=(0.05, 0.35),
      scale=8.0,
      octaves=3,
      persistence=0.4,
      border_width=0.5,
    ),
    "waves": wave_terrain(
      proportion=0.08, amplitude_range=(0.05, 0.25), num_waves=3, border_width=0.25
    ),
    # --- Sustained slopes. 0.45 rise/run = 24 deg, matched to the Go2's ~22 deg spec.
    #     mjlab's ROUGH_TERRAINS_CFG uses 1.0 (45 deg), which is out of scope here.
    "slope": hf_pyramid_slope(
      proportion=0.09, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25
    ),
    "slope_inv": hf_pyramid_slope_inv(
      proportion=0.09, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25
    ),
    # --- Shallow stairs: curbs and thresholds, not staircases. 2-8 cm rise on a 40 cm
    #     tread is feelable blind; anything taller needs exteroception and foothold
    #     planning, which is explicitly out of scope.
    "stairs_low": pyramid_stairs(
      proportion=0.06,
      step_height_range=(0.02, 0.08),
      step_width=0.40,
      platform_width=2.5,
      border_width=0.8,
    ),
    "stairs_low_inv": pyramid_stairs_inv(
      proportion=0.06,
      step_height_range=(0.02, 0.08),
      step_width=0.40,
      platform_width=2.5,
      border_width=0.8,
    ),
    # --- Discrete rubble: the jagged extreme, difficulty-scaled. -------------------
    # random_spread_boxes has add_floor=True, so boxes sit ON ground -- no voids.
    "rubble": random_spread_boxes(
      proportion=0.08,
      num_boxes=60,
      box_width_range=(0.15, 0.6),
      box_length_range=(0.15, 0.8),
      box_height_range=(0.03, 0.15),
      platform_width=1.5,
      border_width=0.25,
    ),
    # merge_similar_heights collapses adjacent cells into larger boxes; without it this
    # column alone emits ~400 geoms per patch.
    "cobbles": box_random_grid(
      proportion=0.04,
      grid_width=0.4,
      grid_height_range=(0.0, 0.08),  # +/-0.08 -> up to a 0.16 m step between cells
      platform_width=1.5,
      merge_similar_heights=True,
    ),
    # obstacle_height_mode="fixed" is REQUIRED. The default "choice" samples from
    # [-h, -h/2, h/2, h], i.e. half the obstacles are pits sunk into the floor -- the
    # fall-into-void failure mode this terrain set exists to avoid.
    "kerbs": discrete_obstacles(
      proportion=0.04,
      obstacle_height_mode="fixed",
      obstacle_height_range=(0.02, 0.10),
      obstacle_width_range=(0.2, 0.8),
      num_obstacles=30,
      platform_width=1.5,
    ),
  }


def make_rugged_terrain_cfg(play: bool = False) -> TerrainEntityCfg:
  """Terrain grid for the rugged task.

  With ``curriculum=True`` mjlab **ignores ``num_cols``** and uses one column per
  sub-terrain, with difficulty increasing along rows. ``proportion`` then no longer sets
  the column count -- it only controls how robots are distributed across columns at spawn.
  So this is a 12-column x 8-row grid regardless of what ``num_cols`` says.
  """
  generator = TerrainGeneratorCfg(
    size=PATCH_SIZE,
    # Not mjlab's 20 m. `out_of_terrain_bounds` scales its limits with border_width, and a
    # 20 m flat apron lets robots escape the difficulty grid onto easy ground and farm
    # tracking reward there.
    border_width=3.0,
    border_height=1.0,
    num_rows=NUM_LEVELS,
    num_cols=1,  # ignored under curriculum=True; column count == len(sub_terrains)
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains=_sub_terrains(),
    color_scheme="height",
    add_lights=True,
    seed=0,  # identical terrain across runs, so learning curves are comparable
  )

  if play:
    # Deterministic eval: a small grid sampled uniformly rather than by curriculum level,
    # paired with the `randomize_terrain` reset event in the env cfg.
    generator.curriculum = False
    generator.num_rows = 4
    generator.num_cols = 6
    generator.border_width = 5.0

  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=generator,
    # Start partway up so early training is not all trivial, but not so high that a fresh
    # policy cannot make progress anywhere.
    max_init_terrain_level=3,
  )
