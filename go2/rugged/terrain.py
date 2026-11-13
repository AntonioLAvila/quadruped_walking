"""Rugged terrain: open ground, then hills, slopes, jagged ground and shallow steps.

Scope is deliberately bounded. This is a robustness upgrade over flat ground, not a
parkour course: the policy is blind and should never need to plan a sequence of footholds.
So every sub-terrain here is *continuously traversable* -- nothing with a ``floor_depth``
(pits), no stepping stones, no beams, and no stairs above ankle height.

Run ``python scripts/view_terrain.py`` to look at the mix before training on it.
"""

from __future__ import annotations

from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  perlin_noise,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  random_spread_boxes,
)

# 8 m patches. At 3 m/s a robot crosses one in under three seconds, so it spends much of
# an episode on its neighbours rather than on the column it spawned in.
PATCH_SIZE = (8.0, 8.0)
NUM_LEVELS = 8


def _sub_terrains() -> dict[str, SubTerrainCfg]:
  """Fresh cfg objects per call. **Dict order is column order.**

  Deliberately not module-level constants and deliberately not ``dataclasses.replace`` of
  a shared cfg: ``replace`` is a shallow copy, and ``TerrainGenerator.__init__`` mutates
  ``sub_cfg.size`` on whatever objects it is handed. Two env cfgs built from a shared dict
  would alias each other.

  Nine columns, down from twelve. Dropped after a 6000-iteration run: ``cobbles`` ended at
  level 0.39 of 7 (a +/-8 cm grid is not learnable blind), ``waves`` duplicated ``hills``
  and ``kerbs`` duplicated ``rubble``.
  """
  return {
    # --- Open ground ------------------------------------------------------------------
    # The 30% of robots that can actually reach the top of the command range. Kept
    # adjacent so they form one 16 m x 64 m strip: ``scripts/eval_rugged.py`` measures
    # top speed by running the length of each.
    "flat": flat(proportion=0.15),
    # Fixed severity (``scale_with_difficulty`` off), so every row is the same pavement:
    # running fast must survive ground that is not a perfect plane.
    "rough_light": random_rough(
      proportion=0.15, noise_range=(0.01, 0.04), noise_step=0.005, border_width=0.25
    ),
    # --- Technical terrain ------------------------------------------------------------
    # Jagged ground that ramps with level, replacing the old fixed-severity column that
    # sat at level 1.8 for an entire run because it had no easy end to start from.
    "rough": random_rough(
      proportion=0.12,
      noise_range=(0.02, 0.12),
      noise_step=0.01,
      border_width=0.25,
      scale_with_difficulty=True,
    ),
    # Smooth, difficulty-scaled relief.
    "hills": perlin_noise(
      proportion=0.14,
      height_range=(0.05, 0.35),
      scale=8.0,
      octaves=3,
      persistence=0.4,
      border_width=0.5,
    ),
    # Sustained slopes. 0.45 rise/run = 24 deg, matched to the Go2's ~22 deg spec.
    # mjlab's ROUGH_TERRAINS_CFG uses 1.0 (45 deg), which no foot friction in the DR
    # range can hold.
    "slope": hf_pyramid_slope(
      proportion=0.10, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25
    ),
    "slope_inv": hf_pyramid_slope_inv(
      proportion=0.10, slope_range=(0.0, 0.45), platform_width=2.0, border_width=0.25
    ),
    # Shallow stairs: curbs and thresholds, not staircases. 2-8 cm rise on a 40 cm tread
    # is feelable blind; anything taller needs exteroception and foothold planning.
    "stairs_low": pyramid_stairs(
      proportion=0.08,
      step_height_range=(0.02, 0.08),
      step_width=0.40,
      platform_width=2.5,
      border_width=0.8,
    ),
    "stairs_low_inv": pyramid_stairs_inv(
      proportion=0.08,
      step_height_range=(0.02, 0.08),
      step_width=0.40,
      platform_width=2.5,
      border_width=0.8,
    ),
    # Discrete rubble. random_spread_boxes has add_floor=True, so boxes sit ON ground --
    # no voids to fall into.
    "rubble": random_spread_boxes(
      proportion=0.08,
      num_boxes=60,
      box_width_range=(0.15, 0.6),
      box_length_range=(0.15, 0.8),
      box_height_range=(0.03, 0.15),
      platform_width=1.5,
      border_width=0.25,
    ),
  }


def make_rugged_terrain_cfg(play: bool = False) -> TerrainEntityCfg:
  """Terrain grid for the rugged task.

  With ``curriculum=True`` mjlab **ignores ``num_cols``** and uses one column per
  sub-terrain, with difficulty increasing along rows. ``proportion`` then no longer sets
  the column count -- it only controls how robots are distributed across columns at spawn.
  So this is a 9-column x 8-row grid regardless of what ``num_cols`` says.
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
