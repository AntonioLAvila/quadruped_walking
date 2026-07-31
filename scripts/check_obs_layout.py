"""Assert the rugged actor observation is packed the way the deployment code assumes.

    python scripts/check_obs_layout.py

This is the single most important check in the repo. The actor observation is a stacked
history, and mjlab lays it out **term-major, oldest to newest**:

    [joint_pos t-4..t | joint_vel t-4..t | ang_vel | gravity | last_action | command]

which is the *opposite* of legged_gym's time-major convention. Any deployment-side buffer
-- ``scripts/verify_rugged.py``, and eventually the real robot -- has to rebuild that
exact vector. Get the ordering wrong and nothing errors: the policy just receives garbage
and walks badly, which is indistinguishable from a bad policy.

So: drive the real environment with a fixed action sequence, rebuild the observation from
scratch in NumPy using only quantities a deployment would have, and require exact equality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from go2.rugged import env_cfg  # noqa: E402
from go2.rugged.constants import (  # noqa: E402
  ACTOR_TERM_WIDTHS as TERM_WIDTHS,
)
from go2.rugged.constants import (  # noqa: E402
  FRAME_DIM,
  HISTORY_LENGTH,
)


class HistoryPacker:
  """Reference implementation of mjlab's per-term history stacking.

  Mirror this exactly in any deployment. Two details that are easy to miss:
  the buffer is backfilled with the first frame rather than zeros, and the flat
  output is term-major (all of term A's history, then all of term B's).
  """

  def __init__(self, history_length: int, frame_dim: int = FRAME_DIM):
    self.h = history_length
    self.frames = np.zeros((history_length, frame_dim), dtype=np.float32)
    self.initialized = False

  def append(self, frame: np.ndarray) -> None:
    if not self.initialized:
      # mjlab's CircularBuffer fills every slot from the first push after a reset,
      # so the policy never sees zero-padding.
      self.frames[:] = frame
      self.initialized = True
    else:
      self.frames = np.roll(self.frames, -1, axis=0)
      self.frames[-1] = frame

  def flat(self) -> np.ndarray:
    """Term-major, oldest to newest."""
    out, offset = [], 0
    for _, width in TERM_WIDTHS:
      out.append(self.frames[:, offset : offset + width].reshape(-1))
      offset += width
    return np.concatenate(out)


def current_frame(env: ManagerBasedRlEnv) -> np.ndarray:
  """The 45-dim single frame, from quantities a real robot could measure."""
  robot = env.scene["robot"]
  data = robot.data
  parts = [
    (data.joint_pos - data.default_joint_pos)[0],
    (data.joint_vel - data.default_joint_vel)[0],
    data.root_link_ang_vel_b[0],
    data.projected_gravity_b[0],
    env.action_manager.action[0],
    env.command_manager.get_command(env_cfg.COMMAND_NAME)[0],
  ]
  return torch.cat(parts).cpu().numpy().astype(np.float32)


def main() -> None:
  cfg = env_cfg.make_go2_rugged_env_cfg(play=False)
  cfg.scene.num_envs = 1
  # Noise and sensor delay are both applied per frame *before* the frame enters the
  # history buffer, so with either on there is nothing deterministic to compare against:
  # a randomly-lagged signal cannot be reconstructed from the live state. The layout --
  # term order, stacking direction, reset backfill -- is what is under test here, and it
  # is independent of both.
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["actor"].terms = env_cfg._actor_terms(HISTORY_LENGTH, delay=False)
  cfg.events.pop("kick", None)

  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, _ = env.reset()

  actor_dim = obs["actor"].shape[-1]
  expected_dim = FRAME_DIM * HISTORY_LENGTH
  assert actor_dim == expected_dim, (
    f"actor obs is {actor_dim} dims, expected {FRAME_DIM} x {HISTORY_LENGTH} = {expected_dim}"
  )

  packer = HistoryPacker(HISTORY_LENGTH)
  packer.append(current_frame(env))
  np.testing.assert_allclose(
    packer.flat(), obs["actor"][0].cpu().numpy(), rtol=0, atol=0,
    err_msg="mismatch on the first frame after reset (backfill behaviour differs)",
  )

  rng = np.random.default_rng(0)
  for step in range(40):
    action = torch.from_numpy(rng.uniform(-1.0, 1.0, (1, 12)).astype(np.float32))
    obs, _, _, _, _ = env.step(action)
    packer.append(current_frame(env))
    np.testing.assert_allclose(
      packer.flat(), obs["actor"][0].cpu().numpy(), rtol=0, atol=0,
      err_msg=f"observation layout mismatch at step {step}",
    )

  print(f"obs layout OK: {actor_dim} dims = {FRAME_DIM} x H{HISTORY_LENGTH}, term-major")
  print("  " + " | ".join(f"{n} x{HISTORY_LENGTH}" for n, _ in TERM_WIDTHS))
  print("  matched the environment exactly over 41 frames including the reset backfill")


if __name__ == "__main__":
  main()
