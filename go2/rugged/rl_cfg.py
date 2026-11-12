"""rsl_rl PPO configuration for the rugged-terrain Go2 task.

The algorithm block is the winner of gait-switching's 15-run Bayesian sweep. **That sweep
ran on flat ground**, so these are a starting point rather than a result: ``sweeps/ppo.yaml``
is retargeted at this task for when it is worth re-running.

rsl_rl reads the observation widths off the live environment, so nothing here depends on
``HISTORY_LENGTH``.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Single source of truth: the command ramp uses it to convert iterations to env steps.
from go2.rugged.constants import NUM_STEPS_PER_ENV

# Half the flat task's 8192. This task adds a 187-ray terrain scan, a raycast per foot and
# heightfield terrain, and 8192 envs OOMs a 16 GB card. Raise it if you have the VRAM.
DEFAULT_NUM_ENVS = 2**12
# The command ramp tops out at iteration 4000; this leaves 2000 at full speed.
DEFAULT_NUM_IT = 6000


def go2_rugged_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.1377,
      entropy_coef=0.001046,
      num_learning_epochs=8,
      num_mini_batches=4,
      learning_rate=3e-4,
      schedule="adaptive",
      gamma=0.9864,
      lam=0.9173,
      desired_kl=0.01174,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_rugged",
    # mjlab's defaults are wandb_project="mjlab" and run_name="", which would make these
    # runs indistinguishable from the flat task's on the dashboard.
    wandb_project="go2-rugged",
    run_name="rugged",
    wandb_tags=("rugged-terrain", "pd-position", "50hz", "blind-history", "sprint-lanes"),
    save_interval=100,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=DEFAULT_NUM_IT,
    # obs_groups defaults to {"actor": ("actor",), "critic": ("critic",)}.
  )
