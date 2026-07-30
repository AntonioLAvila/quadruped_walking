"""rsl_rl PPO configuration for the rugged-terrain Go2 task.

Same hyperparameters as the flat task -- the observation change (blind actor with a
stacked history, 45 x H dims) needs no config change, because rsl_rl discovers the input
dimension at runtime from the live observation TensorDict.

Two differences from ``go2.flat.rl_cfg``:
* ``experiment_name`` so logs and checkpoints do not collide with the flat task's,
* a larger iteration budget: rough terrain typically needs several times the flat budget,
  and the command-envelope ramp does not finish until iteration 2000.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Single source of truth: COMMAND_STAGES uses it to convert iterations to env steps.
from go2.rugged.constants import NUM_STEPS_PER_ENV

# Half the flat task's 8192. This task adds a 187-ray terrain scan, two more raycast
# sensors and heightfield terrain, and 8192 envs OOMs a 16 GB card (warp asks for a
# single 8.6 GB allocation). Raise it if you have the VRAM.
DEFAULT_NUM_ENVS = 2**12
DEFAULT_NUM_IT = 6000
DEFAULT_NUM_MINIBATCH = 8
NUM_ROLLOUT = NUM_STEPS_PER_ENV


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
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=6,
      num_mini_batches=DEFAULT_NUM_MINIBATCH,
      learning_rate=3e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_rugged",
    # Both mjlab defaults are wandb_project="mjlab" and run_name="", which would make
    # rugged runs indistinguishable from the flat task's on the dashboard. Set here
    # rather than passed on the CLI so it cannot be forgotten. The flat task keeps the
    # "mjlab" project so its existing run history stays where it is.
    wandb_project="go2-rugged",
    run_name="rugged",
    wandb_tags=("rugged-terrain", "pd-position", "50hz", "blind-history"),
    save_interval=100,
    num_steps_per_env=NUM_ROLLOUT,
    max_iterations=DEFAULT_NUM_IT,
    # obs_groups defaults to {"actor": ("actor",), "critic": ("critic",)}.
  )
