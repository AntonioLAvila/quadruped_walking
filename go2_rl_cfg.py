"""rsl_rl PPO configuration for the Go2 velocity task.

Cloned from mjlab's Go1 velocity ``rl_cfg.py`` and nudged toward the brax PPO setup in
``configs.py`` (same MLP sizes, entropy, discount; observation normalization on to mirror
``normalize_observations=True``). The actor reads the ``actor`` (48-dim) observation group
and the critic reads ``critic`` (123-dim), i.e. asymmetric actor-critic.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
    # obs_groups defaults to {"actor": ("actor",), "critic": ("critic",)}.
  )
