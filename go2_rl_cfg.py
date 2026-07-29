"""rsl_rl PPO configuration for the Go2 velocity task.

Cloned from mjlab's Go1 velocity ``rl_cfg.py`` and nudged toward the brax PPO setup in
``configs.py`` (same MLP sizes, entropy, discount; observation normalization on to mirror
``normalize_observations=True``). The actor reads the ``actor`` (45-dim) observation group
and the critic reads ``critic`` (120-dim), i.e. asymmetric actor-critic.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

DEFAULT_NUM_ENVS = 2**13
# 8192 is the throughput sweet spot on a 16 GB card: at 4096 env-steps/s drops to
# 184k (fixed per-step overhead dominates) and 16384 only reaches 226k vs 214k for
# 2x the memory. Scale this only if the GPU changes.

# NUM_ROLLOUT was 50, giving 8192*50 = 410k samples per iteration but only
# 48 gradient steps (6 epochs * 8 minibatches) to consume them -- one update per
# ~8.5k samples, well off the ~5k that legged-gym/mjlab configs use. Dropping to
# mjlab's default 24 (with 5 epochs * 4 minibatches below) yields ~1.7x more
# gradient updates per environment sample at the same collection cost, so a given
# reward should arrive in fewer total steps. Iterations are scaled to match: at
# 24 steps an iteration is 197k samples, so 2000 iters ~= the old 410M-sample
# budget. The previous run's reward curve knelt hard around 130M samples
# (reward 53) and only crept from 53 to 65 over the remaining 280M, so stopping
# near iteration 1000 is usually enough.
DEFAULT_NUM_IT = 2000
DEFAULT_NUM_MINIBATCH = 4
NUM_ROLLOUT = 24


def go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      num_mini_batches=DEFAULT_NUM_MINIBATCH,
      learning_rate=3e-4,
      schedule="adaptive",
      # A discount is a horizon in *seconds*, not in steps: 1/(1-gamma) steps
      # x ctrl_dt. At this repo's 200 Hz control (ctrl_dt = 0.005), the usual
      # 0.99 buys only a 0.5 s horizon -- roughly one Go2 gait cycle, so the
      # policy can barely see a full stride and has little reason to produce a
      # periodic one. mjlab's Go1 runs 0.99 at ctrl_dt = 0.02, i.e. 2.0 s.
      # 0.99 ** (0.005 / 0.02) = 0.9975 reproduces that same 2.0 s horizon at
      # our control rate.
      gamma=0.9975,
      # lam stays at 0.95 on purpose. GAE's window is 1/(1 - gamma*lam) = 19
      # steps here, which still fits inside the 24-step rollout above. Rescaling
      # lam by the same dt ratio (-> 0.987) would stretch that window to ~65
      # steps, far past the rollout boundary, so the advantages would come
      # almost entirely from the value bootstrap rather than observed reward.
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=200,  # ~10 checkpoints across DEFAULT_NUM_IT, as before.
    num_steps_per_env=NUM_ROLLOUT,
    max_iterations=DEFAULT_NUM_IT,
    # obs_groups defaults to {"actor": ("actor",), "critic": ("critic",)}.
  )
