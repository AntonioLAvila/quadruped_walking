PPO_CONFIG = dict(
    num_timesteps=200_000_000,
    num_evals=10,
    num_envs=8192,
    episode_length=1000,
    action_repeat=1,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=0.01,
    discounting=0.97,
    reward_scaling=1.0,
    max_grad_norm=1.0,
    normalize_observations=True,
    num_resets_per_eval=10,
)

NETWORK_FACTORY_CONFIG = dict(
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
