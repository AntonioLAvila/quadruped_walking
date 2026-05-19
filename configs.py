from ml_collections import config_dict

PPO_CONFIG = dict(
    num_timesteps=200_000_000,
    num_evals=10,
    num_envs=8192,
    episode_length=1500, # 7.5s
    action_repeat=1,
    unroll_length=50,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=0.01,
    discounting=0.99,
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

def default_go2_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.005,
        sim_dt=0.0025,
        action_scale=30.0,
        min_height=0.2,      # termination threshold (m)
        history_len=1,
        impl='warp', # using warp. jax is basically unusable rip non-nvidia
        naconmax=4*(2**15),
        njmax=2**7,
        naccdmax=2**13,
        soft_joint_limit_factor=0.9,
        kick_config=config_dict.create(
            kick_wait_time=[0.05, 0.2], # s
            kick_vel=[0.0, 3.0],
            kick_duration=[0.05, 0.2], # s
            enable=False
        ),
        command_config=config_dict.create( # v_xy, yaw
            # Uniform distribution for command amplitude.
            bounds=[1.5, 0.8, 1.2],
            # Probability of not zeroing out new command.
            probs=[0.9, 0.25, 0.5]
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Survival
                healthy=1e-3,
                # Tracking.
                tracking_lin_vel=3.0,
                tracking_ang_vel=1.5,
                # Base penalties.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Other.
                dof_pos_limits=-1e-3,
                pose=1e-3,
                # Other.
                termination=-5.0,
                stand_still=0,
                # Regularization.
                torques=-1e-9,
                action_rate=-1e-9,
                energy=-1e-9,
                # Feet.
                feet_clearance=-2.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=2.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                q=0.03,
                qd=1.5,
                gyro=0.2,
                gravity=0.05,
                body_v=0.1
            ),
        ),
    )