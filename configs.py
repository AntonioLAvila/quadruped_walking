from ml_collections import config_dict

PPO_CONFIG = dict(
    num_timesteps=200_000_000,
    num_evals=10,
    num_envs=8192,
    episode_length=1000, # 5s
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
        action_scale=10.0,
        nominal_height=0.35,  # standing height of the Go2 base (m)
        min_height=0.25,      # termination threshold (m)
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
                healthy=2.0,
                # Tracking.
                tracking_lin_vel=2.0,
                tracking_ang_vel=2.0,
                # Base penalties.
                lin_vel_z=-1.0,
                ang_vel_xy=-0.15,
                orientation=-8.0,
                # Other.
                dof_pos_limits=-1.0,
                pose=0.1,
                # Other.
                termination=-20.0,
                stand_still=-2.0,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.005,
                energy=-0.001,
                # Feet.
                feet_clearance=-0.2,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=1.0,
            ),
            tracking_sigma=0.5,
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