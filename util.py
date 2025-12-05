def get_run_id(args):
    arg_parts = [
        f"torque{args.torque_scale}",
        f"alpha{args.alpha}",
        f"noise{args.noise_type}",
        f"ep_length{args.ep_length}",
        f"obs_delay{args.obs_delay}",
        f"num_episodes{args.num_episodes}",
        f"history_len{args.history_length}"
    ]
    run_id = "_".join(arg_parts)

    return run_id

DB_NAME = "quadruped_walking_test_db.json"