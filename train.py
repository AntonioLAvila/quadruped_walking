from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import Go1_Env
from argparse import ArgumentParser

import torch

torch.set_num_threads(1)


def train(args):

    def make_env(**kwargs):
        env = Go1_Env(
            torque_scale=args.torque_scale,
            history_length=args.history_length,
            noise_type=args.noise_type,
            alpha=args.alpha,
            populate_info=args.populate_info,
            obs_delay=args.obs_delay,
            **kwargs
        )
        return env

    vec_env = make_vec_env(
        make_env, n_envs=args.num_envs, seed=args.seed, vec_env_cls=SubprocVecEnv
    )
    
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99, epsilon=1e-8)

    print(f"Training on {args.num_envs} envs\nSaving models to '{args.model_dir}'")

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq // args.num_envs,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq // args.num_envs,
        save_path=args.model_dir,
        name_prefix="ckpt",
        save_vecnormalize=True,
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=2e-5,
        clip_range=0.3,
        batch_size=256,
        ent_coef=0.001,
        device='cpu'
    )

    model.learn(
        total_timesteps=args.num_steps,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=callback,
    )

    model.save(f"{args.model_dir}/final_model")
    vec_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, required=True)
    parser.add_argument("--history_length", type=int, required=True)


    parser.add_argument("--populate_info", action="store_true", help="If set, env will populate detailed info (torque, power, etc.)")
    parser.add_argument("--noise_type", type=str, required=False, default='None', help='None, HPF or LPF')
    parser.add_argument("--alpha", type=float, required=False, default=0.5)
    parser.add_argument("--torque_scale", type=float, required=False, default=1.0)
    parser.add_argument("--num_envs", type=int, required=False, default=12)
    parser.add_argument("--log_dir", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--obs_delay", type=int, required=False, default=0, help="Num env steps to delay observations by")

    args = parser.parse_args()

    train(args)
