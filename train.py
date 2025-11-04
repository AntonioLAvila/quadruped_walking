from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from env import A1_Env, BasicExtractor
from argparse import ArgumentParser


def make_env(**kwargs):
    return A1_Env(observation_extractor=BasicExtractor(), **kwargs)

def train(args):
    vec_env = make_vec_env(
        make_env,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )

    print(f"Training on {args.num_envs} envs\nSaving models to '{args.model_dir}'")

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=args.model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    callback = CallbackList([eval_callback, checkpoint_callback])


    model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=args.log_dir)

    model.learn(
        total_timesteps=args.num_steps,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=callback,
    )
    
    model.save(f"{args.model_dir}/final_model")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_envs', type=int, required=True)
    parser.add_argument('--num_steps', type=int, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, required=True)

    parser.add_argument('--seed', type=int, required=False, default=0)
    args = parser.parse_args()

    train(args)