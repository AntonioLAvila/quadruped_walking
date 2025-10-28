from env import A1_Env, BasicExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_steps', type=int, required=False, default=45_000_000)
    parser.add_argument('--model_path', type=str, required=False, default='buh')
    parser.add_argument('--checkpoint_freq', type=int, required=False, default=10_000_000)
    parser.add_argument('--visualize', type=bool, required=False, default=False)
    parser.add_argument('--n_envs', type=int, required=False, default=8)
    args = parser.parse_args()
    args = vars(args)

    env = A1_Env(BasicExtractor, visualize=args['visualize'])
    
    # Create output directories
    os.makedirs(args['model_path'], exist_ok=True)
    checkpoint_dir = os.path.join(args['model_path'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # # Initialize PPO
    # model = PPO('MlpPolicy', env, verbose=1)
    
    # Define checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args['checkpoint_freq'],
        save_path=checkpoint_dir,
        name_prefix='ppo_a1_checkpoint'
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8192,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        device="cuda",
        verbose=1
    )


    # Train with checkpoints
    model.learn(
        total_timesteps=args['n_steps'],
        progress_bar=True,
        callback=checkpoint_callback
    )

    # # Save final model
    model.save(os.path.join(args['model_path'], 'ppo_a1_final'))
