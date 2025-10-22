from pydrake.all import (
    Meshcat,
    StartMeshcat
)
from stable_baselines3 import PPO
from env import make_gym_env, reward_fn, make_simulation_maker
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=int, required=True)
    args = parser.parse_args()
    args = vars(args)

    env, meshcat = make_gym_env(reward_fn, make_simulation_maker, visualize=True)

    model = PPO.load(args['model_path'], env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(action)
        env.render()
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, _ = env.reset()