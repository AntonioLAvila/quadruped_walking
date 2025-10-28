from pydrake.all import (
    Meshcat,
    StartMeshcat
)
from stable_baselines3 import PPO
from env import A1_Env, BasicExtractor
from argparse import ArgumentParser
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, default='buh')
    args = parser.parse_args()
    args = vars(args)

    env = A1_Env(BasicExtractor, visualize=True)
    env.simulator.set_target_realtime_rate(1)

    model = PPO.load(args['model_path'], env=env)

    obs, _ = env.reset()
    print(obs)
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, _ = env.reset()
            # print(obs)