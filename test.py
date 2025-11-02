from gymnasium.wrappers import RecordVideo
from env import A1_Env, BasicExtractor
from argparse import ArgumentParser
from stable_baselines3 import PPO
from tqdm import tqdm
import time
import numpy as np
import cv2


def test(args):
    if not args.record:
        env = A1_Env(BasicExtractor(), render_mode='human')
        sleep_time = 0.015
        env.mujoco_renderer.render('human')
        print('buh')
    else:
        env = A1_Env(
            BasicExtractor(),
            render_mode='rgb_array',
            camera_name='tracking', # TODO make this
            width=1920,
            height=1080
        )
        env = RecordVideo(env, video_folder=args.output)
        sleep_time = 0.0
    
    model = PPO.load(path=args.model_path, env=env, verbose=1)
    
    total_reward = 0
    total_length = 0
    for _ in tqdm(range(args.num_episodes)):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Slow down the rendering
            time.sleep(sleep_time)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward

    print(
        f"Avg episode reward: {total_reward / args.num_episodes}\nAvg episode length: {total_length / args.num_episodes}"
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--record', type=bool, required=False, default=False)
    parser.add_argument('--num_episodes', type=str, required=False, default=1)
    parser.add_argument('--output', type=str, required=False, default='osidjfsiodf') # TODO fix this
    args = parser.parse_args()

    test(args)