from gymnasium.wrappers import RecordVideo
from env import A1_Env, BasicExtractor
from argparse import ArgumentParser
from stable_baselines3 import PPO
from tqdm import tqdm


def test(args):
    if not args.record:
        env = A1_Env(BasicExtractor(), render_mode='human')
        env.mujoco_renderer.render('human')
    else:
        env = A1_Env(
            BasicExtractor(),
            render_mode='rgb_array',
            camera_name='tracking',
            width=1920,
            height=1080
        )
        env = RecordVideo(env, video_folder=args.output)
    
    model = PPO.load(path=args.model_path, env=env, verbose=1)
    
    total_reward = 0
    total_length = 0
    for _ in tqdm(range(int(args.num_episodes))):
        obs, _ = env.reset()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # action = action*env._torque_scale
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            print('----------Info--------')
            for k,v in info.items():
                print(k, ' : ', v)
            print('\n')


            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward

    print(
        f"Avg episode reward: {total_reward / args.num_episodes}\nAvg episode length: {total_length / args.num_episodes}"
    )
    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--record', type=bool, required=False, default=False)
    parser.add_argument('--num_episodes', type=str, required=False, default=1)
    parser.add_argument('--output', type=str, required=False, default='osidjfsiodf') # TODO fix this
    args = parser.parse_args()

    test(args)