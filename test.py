from gymnasium.wrappers import RecordVideo
from env import Go1_Env
from argparse import ArgumentParser
from stable_baselines3 import PPO
from tqdm import tqdm
import time
import numpy as np


def test(args):
    if not args.record:
        env = Go1_Env(
            torque_scale=args.torque_scale,
            history_length=args.history_length,
            noise_type=args.noise_type,
            alpha=args.alpha,
            render_mode=args.render_mode
        )
    else:
        env = Go1_Env(
            torque_scale=args.torque_scale,
            history_length=args.history_length,
            noise_type=args.noise_type,
            alpha=args.alpha,
            render_mode='rgb_array',
            camera_name='tracking',
            width=1920,
            height=1080,
        )
        env = RecordVideo(env, video_folder=args.output)

    model = PPO.load(path=args.model_path, env=env, verbose=1, device='cpu')

    total_reward = 0
    total_length = 0
    n_falls = 0
    avg_vel = np.zeros(3)
    avg_omega = np.zeros(3)
    avg_g_proj = np.zeros(3)
    total_steps = 0

    torque_norm_sum = 0.0
    power_sum = 0.0

    for _ in tqdm(range(args.num_episodes)):
        obs, _ = env.reset()
        ep_reward = 0
        ep_len = 0

        for _ in range(args.ep_length):
            total_steps += 1
            ep_len += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action, populate_info=True)

            ep_reward += reward
            total_reward += reward
            avg_vel += info['vel']
            avg_omega += info['omega']
            avg_g_proj += info['g_proj']

            torque = info.get('torque', None)
            if torque is not None:
                torque_norm_sum += float(np.linalg.norm(torque))
            power_sum += float(info.get('power', 0.0))

            if args.render_mode == 'human' or args.render_mode == 'rgb_array':
                time.sleep(0.25 / 15)

            if terminated:
                n_falls += 1
                break

        total_length += ep_len

    if total_steps > 0:
        avg_vel /= total_steps
        avg_omega /= total_steps
        avg_g_proj /= total_steps
        avg_torque_norm = torque_norm_sum / total_steps
        avg_power = power_sum / total_steps
    else:
        avg_torque_norm = 0.0
        avg_power = 0.0

    print("\n--- Run Summary ---")
    print(f"Avg episode reward: {total_reward / args.num_episodes:.3f}")
    print(f"Avg episode length: {total_length / args.num_episodes:.2f}")
    print(f"Falls: {n_falls} / {args.num_episodes}")
    print(f"Avg velocity: {avg_vel}")
    print(f"Avg angular velocity: {avg_omega}")
    print(f"Avg gravity projection: {avg_g_proj}")
    print(f"Avg torque norm (per step): {avg_torque_norm:.6f}")
    print(f"Avg power (per step): {avg_power:.6f}")

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--history_length', type=int, required=True)

    parser.add_argument("--no_populate_info", action="store_false", help="Disable info population (enabled by default)")
    parser.add_argument("--kick_robot", action="store_true", help="If set, env will occasionally kick the robot")
    parser.add_argument("--ep_length", type=int, required=False, default=3000)
    parser.add_argument("--noise_type", type=str, required=False, default='None', help='None, HPF or LPF')
    parser.add_argument("--alpha", type=float, required=False, default=0.5)
    parser.add_argument('--torque_scale', type=float, required=False, default=1.0)
    parser.add_argument('--record', action='store_true', help='Record video to --output folder')
    parser.add_argument('--num_episodes', type=int, required=False, default=1)
    parser.add_argument('--output', type=str, required=False, default='videos')
    parser.add_argument('--render_mode', type=str, required=False, default='human')
    args = parser.parse_args()

    test(args)
