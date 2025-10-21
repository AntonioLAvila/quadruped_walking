from env import make_simulation_maker, reward_fn, make_gym_env
from stable_baselines3 import PPO
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_steps', type=int, required=True)
    args = parser.parse_args()
    args = vars(args)

    env = make_gym_env(reward_fn, make_simulation_maker)
   
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=args['num_steps'], progress_bar=True)

    model.save('ppo_A1_final')