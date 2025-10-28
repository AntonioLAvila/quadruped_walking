from env import A1_Env, BasicExtractor
from stable_baselines3 import PPO
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_steps', type=int, required=False, default=2000000)
    parser.add_argument('--model_path', type=str, required=False, default='buh')
    parser.add_argument('--visualize', type=bool, required=False, default=False)
    args = parser.parse_args()
    args = vars(args)

    env = A1_Env(BasicExtractor, visualize=args['visualize'])
       
    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=args['num_steps'], progress_bar=True)

    model.save(args['model_path'])