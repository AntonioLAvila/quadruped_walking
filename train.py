from env import make_simulation_maker, reward_fn, make_gym_env
from stable_baselines3 import PPO

if __name__ == '__main__':
    env = make_gym_env(reward_fn, make_simulation_maker)

    # env.reset()
    # env.render()

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=100000, progress_bar=True)

    model.save('ppo_A1_final')


    # obs, _ = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         print("Episode finished. Resetting.")
    #         obs, _ = env.reset()