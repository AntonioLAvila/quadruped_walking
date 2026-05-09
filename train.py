import functools
from datetime import datetime
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from env import Go2Env

def main():
    env = Go2Env()
    eval_env = Go2Env()

    # Load baseline PPO hyperparameters 
    # Go1 as a strong baseline since they are similar quadrupeds
    env_name_for_config = 'Go1JoystickFlatTerrain'
    ppo_params = locomotion_params.brax_ppo_config(env_name_for_config)

    # 3. Setup plotting for training progress
    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    plt.ion() 
    fig, ax = plt.subplots()

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        ax.clear()
        ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
        ax.set_xlabel("# environment steps")
        ax.set_ylabel("reward per episode")
        ax.set_title(f"Reward: {y_data[-1]:.3f}")
        ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        plt.draw()
        plt.pause(0.1)

    # Configure the PPO network factory
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    # Create the training function
    train_fn = functools.partial(
        ppo.train, 
        **ppo_training_params,
        network_factory=network_factory,
        randomization_fn=None, 
        progress_fn=progress
    )

    print("Starting JIT compilation and training. This may take a few minutes...")

    # Run training and wrap environments for Brax
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to Train: {times[-1] - times[1]}")

    import flax
    with open('go2_params.pkl', 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print("Model saved to go2_params.pkl")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()