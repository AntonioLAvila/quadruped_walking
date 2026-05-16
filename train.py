import jax
import jax._src.api as _jax_api
if not hasattr(jax, 'device_put_replicated'):
    jax.device_put_replicated = _jax_api.device_put_replicated

import functools
from datetime import datetime
import matplotlib.pyplot as plt
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from env import Go2Env
from brax.io import model
from configs import PPO_CONFIG, NETWORK_FACTORY_CONFIG

def main():
    env = Go2Env()
    eval_env = Go2Env()

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
        ax.set_xlim([0, PPO_CONFIG["num_timesteps"] * 1.25])
        ax.set_xlabel("# environment steps")
        ax.set_ylabel("reward per episode")
        ax.set_title(f"Reward: {y_data[-1]:.3f}")
        ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        print(f"Reward: {metrics["eval/episode_reward"]}")

        plt.draw()
        plt.pause(0.1)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **NETWORK_FACTORY_CONFIG,
    )

    train_fn = functools.partial(
        ppo.train,
        **PPO_CONFIG,
        network_factory=network_factory,
        randomization_fn=None,
        progress_fn=progress,
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

    model_path = 'go2_params'
    model.save_params(model_path, params)
    print(f"Model saved to {model_path}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()