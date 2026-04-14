from mujoco_playground.config import locomotion_params
from datetime import datetime
import matplotlib.pyplot as plt
from mujoco_playground import registry, wrapper
from IPython.display import clear_output, display
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo


env_name = 'Go1JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

ppo_params = locomotion_params.brax_ppo_config(env_name)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

plt.ion() 
fig, ax = plt.subplots()
def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  ax.clear() # Clear the previous plot
  ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
  ax.set_xlabel("# environment steps")
  ax.set_ylabel("reward per episode")
  ax.set_title(f"Reward: {y_data[-1]:.3f}")
  ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  plt.draw()
  plt.pause(0.1)

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")