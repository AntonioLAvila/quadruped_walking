import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mediapy as media
import functools

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import wrapper

from env import Go2Env
from configs import NETWORK_FACTORY_CONFIG

COMMAND = jp.array([1.5, 0.0, 0.0])  # vx=1.5 m/s, vy=0.0, yaw=0.0 rad/s
TIME = 10 # seconds


def main():
    env = Go2Env()

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **NETWORK_FACTORY_CONFIG,
    )

    # Wrapped env gives us the same obs/action sizes that training used
    wrapped_env = wrapper.wrap_for_brax_training(env)
    obs_size = wrapped_env.observation_size   # {'state': (48,), 'privileged_state': (123,)}
    action_size = wrapped_env.action_size     # 12

    # Build the same PPO network as training.
    # normalize_observations=True in the train config, so running_statistics.normalize is used
    ppo_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    print("Loading model params...")
    params = model.load_params('go2_params')
    print("Params loaded.")

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    state = jit_reset(key)

    # Pin command for the whole episode
    state = state.replace(info={
        **state.info,
        'command': COMMAND,
        'steps_until_cmd': jp.array(1e9, dtype=jp.int32),
    })

    renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    mj_data = mujoco.MjData(env.mj_model)

    frames = []
    fps = 30
    sim_steps_per_frame = max(1, int((1.0 / fps) / env.dt))
    total_steps = int(TIME / env.dt)
    print(f"Simulating {total_steps} steps ({total_steps * env.dt:.1f} s)...")

    for i in range(total_steps):
        rng, act_rng = jax.random.split(rng)

        # Pass the full dict obs the policy network internally selects obs['state']
        # and uses the matching slice of normalizer_params
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, act)

        # Keep command pinned after each step
        state = state.replace(info={
            **state.info,
            'command': COMMAND,
            'steps_until_cmd': jp.array(1e9, dtype=jp.int32),
        })

        if i % sim_steps_per_frame == 0:
            mj_data.qpos[:] = np.array(state.data.qpos)
            mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(env.mj_model, mj_data)
            renderer.update_scene(mj_data, camera='track') # Defined in the mjcf
            frames.append(renderer.render())

    output_path = "go2_test.mp4"
    media.write_video(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
