import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import mediapy as media
import functools
import flax

# Brax / Mujoco Playground imports
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training import networks as networks_lib 
from mujoco_playground.config import locomotion_params
from mujoco_playground import wrapper

# Import your custom environment
from env import Go2Env

def main():
    print("Initializing environment...")
    base_env = Go2Env()
    # Ensure the wrapper matches training (history, action repeat, etc.)
    env = wrapper.wrap_for_brax_training(base_env, episode_length=1000, action_repeat=1)

    # Load config
    env_name_for_config = 'Go1JoystickFlatTerrain'
    ppo_params = locomotion_params.brax_ppo_config(env_name_for_config)
    
    # --- Step 1: Handle Structured Observations ---
    obs_structure = env.observation_size
    if isinstance(obs_structure, dict):
        raw_size = obs_structure['state']
        policy_obs_size = raw_size[0] if isinstance(raw_size, (tuple, list)) else raw_size
    else:
        policy_obs_size = obs_structure[0] if isinstance(obs_structure, (tuple, list)) else obs_structure

    print(f"Policy observation size: {policy_obs_size}")

    # Create the PPO networks
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )
    
    networks = network_factory(
        int(policy_obs_size), 
        env.action_size
    )

    print("Loading saved model weights...")
    try:
        with open('go2_params.pkl', 'rb') as f:
            rng = jax.random.PRNGKey(0)
            dummy_obs_policy = jp.zeros((1, int(policy_obs_size)))
            
            # Fix for "init() takes 1 positional argument but 2 were given"
            # We try passing obs as a keyword, then fallback to positional
            try:
                dummy_policy_params = networks.policy_network.init(rng, obs=dummy_obs_policy)
            except TypeError:
                dummy_policy_params = networks.policy_network.init(rng, dummy_obs_policy)
            
            # Normalizer params tree mapping
            def create_stats(size):
                shape = size if isinstance(size, (tuple, list)) else (size,)
                return networks_lib.RunningStatisticsState(
                    mean=jp.zeros(shape), 
                    var=jp.ones(shape), 
                    count=jp.array(0, dtype=jp.int32)
                )
            
            dummy_normalizer_params = jax.tree_util.tree_map(create_stats, obs_structure)
            
            # Unpack bytes into the correct structure
            dummy_structure = (dummy_policy_params, dummy_normalizer_params)
            params = flax.serialization.from_bytes(dummy_structure, f.read())
            print("Weights successfully loaded!")
            
    except Exception as e:
        print(f"Error unpacking weights: {e}")
        return

    # --- Step 2: Prepare Inference ---
    make_inference_fn = ppo.make_inference_fn(networks)
    inference_fn = make_inference_fn(params, deterministic=True)
    
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    print("Starting simulation loop (JIT compilation may take a minute)...")
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    
    # Initial state (batch size 1)
    state = jit_reset(jax.random.split(reset_rng, 1))

    # --- Step 3: Inject Commands ---
    new_info = state.info.copy()
    new_info['command'] = jp.array([[1.0, 0.0, 0.0]]) # Walk forward
    new_info['steps_until_cmd'] = jp.array([999999])
    state = state.replace(info=new_info)

    # Setup Renderer
    renderer = mujoco.Renderer(base_env.mj_model, height=480, width=640)
    mj_data = mujoco.MjData(base_env.mj_model)
    
    frames = []
    fps = 30
    sim_steps_per_frame = int((1.0 / fps) / base_env.dt)
    total_steps = int(5.0 / base_env.dt) # 5 seconds simulation

    for step in range(total_steps):
        rng, act_rng = jax.random.split(rng)
        
        # Policy inference
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, act)

        if step % sim_steps_per_frame == 0:
            # Sync MJX to MuJoCo
            mj_data.qpos = np.array(state.pipeline_state.qpos[0])
            mj_data.qvel = np.array(state.pipeline_state.qvel[0])
            
            mujoco.mj_kinematics(base_env.mj_model, mj_data)
            renderer.update_scene(mj_data, camera="track")
            frames.append(renderer.render())

    # --- Step 4: Save Video ---
    output_path = "go2_test_flight.mp4"
    media.write_video(output_path, frames, fps=fps)
    print(f"Success! Video saved as {output_path}")

if __name__ == "__main__":
    main()