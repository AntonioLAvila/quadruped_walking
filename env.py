import os
import time
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from robot_descriptions import a1_mj_description

# Stable Baselines3 imports (kept for context, not used for visualization)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Global constant for action scaling
torque_scale = 33.5

# --- 1. Custom Unitree A1 MuJoCo Environment Definition ---
# This class sets up the Unitree A1 robot within the Gymnasium/MuJoCo framework.
class UnitreeA1Env(MujocoEnv):
    """
    Custom Environment for the Unitree A1 Quadruped robot.
    Inherits from MujocoEnv to handle the physics simulation boilerplate.
    """
    # Define the Unitree A1 XML path (must be manually acquired and placed here)
    metadata = {
        "render_modes": [
            "human", 
            "rgb_array", 
            "depth_array"
        ], 
        "render_fps": 60
    }
    
    def __init__(self, render_mode=None, **kwargs):
        
        # --- FIX: Safely extract 'max_episode_steps' from kwargs ---
        # This prevents it from being passed to the parent MujocoEnv constructor,
        # where it is not a valid argument. We store it for later use.
        custom_max_steps = kwargs.pop('max_episode_steps', None) 
        # -----------------------------------------------------------

        # Call the parent constructor with the remaining kwargs
        MujocoEnv.__init__(
            self, 
            model_path=a1_mj_description.MJCF_PATH,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64),
            render_mode=render_mode,
            default_camera_config={
                'distance': 2.0,
                'elevation': -20,
                'azimuth': 135
            },
            frame_skip=1,
            **kwargs
        )
        
        # Action space: 12 motors, typically normalized to [-1, 1] for torque control
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float64)

        # Base mass of the robot's torso (used for calculating forward velocity)
        self.torso_body_index = self.model.body('trunk').id
        
        # Manually track step count since we are not using the standard gymnasium wrapper
        self.step_count = 0
        if self.spec is None:
            # Use the value passed in the constructor, or default to 1000 if not provided
            self.max_episode_steps = custom_max_steps if custom_max_steps is not None else 1000 
        else:
             # If registered via gym.make, use the spec's value
             self.max_episode_steps = self.spec.max_episode_steps

    def step(self, action):
        """
        Executes one step in the environment.
        """        
        # Get the previous x-position for calculating velocity
        x_position_before = self.data.qpos[0]

        # Scale action from [-1, 1] to actual torque range
        self.do_simulation(action * torque_scale, self.frame_skip)
        
        # Get the new x-position
        x_position_after = self.data.qpos[0]

        # --- Reward Calculation ---
        forward_velocity = (x_position_after - x_position_before) / self.dt
        forward_reward = forward_velocity * 1.2
        
        control_cost = np.dot(action, action) * 0.001
        
        z_height = self.data.qpos[2] # qpos[2] is the z-coordinate of the base
        z_cost = np.square(z_height - 0.3) * 1.0
        
        survival_reward = 1.0
        
        reward = forward_reward - control_cost - z_cost + survival_reward

        # --- Termination Condition ---
        min_z_height = 0.2  # Robot is considered fallen if below this height
        terminated = z_height < min_z_height
        
        # Check for truncation (max episode steps)
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        # --- Observation and Info ---
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -control_cost,
            "x_position": x_position_after,
            "z_height": z_height,
            "x_velocity": forward_velocity
        }

        return observation, reward, terminated, truncated, info

    def _reward(self, action, observation):
        # The reward logic is already integrated into step(), so this is not strictly needed
        pass

    def reset_model(self, seed=None):
        """
        Resets the simulation to a starting state.
        """
        # Call the parent reset_model for boilerplate setup
        super().reset_model()
        self.step_count = 0 # Reset step counter

        # Add noise to initial joint positions for robust training
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.05, high=0.05, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.05, high=0.05, size=self.model.nv
        )
        # Set the initial base height for the robot (qpos[2])
        qpos[2] = 0.45 

        self.set_state(qpos, qvel)

        return self._get_obs(), {} # Return observation and info dict (as per gymnasium standard)

    def _get_obs(self):
        """
        Returns the observation vector. 
        """
        # The parent's _get_obs combines qpos (except for 3D base position), qvel, and external contact forces (cfrc_ext).
        return np.concatenate(
            [
                self.data.qpos.flat[3:],  # Skip base x, y, z positions (3 values)
                self.data.qvel.flat,
                self.data.cfrc_ext.flat,
            ]
        )


# --- 2. Visualization/Demo Script ---
if __name__ == "__main__":
    
    # 1. Create the environment with the 'human' render mode
    # The 'max_episode_steps' argument now works because the constructor handles it.
    print("Initializing Unitree A1 environment for visualization...")
    env = UnitreeA1Env(
        render_mode='human',
        max_episode_steps=500  # Run for up to 500 simulation steps
    )
    
    # 2. Start a loop to interact with the environment
    
    episodes = 2
    for episode in range(episodes):
        print(f"\n--- Starting Episode {episode + 1}/{episodes} ---")
        
        # Reset the environment and get the initial observation
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not terminated and not truncated:
            step += 1
            
            # Agent takes a random action for demonstration
            action = env.action_space.sample()
            
            # Execute the action
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Render the environment (updates the visualization window)
            if env.render_mode == 'human':
                env.render()
                # Use a small sleep to slow down the visualization for better viewing
                time.sleep(0.01) 

            # Log information (optional)
            if step % 50 == 0:
                 print(f"Step: {step}, Height (Z): {info['z_height']:.3f}, Vel (X): {info['x_velocity']:.3f}, Reward: {reward:.2f}")

        print(f"Episode finished after {step} steps. Total reward: {total_reward:.2f}")
    
    # 3. Clean up and close the environment
    env.close()
    print("Visualization complete. Environment closed.")
