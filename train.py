import os

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env import UnitreeA1Env


# --- Setup and Hyperparameters ---
TIMESTEPS = int(2e6)  # Total training timesteps
MODEL_NAME = "ppo_unitree_a1"
LOG_DIR = "./a1_training_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 1. Create a vectorized environment
# Use 4 parallel environments for faster data collection
# Note: 'UnitreeA1Env-v0' is a convention, the custom class is directly used here
try:
    env = make_vec_env(UnitreeA1Env, n_envs=4, env_kwargs={'max_episode_steps': 1000})
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR: Environment initialization failed.")
    print("Ensure 'unitree_a1.xml' is in the current directory and MuJoCo dependencies are installed.")
    print(f"Details: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()


# 2. Instantiate the PPO agent
# PPO (Proximal Policy Optimization) is highly effective for continuous control tasks like locomotion.
model = PPO(
    "MlpPolicy",  # Multi-layer Perceptron Policy for vector observations
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048, # Number of steps to run for each environment per update
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]) # Simple network architecture
    ),
    tensorboard_log=LOG_DIR
)

# 3. Setup Evaluation Callback (Optional but highly recommended)
# This evaluates the agent periodically on a separate environment and saves the best model.
eval_env = UnitreeA1Env(render_mode='rgb_array', max_episode_steps=1000)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{LOG_DIR}/best_model/",
    log_path=LOG_DIR,
    eval_freq=50000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# 4. Start Training
print(f"\n--- Starting PPO Training for {TIMESTEPS} timesteps ---")
model.learn(
    total_timesteps=TIMESTEPS, 
    callback=eval_callback,
    progress_bar=True
)

# 5. Save the final model
final_model_path = f"{LOG_DIR}/{MODEL_NAME}_final.zip"
model.save(final_model_path)
print(f"\nTraining finished. Model saved to: {final_model_path}")

# --- Optional: Run the trained model for demonstration (uncomment to test) ---
# print("\n--- Testing the trained agent (Close viewer to end) ---")
# try:
#     del model # Clear memory
#     model = PPO.load(final_model_path, env=eval_env)
#     obs, info = eval_env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = eval_env.step(action)
#         if terminated or truncated:
#             obs, info = eval_env.reset()
#         eval_env.render()
# except Exception as e:
#     print(f"Could not run demonstration: {e}")
# finally:
#     eval_env.close()