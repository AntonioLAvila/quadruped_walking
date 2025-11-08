from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import Go1_Env
from argparse import ArgumentParser
from clearml import Task, Logger


'''
Kills performance
'''

def make_env(**kwargs):
    return Go1_Env(torque_scale=3, **kwargs)


# ===== Custom ClearML Callback =====
class ClearMLCallback(BaseCallback):
    def __init__(self, task, verbose=0):
        super().__init__(verbose)
        self.task = task
        self.clearml_logger = task.get_logger()

    def _on_step(self) -> bool:
        # Log rollout metrics if available
        if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
            self.clearml_logger.report_scalar(
                title="Training",
                series="Episode Reward Mean",
                iteration=self.num_timesteps,
                value=self.model.logger.name_to_value["rollout/ep_rew_mean"],
            )

        if "train/entropy_loss" in self.model.logger.name_to_value:
            self.clearml_logger.report_scalar(
                title="Training",
                series="Entropy Loss",
                iteration=self.num_timesteps,
                value=self.model.logger.name_to_value["train/entropy_loss"],
            )

        if "train/value_loss" in self.model.logger.name_to_value:
            self.clearml_logger.report_scalar(
                title="Training",
                series="Value Loss",
                iteration=self.num_timesteps,
                value=self.model.logger.name_to_value["train/value_loss"],
            )

        if "train/approx_kl" in self.model.logger.name_to_value:
            self.clearml_logger.report_scalar(
                title="Training",
                series="Approx KL",
                iteration=self.num_timesteps,
                value=self.model.logger.name_to_value["train/approx_kl"],
            )

        if "train/clip_fraction" in self.model.logger.name_to_value:
            self.clearml_logger.report_scalar(
                title="Training",
                series="Clip Fraction",
                iteration=self.num_timesteps,
                value=self.model.logger.name_to_value["train/clip_fraction"],
            )

        return True


def train(args):
    # --- ClearML task setup ---
    task = Task.init(
        project_name="Quadruped-Walking",
        task_name=f"PPO-Go1-{args.num_envs}envs",
        tags=["ppo", "quadruped", "torque"],
        reuse_last_task_id=True
    )
    task.connect(vars(args))  # logs your hyperparameters

    # --- Environment setup ---
    vec_env = make_vec_env(
        make_env,
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv
    )

    print(f"Training on {args.num_envs} envs\nSaving models to '{args.model_dir}'")

    # --- Callbacks ---
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq // args.num_envs,
        save_path=args.model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    clearml_callback = ClearMLCallback(task)

    callback = CallbackList([eval_callback, checkpoint_callback, clearml_callback])

    # --- PPO model ---
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device="cpu"
    )

    # --- Train ---
    model.learn(
        total_timesteps=args.num_steps,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=callback,
    )

    model.save(f"{args.model_dir}/final_model")
    task.close()  # finalize ClearML task


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--eval_freq", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    train(args)
