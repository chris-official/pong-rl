import os
import optuna
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param name: Name of the output model file.
    :param mean_window: The number of steps to consider when computing the average reward.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, name: str = "best_model", mean_window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, name)
        self.mean_window = mean_window
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.mean_window:])
                if self.verbose > 0:
                    print(f"Step: {self.num_timesteps:,} | Best mean reward: {self.best_mean_reward:.2f} | Current mean reward: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class PruningCallback(BaseCallback):
    def __init__(self, trial: optuna.trial.Trial, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.trial = trial
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_rwd = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(ep_rwd) > 0:
                ep_rew_mean = np.mean(ep_rwd)
            else:
                ep_rew_mean = -21.0
            self.trial.report(ep_rew_mean, step=self.num_timesteps)
            if self.trial.should_prune():
                if self.verbose > 0:
                    print(f"Trial {self.trial.number} was pruned at step: {self.num_timesteps}")
                raise optuna.TrialPruned()

        return True