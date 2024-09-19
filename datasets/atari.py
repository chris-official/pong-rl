import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import PPO
from typing import Literal


class AtariPongDataset(Dataset):
    def __init__(
            self,
            dataset_length: int = 10_000,
            epsilon: float = 0.01,
            seed: int = None,
            mode: Literal["class", "proba"] = "proba",
            model_path: str = "models/ppo/ppo_nature_cnn_v1_best",
    ):
        self.env = make_atari_env(env_id="PongNoFrameskip-v4", n_envs=1, seed=seed)
        self.dataset_length = dataset_length
        self.epsilon = epsilon
        self.seed = seed
        self.mode = mode

        if self.seed is not None:
            np.random.seed(self.seed)

        self.action = 0
        _obs = self.env.reset()

        algo = PPO.load(model_path, device="cuda")
        self.model = torch.nn.Sequential(algo.policy.pi_features_extractor, algo.policy.action_net)
        self.model.eval()

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # chance to execute random action
        if np.random.rand() < self.epsilon:
            # sample random action
            self.action = np.random.choice([0, 2, 3])

        # step environment
        obs, reward, done, info = self.env.step(np.array([self.action]))

        # convert stack of frames to torch tensor
        sample = torch.tensor(obs, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)  # (B, C, H, W)

        # normalize image
        sample = sample / 255.0

        # get logits from model
        with torch.no_grad():
            logits = self.model(sample).cpu()  # (B, 6)

        # convert logits to label
        label = torch.argmax(logits, dim=1).squeeze()  # (,)
        self.action = label.item()

        if self.mode == "proba":
            label = torch.nn.functional.softmax(logits, dim=1).squeeze()  # (6,)

        # reset environment if terminated or truncated
        if done:
            _obs = self.env.reset()

        # return sample and label
        return sample.squeeze(0).cpu(), label

    def plot(self) -> None:
        """Plot a sample and label."""
        sample, label = self[0]
        plt.imshow(sample[-1, :, :], cmap="gray")
        plt.grid(False)
        if label is not None:
            plt.title(f"Action {label}")
        plt.show()

    def plot_distribution(self) -> None:
        """Plot the action distribution."""
        assert self.mode == "proba", "Mode must be 'proba' to plot distribution"
        sample, label = self[0]
        label = label.squeeze().cpu().detach().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sample[-1, :, :], cmap="gray")
        ax[0].set_title(f"Target: {label.round(2)}")
        ax[1].bar(range(6), label)
        ax[1].set_title(f"Action Probabilities, Prediction: {label.argmax()}")
        ax[1].set_xticks(range(6))
        ax[1].set_xticklabels(['NOOP', 'FIRE', 'RIGHT / UP', 'LEFT / DOWN', 'RIGHTFIRE', 'LEFTFIRE'], rotation=45, ha="right")
        ax[1].set_ylim(0, 1)
        ax[0].grid(False)
        ax[1].grid(False)
        plt.show()
