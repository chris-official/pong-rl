import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import PPO
from typing import Literal
from numba import njit


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


class AtariAlgoPongDataset(AtariPongDataset):
    def __init__(
            self,
            dataset_length: int = 10_000,
            epsilon: float = 0.01,
            seed: int = None,
            mode: Literal["class", "proba"] = "proba",
            model_path: str = "models/ppo/ppo_nature_cnn_v1_best",
            label_type: Literal["algo", "model", "both"] = "algo",
            num_classes: int = 6,
    ):
        super().__init__(dataset_length, epsilon, seed, mode, model_path)
        self.action = np.array([1., 0., 0., 0., 0., 0.], dtype=np.float32)
        self.label_type = label_type
        self.num_classes = num_classes

        assert self.num_classes in [3, 6], "Number of classes must be 3 or 6"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        if self.label_type == "algo":
            return self._getitem_algo(idx)
        elif self.label_type == "model":
            return self._getitem_model(idx)
        elif self.label_type == "both":
            return self._getitem_both(idx)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}. Supported label types are: 'algo', 'model', 'both'")

    def _getitem_algo(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # chance to execute random action
        if np.random.rand() < self.epsilon:
            # sample random action
            self.action = np.random.choice([0, 2, 3])
        else:
            self.action = np.argmax(self.action)
            if self.num_classes == 3:
                if self.action != 0:
                    self.action += 1

        # step environment
        obs, reward, done, info = self.env.step(np.array([self.action]))

        _ball_x, ball_y = self.get_ball_position(obs, color=87)
        _paddle_x, paddle_y = self.get_paddle_position(obs, color=147)
        self.action = self.get_probabilities(ball_y, paddle_y, 63, 0.2, self.num_classes)

        # convert stack of frames to torch tensor
        sample = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)  # (B, C, H, W)

        # normalize image
        sample = sample / 255.0

        # reset environment if terminated or truncated
        if done:
            _obs = self.env.reset()

        # return sample and label
        return sample.squeeze(0), torch.tensor(self.action, dtype=torch.float32)

    def _getitem_model(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _getitem_both(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        # chance to execute random action
        if np.random.rand() < self.epsilon:
            # sample random action
            self.action = np.random.choice([0, 2, 3])
        else:
            self.action = np.argmax(self.action)

        # step environment
        obs, reward, done, info = self.env.step(np.array([self.action]))

        _ball_x, ball_y = self.get_ball_position(obs, color=87)
        _paddle_x, paddle_y = self.get_paddle_position(obs, color=147)
        self.action = self.get_probabilities(ball_y, paddle_y, 63, 0.2, self.num_classes)

        # convert stack of frames to torch tensor
        sample = torch.tensor(obs, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)  # (B, C, H, W)

        # normalize image
        sample = sample / 255.0

        # get logits from model
        with torch.no_grad():
            logits = self.model(sample).cpu()  # (B, 6)

        # convert logits to label
        label = torch.argmax(logits, dim=1).squeeze()  # (,)

        if self.mode == "proba":
            label = torch.nn.functional.softmax(logits, dim=1).squeeze()  # (6,)

        # reset environment if terminated or truncated
        if done:
            _obs = self.env.reset()

        # return sample and label
        return sample.squeeze(0).cpu(), label, self.action

    @staticmethod
    @njit
    def get_probabilities(ball_y: float, paddle_y: float, obs_dim: int, std: float = 0.3, num_classes: int = 6) -> np.ndarray:
        """Calculate probabilities for moving up and down."""

        # Return NOOP if ball or paddle position is None
        if ball_y is None or paddle_y is None:
            if num_classes == 3:
                return np.array([1., 0., 0.], dtype=np.float32)
            return np.array([1., 0., 0., 0., 0., 0.], dtype=np.float32)

        # Calculate the vertical distance between the ball and the paddle
        dist = ball_y - paddle_y

        # Normalize x to the range [0, 1]
        dist_normalized = (dist + obs_dim) / (obs_dim * 2)

        # Calculate probabilities using a smooth transition
        prob_up = np.exp(-((dist_normalized - 0) ** 2) / (2 * (std ** 2)))
        prob_down = np.exp(-((dist_normalized - 1) ** 2) / (2 * (std ** 2)))

        # Normalize the probabilities so that they sum to 1
        prob_sum = prob_up + prob_down
        prob_up /= prob_sum
        prob_down /= prob_sum

        if num_classes == 3:
            return np.array([0., prob_up, prob_down], dtype=np.float32)
        return np.array([0., 0., prob_up, prob_down, 0., 0.], dtype=np.float32)

    @staticmethod
    @njit
    def get_ball_position(obs: np.ndarray, color: int) -> tuple[int, int] | tuple[None, None]:
        """Get position of a color in the image."""
        mask = np.where(obs[0, 14:77, 11:73, 0] != color)
        y, x = mask[0], mask[1]
        if len(y) == 0:
            return None, None
        y, x = y + 14, x + 11
        pos_y = (y.max() + y.min()) / 2
        pos_x = (x.max() + x.min()) / 2
        return pos_x, pos_y

    @staticmethod
    @njit
    def get_paddle_position(obs: np.ndarray, color: int) -> tuple[int, int] | tuple[None, None]:
        """Get position of a color in the image."""
        mask = np.where(obs[0, 14:77, 73:76, 0] == color)
        y, x = mask[0], mask[1]
        if len(y) == 0:
            return None, None
        y, x = y + 14, x + 73
        pos_y = (y.max() + y.min()) / 2
        pos_x = (x.max() + x.min()) / 2
        return pos_x, pos_y

    def plot_distribution(self) -> None:
        """Plot the action distribution."""
        assert self.mode == "proba", "Mode must be 'proba' to plot distribution"
        out = self[0]

        if self.num_classes == 6:
            if len(out) == 2:
                sample, label = out
                algo_label = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
            else:
                sample, label, algo_label = out
            label = label.squeeze().cpu().detach().numpy()

            model_idx = label.argmax()
            algo_idx = algo_label.argmax()
            actions = ['NOOP', 'NOOP', 'UP', 'DOWN', 'UP', 'DOWN']

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(sample[-1, :, :], cmap="gray")
            ax[0].set_title("Playing Field")
            ax[1].bar(range(6), label)
            ax[1].bar(range(6), algo_label, alpha=0.5)
            ax[1].set_title(f"Model: {actions[model_idx]} | Algo: {actions[algo_idx]}")
            ax[1].set_xticks(range(6))
            ax[1].set_xticklabels(actions, rotation=45, ha="right")
            # ['NOOP', 'FIRE / NOOP', 'RIGHT / UP', 'LEFT / DOWN', 'RIGHTFIRE / UP', 'LEFTFIRE / DOWN']
            ax[1].set_ylim(0, 1)
            ax[0].grid(False)
            ax[1].grid(False)
            plt.show()
        else:
            if len(out) == 2:
                sample, label = out
                algo_label = np.array([0., 0., 0.], dtype=np.float32)
            else:
                sample, label, algo_label = out
            label = label.squeeze().cpu().detach().numpy()

            if len(label) == 3:
                label = np.array([label[0], 0., label[1], label[2], 0., 0.], dtype=np.float32)
            if len(algo_label) == 3:
                algo_label = np.array([algo_label[0], 0., algo_label[1], algo_label[2], 0., 0.], dtype=np.float32)

            model_idx = label.argmax()
            algo_idx = algo_label.argmax()

            actions = ['NOOP', 'NOOP', 'UP', 'DOWN', 'UP', 'DOWN']

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(sample[-1, :, :], cmap="gray")
            ax[0].set_title("Playing Field")
            ax[1].bar(range(6), label)
            ax[1].bar(range(6), algo_label, alpha=0.5)
            ax[1].set_title(f"Model: {actions[model_idx]} | Algo: {actions[algo_idx]}")
            ax[1].set_xticks(range(6))
            ax[1].set_xticklabels(actions, rotation=45, ha="right")
            # ['NOOP', 'FIRE / NOOP', 'RIGHT / UP', 'LEFT / DOWN', 'RIGHTFIRE / UP', 'LEFTFIRE / DOWN']
            ax[1].set_ylim(0, 1)
            ax[0].grid(False)
            ax[1].grid(False)
            plt.show()
