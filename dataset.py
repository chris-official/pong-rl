import torch
import numpy as np
from numba import njit
from torch.utils.data import Dataset
from stable_baselines3.common.env_util import make_atari_env


class StaticImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class PongDataset(Dataset):
    def __init__(
            self,
            dataset_length: int = 10_000,
            epsilon: float = 0.01,
            sigma: float = 0.2,
            seed: int = None,
    ):
        self.env = make_atari_env(env_id="PongNoFrameskip-v4", n_envs=1, seed=seed)
        self.dataset_length = dataset_length
        self.epsilon = epsilon
        self.sigma = sigma
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        _obs = self.env.reset()
        self.action = np.array([1., 0., 0., 0., 0., 0.], dtype=np.float32)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # chance to execute random action
        if np.random.rand() < self.epsilon:
            # sample random action
            self.action = np.random.choice([0, 2, 3])
        else:
            self.action = np.argmax(self.action)

        # step environment
        obs, reward, done, info = self.env.step(np.array([self.action]))

        # get ball and paddle position
        _ball_x, ball_y = self.get_position(obs, color=87, bound_left=11, bound_right=73)
        _paddle_x, paddle_y = self.get_position(obs, color=147, bound_left=73, bound_right=76)

        # calculate and smooth probabilities for moving up and down
        self.action = self.get_probabilities(ball_y, paddle_y, 63, self.sigma)

        # convert stack of frames to torch tensor
        sample = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2)  # (B, C, H, W)

        # normalize image
        sample = sample / 255.0

        # reset environment if terminated or truncated
        if done:
            _obs = self.env.reset()

        # return sample and label
        return sample.squeeze(0), torch.tensor(self.action, dtype=torch.float32)

    @staticmethod
    @njit
    def get_probabilities(ball_y: float, paddle_y: float, obs_dim: int, std: float = 0.2) -> np.ndarray:
        """Calculate probabilities for moving up and down."""

        # Return NOOP if ball or paddle position is None
        if ball_y is None or paddle_y is None:
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

        return np.array([0., 0., prob_up, prob_down, 0., 0.], dtype=np.float32)

    @staticmethod
    @njit
    def get_position(obs: np.ndarray, color: int, bound_left: int, bound_right: int) -> tuple[int, int] | tuple[None, None]:
        """Get position of a color in the image."""

        # Find the position of the color in the image
        mask = np.where(obs[0, 14:77, bound_left:bound_right, 0] != color)
        y, x = mask[0], mask[1]
        if len(y) == 0:
            return None, None

        # correct coordinates to account for cropping
        y, x = y + 14, x + bound_left

        # calculate the center of the object
        pos_y = (y.max() + y.min()) / 2
        pos_x = (x.max() + x.min()) / 2

        # return the coordinates
        return pos_x, pos_y
