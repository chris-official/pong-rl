import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from collections import deque


class PongDataset(Dataset):
    def __init__(self, dataset_length=10_000, framestack=4, channel=0, epsilon=0.01):
        self.env = gym.make("PongNoFrameskip-v4")
        self.dataset_length = dataset_length
        self.framestack = framestack
        self.channel = channel
        self.epsilon = epsilon

        self.action = 0
        self.sample = deque(maxlen=framestack)
        self.reset()

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # chance to execute random action
        if np.random.rand() < self.epsilon:
            # sample random action
            self.action = np.random.choice([0, 2, 3])

        # step environment
        obs, reward, terminated, truncated, info = self.env.step(self.action)

        # crop image
        obs = self.crop_image(obs, self.channel)

        # get ball and paddle position
        ball_x, ball_y = self.get_position(obs, 236)
        paddle_x, paddle_y = self.get_position(obs, 92)

        # determine action based on ball and paddle position
        if ball_y is None or paddle_y is None:
            self.action = 0  # NOOP
        elif ball_y > paddle_y:
            self.action = 3  # DOWN / LEFT
        elif ball_y < paddle_y:
            self.action = 2  # UP / RIGHT
        else:
            self.action = 0  # NOOP

        # append image to stack of frames
        self.sample.append(obs)

        # convert stack of frames to torch tensor
        sample = torch.tensor(np.stack(self.sample), dtype=torch.float32)  # (C, H, W)
        # convert action to torch tensor
        label = torch.tensor(self.action, dtype=torch.long)

        # reset environment if terminated or truncated
        if terminated or truncated:
            self.reset()

        # return sample and label
        return sample, label

    def reset(self):
        """Reset environment and sample stack of frames."""
        _obs, _info = self.env.reset()
        # warm up environment
        for _ in range(58):
            # step environment
            obs, _reward, _terminated, _truncated, _info = self.env.step(0)
            # crop image
            obs = self.crop_image(obs, self.channel)
            # append image to stack of frames
            self.sample.append(obs)

    @staticmethod
    def crop_image(obs, channel=0):
        """Crop image to play field."""
        return obs[34:194:2, ::2, channel]

    @staticmethod
    def get_position(obs, color):
        """Get position of a color in the image."""
        mask = np.where(obs == color)
        y, x = mask[0], mask[1]
        if len(y) == 0:
            return None, None
        pos_y = (y.max() + y.min()) / 2
        pos_x = (x.max() + x.min()) / 2
        return pos_x, pos_y

    def plot(self):
        """Plot a sample and label."""
        sample, label = self[0]
        plt.imshow(sample[-1, :, :], cmap="gray")
        plt.grid(False)
        if label is not None:
            plt.title(f"Action {label}")
        plt.show()
