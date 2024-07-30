##
from typing import Type
import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, activation: Type[nn.Module]):
        super().__init__()
        self.block = nn.Sequential(
            activation(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            activation(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, depth: int, activation: Type[nn.Module]):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, depth, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(depth, activation),
            ResBlock(depth, activation),
        )

    def forward(self, x):
        return self.sequence(x)


class ImpalaCNN(BaseFeaturesExtractor):
    """
    CNN from IMPALA paper:
    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
    """

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
            depths: list[int] = None,
            activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            f"This model must be used with a gym.spaces.Box observation space, not {observation_space}"
        )
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            f"You should use this model only with images not with {observation_space}"
        )
        assert len(depths) > 0, "depths must be a non-empty list"

        n_input_channels = observation_space.shape[0]
        image_size = observation_space.shape[1:]
        if depths is None:
            depths = [16, 32, 32]

        layers = []
        # Parameter free image size reduction to 40x40
        if image_size == (160, 160):
            layers.append(nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)))
            layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))

        for i, depth in enumerate(depths):
            if i == 0:
                layers.append(ConvSequence(n_input_channels, depth, activation))
            else:
                layers.append(ConvSequence(depths[i - 1], depth, activation))

        self.extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.activation = activation()

        # Compute shape by doing one forward pass
        with th.no_grad():
            in_features = self.extractor(th.as_tensor(observation_space.sample()[None]).float()).view(1, -1).shape[1]

        self.linear = nn.Linear(in_features, features_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self.extractor(obs)
        x = self.flatten(x)
        x = self.activation(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
