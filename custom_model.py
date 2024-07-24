from typing import Type
import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class SimpleCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
            activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            f"NatureCNN must be used with a gym.spaces.Box observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            f"You should use a CNN only with images not with {observation_space}"
        )
        n_input_channels = observation_space.shape[0]
        self.activation = activation()
        self.conv1 = nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(128*4*4, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:  # 4 x 42 x 42: frame_stacks x screen_size x screen_size
        x = self.conv1(observations)  # 4 x 42 x 42 -> 16 x 40 x 40
        x = self.pool(x)  # 16 x 40 x 40 -> 16 x 20 x 20
        x = self.activation(x)
        x = self.conv2(x)  # 16 x 20 x 20 -> 32 x 18 x 18
        x = self.activation(x)
        x = self.conv3(x)  # 32 x 18 x 18 -> 64 x 14 x 14
        x = self.activation(x)
        x = self.conv4(x)  # 64 x 14 x 14 -> 128 x 8 x 8
        x = self.pool(x)  # 128 x 8 x 8 -> 128 x 4 x 4
        x = self.activation(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple] = 3,
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = None, groups: int = 1,
                 activation: Type[nn.Module] = nn.Mish):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        return self.conv_block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim, activation=nn.SiLU):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_squeeze = nn.Conv2d(in_channels, reduced_dim, 1)
        self.act = activation()
        self.conv_expand = nn.Conv2d(reduced_dim, in_channels, 1)
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        x = self.pool(inputs)  # reduce each image to a single value: C x H x W -> C x 1 x 1
        x = self.conv_squeeze(x)  # reduce number of channels
        x = self.act(x)
        x = self.conv_expand(x)  # expand number of channels
        x = self.sig(x)  # create channel-wise importance weights
        x = inputs * x  # scale the input according to importance weights
        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple] = 3,
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = None, expand_ratio: int = 2,
                 reduction: int = 2, activation: Type[nn.Module] = nn.Mish):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        reduced_dim = in_channels // reduction
        self.expand = in_channels != hidden_dim

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim, 3, 1, "same", 1, activation)

        self.depthwise = ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, hidden_dim, activation)
        self.squeeze = SqueezeExcitation(hidden_dim, reduced_dim, activation)
        self.pointwise = nn.Conv2d(hidden_dim, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.expand:
            x = self.expand_conv(x)  # expand number of channels to hidden_dim
        x = self.depthwise(x)  # depthwise convolution
        x = self.squeeze(x)  # squeeze and excitation
        x = self.pointwise(x)  # pointwise convolution
        x = self.norm(x)  # batch normalization
        return x


class EfficientNet(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            hidden_dim: int = 32,
            normalized_image: bool = False,
            activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            f"This model must be used with a gym.spaces.Box observation space, not {observation_space}"
        )
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            f"You should use this model only with images not with {observation_space}"
        )

        input_channels = observation_space.shape[0]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.extractor = nn.Sequential(
            ConvBlock(input_channels, hidden_dim, 3, 1, 0, 1, activation),  # 4 x 42 x 42 -> 16 x 40 x 40
            MBConvBlock(hidden_dim, hidden_dim * 2, 3, 1, 0, 2, 2, activation),  # 16 x 40 x 40 -> 32 x 38 x 38
            MBConvBlock(hidden_dim * 2, hidden_dim * 4, 5, 2, 0, 2, 2, activation),  # 32 x 38 x 38 -> 64 x 17 x 17
            MBConvBlock(hidden_dim * 4, hidden_dim * 8, 5, 2, 0, 1, 2, activation),  # 64 x 17 x 17 -> 128 x 7 x 7
            ConvBlock(hidden_dim * 8, features_dim, 1, 1, 0, 1, activation),  # 128 x 7 x 7 -> 512 x 7 x 7
        )

    def forward(self, x):
        x = self.extractor(x)  # 4 x 42 x 42 -> 512 x 7 x 7
        x = self.avgpool(x)  # 512 x 7 x 7 -> 512 x 1 x 1
        x = self.flatten(x)  # 512 x 1 x 1 -> 512
        return x
