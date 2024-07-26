import copy
import math
import torch
import torch.nn as nn
import gymnasium as gym
from functools import partial
from dataclasses import dataclass
from typing import List, Callable, Union, Tuple, Optional, Sequence, Type, Literal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, MBConv)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
    ) -> None:
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, FusedMBConv)


class Conv2dNormActivation(torch.nn.Sequential):
    """
        Configurable block used for Convolution2d-Normalization-Activation blocks.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
            kernel_size: (int, optional): Size of the convolving kernel. Default: 3
            stride (int, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
            activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
            dilation (int): Spacing between kernel elements. Default: 1
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Optional[Union[int, Tuple[int, ...], str]] = None,
            groups: int = 1,
            norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
            activation_layer: Callable[..., nn.Module] = nn.ReLU,
            dilation: Union[int, Tuple[int, ...]] = 1,
    ) -> None:
        assert norm_layer is not None, "A norm_layer should be specified."
        assert activation_layer is not None, "An activation_layer should be specified."
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                raise ValueError("Padding should be specified when kernel_size or dilation are tuples.")

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
            ),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        ]

        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., nn.Module], optional): ``delta`` activation. Default: ``nn.ReLU``
        scale_activation (Callable[..., nn.Module]): ``sigma`` activation. Default: ``nn.Sigmoid``
    """

    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., nn.Module] = nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(inputs)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self._scale(inputs)
        return scale * inputs


class MBConv(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            SqueezeExcitation(expanded_channels, squeeze_channels, activation=partial(activation_layer, inplace=True))
        )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.block(inputs)
        if self.use_res_connect:
            result += inputs
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf: FusedMBConvConfig,
            norm_layer: Callable[..., nn.Module],
            activation_layer: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.block(inputs)
        if self.use_res_connect:
            result += inputs
        return result


class EfficientNet(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            features_dim: int = 512,
            activation_layer: Type[nn.Module] = nn.Mish,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            normalized_image: bool = False,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            features_dim (int): Number of features in the output layer
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        assert isinstance(observation_space, gym.spaces.Box), (
            f"Model must be used with a gym.spaces.Box observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            f"You should use a CNN only with images not with {observation_space}"
        )
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        n_input_channels = observation_space.shape[0]
        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                n_input_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                stage.append(block_cnf.block(block_cnf, norm_layer, activation_layer))

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                features_dim,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def efficientnet_config(
        arch: Literal["efficientnet", "efficientnet_v2"],
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        features_dim: int = 512,
        activation_layer: Type[nn.Module] = nn.Mish,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d
) -> dict:

    if arch == "efficientnet":
        bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
        # expand_ratio, kernel, stride, input_channels, out_channels, num_layers
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 4),
            bneck_conf(6, 3, 1, 112, 192, 1),
        ]
    elif arch == "efficientnet_v2":
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 16, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 3),
            MBConvConfig(4, 3, 2, 48, 96, 4),
            MBConvConfig(4, 3, 2, 96, 128, 5),
        ]
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return {
        "inverted_residual_setting": inverted_residual_setting,
        "features_dim": features_dim,
        "activation_layer": activation_layer,
        "norm_layer": norm_layer,
    }
