import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Callable, Dict, Optional, Type, Union

import stable_baselines3.common.atari_wrappers as sb3_wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


class CropFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(160, 160, 1),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        # Crop and select the blue channel
        frame = frame[34:194, :, 2]
        # Maximize contrast
        frame[frame == 17] = 0  # background
        frame[frame == 74] = 85  # opponent paddle
        frame[frame == 92] = 170  # ai paddle
        frame[frame == 236] = 255  # ball
        return frame[:, :, None]


class CustomAtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This corresponds to repeating the action ``frame_skip`` times.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = sb3_wrapper.StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = sb3_wrapper.NoopResetEnv(env, noop_max=noop_max)
        if frame_skip > 1:
            env = sb3_wrapper.MaxAndSkipEnv(env, skip=frame_skip)
        env = sb3_wrapper.EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = sb3_wrapper.FireResetEnv(env)
        env = CropFrame(env)
        env = sb3_wrapper.ClipRewardEnv(env)

        super().__init__(env)


def make_custom_atari_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=CustomAtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
