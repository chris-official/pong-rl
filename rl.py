from typing import Type, Literal
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack, VecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, DQN
from time import sleep
import torch
import numpy as np
from wrapper import make_custom_atari_env


def evaluate(model: PPO | DQN, env: VecEnv = None, episodes: int = 1, deterministic: bool = True) -> tuple:
    if env is None:
        env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, episodes, deterministic)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward
