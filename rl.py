from typing import Type
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecEnv
from stable_baselines3 import PPO
import gymnasium as gym
import torch


def setup_vec_env(
        n_envs: int = 1,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_repeat_probability: float = 0.0,
        frame_stacks: int = 4,
        seed: int = 0,
) -> VecEnv:
    # Create the environment
    vec_env = make_atari_env(
        "PongNoFrameskip-v4",
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs={
            "noop_max": 30,
            "frame_skip": frame_skip,
            "screen_size": screen_size,
            "terminal_on_life_loss": True,
            "clip_reward": True,
            "action_repeat_probability": action_repeat_probability,
        },
    )
    # Stack multiple frames
    vec_env = VecFrameStack(vec_env, n_stack=frame_stacks)
    return vec_env


def setup_model(
        vec_env: VecEnv,
        model_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_kwargs: dict = None,
        device: str = "auto",
        log_dir: str = "C:/Users/cgoet/PycharmProjects/Pong-RL/logs/",
) -> PPO:
    if model_kwargs is None:
        model_kwargs = {"features_dim": 256}
    # Create the model
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        stats_window_size=100,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch={"pi": [64], "vf": [64]},
            activation_fn=torch.nn.Mish,
            ortho_init=True,
            features_extractor_class=model_class,
            features_extractor_kwargs=model_kwargs,
            share_features_extractor=True,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=None,
            normalize_images=True,
        ),
        verbose=0,
        seed=None,
        device=device,
    )
    return model


def initialize(
        n_envs: int = 1,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_repeat_probability: float = 0.0,
        frame_stacks: int = 4,
        seed: int = 0,
        model_class: BaseFeaturesExtractor = NatureCNN,
        model_kwargs: dict = None,
        device: str = "auto",
        log_dir: str = "C:/Users/cgoet/PycharmProjects/Pong-RL/logs/",
) -> tuple:
    # Create the environment
    vec_env = setup_vec_env(n_envs, frame_skip, screen_size, action_repeat_probability, frame_stacks, seed)
    # Create the model
    model = setup_model(vec_env, model_class, model_kwargs, device, log_dir)
    return model, vec_env


def train(model: PPO, steps: int = 1000, name: str = "pong_ppo", new_run: bool = True) -> PPO:
    model.learn(total_timesteps=steps, tb_log_name=name, reset_num_timesteps=new_run, callback=None)
    return model


def save(model: PPO, path: str = "C:/Users/cgoet/PycharmProjects/Pong-RL/models/pong_ppo") -> None:
    model.save(path)


def load(path: str = "C:/Users/cgoet/PycharmProjects/Pong-RL/models/pong_ppo", vec_env=None) -> PPO:
    model = PPO.load(path)
    if vec_env:
        model.set_env(vec_env)
    return model


def evaluate(model: PPO, env: VecEnv, episodes: int = 1, deterministic: bool = True) -> tuple:
    mean_reward, std_reward = evaluate_policy(model, env, episodes, deterministic)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def visualize(model: PPO = None, screen_size: int = 84, episodes: int = 1) -> list:
    env = gym.make(
        "PongNoFrameskip-v4",
        difficulty=None,
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0.,
        render_mode="human",
    )
    input_env = WarpFrame(env, width=screen_size, height=screen_size)
    input_env = DummyVecEnv([lambda: input_env])
    input_env = VecFrameStack(input_env, n_stack=4)

    episode_rewards = []
    for episode in range(episodes):
        obs = input_env.reset()
        dones = False
        total_reward = 0

        while not dones:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = [input_env.action_space.sample()]
            obs, reward, dones, info = input_env.step(action)
            total_reward += reward[0]

        episode_rewards.append(total_reward)
        print(f"Episode {episode} reward: {total_reward}")

    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")

    return episode_rewards
