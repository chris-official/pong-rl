from typing import Type
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import gymnasium as gym
import torch
from wrapper import make_custom_atari_env


def setup_vec_env(
        n_envs: int = 1,
        frame_skip: int = 4,
        action_repeat_probability: float = 0.0,
        monitor_log_path: str = None,
        frame_stacks: int = 4,
        seed: int = None,
) -> VecEnv:
    # Create the environment
    vec_env = make_custom_atari_env(
        "PongNoFrameskip-v4",
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs={
            "noop_max": 30,
            "frame_skip": frame_skip,
            "action_repeat_probability": action_repeat_probability,
            "monitor_log_path": monitor_log_path,
        },
    )
    # Stack multiple frames
    vec_env = VecFrameStack(vec_env, n_stack=frame_stacks)
    return vec_env


def setup_model(
        vec_env: VecEnv,
        model_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_kwargs: dict = None,
        net_arch: dict = None,
        device: str = "auto",
        log_dir: str = None,
) -> PPO:
    if model_kwargs is None:
        model_kwargs = {"features_dim": 256}
    if net_arch is None:
        net_arch = {"pi": [64], "vf": [64]}
    # Create the model
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        stats_window_size=100,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Mish,
            ortho_init=True,
            features_extractor_class=model_class,
            features_extractor_kwargs=model_kwargs,
            share_features_extractor=True,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(weight_decay=1e-4),
            normalize_images=True,
        ),
        verbose=0,
        seed=None,
        device=device,
    )
    return model


def setup_callback(eval_env: VecEnv, n_envs: int = 8, eval_freq: int = 100_000, n_eval_episodes: int = 3,
                   log_path: str = None, best_model_save_path: str = None) -> EvalCallback:
    callback = EvalCallback(
        eval_env=eval_env,
        callback_on_new_best=None,
        callback_after_eval=None,
        n_eval_episodes=n_eval_episodes,
        eval_freq=max(eval_freq // n_envs, 1),
        log_path=log_path,
        best_model_save_path=best_model_save_path,
        deterministic=True,
        render=False,
        verbose=1,
        warn=True,
    )
    return callback


def train(model: PPO, steps: int = 1000, name: str = "pong_ppo", new_run: bool = True, callbacks: list = None) -> PPO:
    model.learn(total_timesteps=steps, tb_log_name=name, reset_num_timesteps=new_run, callback=callbacks)
    return model


def save(model: PPO, path: str) -> None:
    model.save(path)


def load(path: str, vec_env: VecEnv = None) -> PPO:
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
