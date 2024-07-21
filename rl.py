from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym
from time import sleep


def print_envs():
    print(gym.envs.registration.registry.keys())


def initialize():
    vec_env = make_atari_env("ALE/Pong-v5", n_envs=6, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO("CnnPolicy", vec_env, verbose=0)
    return model


def train(model, steps: int = 1000):
    model.learn(total_timesteps=steps)
    return model


def test(model, episodes: int = 1, render: bool = False) -> list:
    # Create the environment
    test_env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
    test_env = VecFrameStack(test_env, n_stack=4)
    delay = 1.0 / 24.0

    episode_rewards = []
    for episode in range(episodes):
        # obs shape (n_envs, 84, 84, n_frame_stacks)
        obs = test_env.reset()
        dones = False
        total_reward = 0

        while not dones:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, dones, info = test_env.step(action)
            total_reward += reward
            if render:
                test_env.render("human")
                sleep(delay)

        episode_rewards.append(total_reward)
        print(f"Episode {episode} reward: {total_reward}")

    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")

    return episode_rewards
