import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.algorithm import Algorithm


def get_algo_weights(algo):
    # Get weights of the default local policy
    return algo.get_policy().get_weights()


def load_algo(algo, path: str):
    # Load the trained model
    return algo.restore(path)


def load_algo_new(path):
    # Load the trained model
    return Algorithm.from_checkpoint(path)


def init():
    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment("ALE/Pong-v5")
        .framework("torch")
        .env_runners(num_env_runners=4)
        .training(
            train_batch_size=750,
            sgd_minibatch_size=128,
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.3,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_sgd_iter=2,
            lr=0.00015,
            grad_clip=100.0,
            grad_clip_by="global_norm",
            vf_loss_coeff=1.0,
        )
        .model(
            {
                "vf_share_layers": True,
                "conv_filters": [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
                "conv_activation": "relu",  # Mish
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",  # Mish
                "dim": 84,  # image size faster: 42
            }
        )
        .resources(num_gpus=1)
        .callbacks([TBXLoggerCallback(logdir="./logs")])
    )

    # Create the PPO trainer
    algo = config.build()

    return algo


def train(algo, episodes: int = 10):
    # Training loop
    for i in range(episodes):
        result = algo.train()
        print(f"Iteration: {i}, reward: {result["env_runners"]["episode_return_mean"]}")

        # Save the model every 10 iterations
        if i % 10 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")

    return algo


def test(algo, episodes: int = 1) -> list:
    # Create the environment
    env = gym.make("ALE/Pong-v5", render_mode="human")

    # Run a few episodes
    episode_rewards = []
    for episode in range(episodes):
        obs, info = env.reset()
        episode_over = False
        total_reward = 0

        while not episode_over:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated

        episode_rewards.append(total_reward)
        print(f"Episode {episode} reward: {total_reward}")

    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")

    return episode_rewards
