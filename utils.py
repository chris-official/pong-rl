from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


def print_model_parameters(model: PPO, print_layers: bool = False, shared_extractor: bool = True) -> None:
    """Print the number of parameters of a Stable-Baselines3 model."""
    modules = model.policy._modules.keys()

    total_params = 0
    for module in modules:
        module_params = 0
        for name, param in model.policy._modules[module].named_parameters():
            if param.requires_grad:
                count = param.numel()
                module_params += count
                if shared_extractor:
                    if module not in ["pi_features_extractor", "vf_features_extractor"]:
                        total_params += count
                else:
                    if module != "features_extractor":
                        total_params += count
                if print_layers:
                    print(f"    {name}: {count:,}")
        print(f"{module}: {module_params:,}")
    print(f"Total number of parameters: {total_params:,}")


def evaluate(model: PPO, env: VecEnv = None, episodes: int = 1, deterministic: bool = False) -> tuple:
    if env is None:
        env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, episodes, deterministic)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward
