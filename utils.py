import gymnasium as gym
import cv2
import matplotlib.pyplot as plt


def print_policy(model) -> None:
    print(model.policy)


def print_model_parameters(model, print_layers: bool = False, shared_extractor: bool = True) -> None:
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


def plot_processing(screen_size: int = 84) -> None:
    env = gym.make("ALE/Pong-v5", obs_type="rgb", render_mode=None)

    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

    plt.imshow(obs)
    plt.show()

    frame = cv2.resize(obs, (screen_size, screen_size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    plt.imshow(frame, cmap="gray")
    plt.show()
