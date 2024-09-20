import numpy as np
import torch
import wandb
import lightning as L
import gymnasium as gym
from dataset import PongDataset
from torch.utils.data import DataLoader
from stable_baselines3.common.torch_layers import NatureCNN
from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from datasets.atari import AtariPongDataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassAUROC


class SaveBestModel:
    def __init__(self, path="checkpoints"):
        self.best_loss = float("inf")
        self.path = path

    def on_train_epoch_end(self, fabric, loss, model, optimizer, step):
        if loss < self.best_loss:
            self.best_loss = loss
            fabric.save(
                self.path + f"/best_checkpoint_step={step}_loss={loss:.4f}.ckpt",
                {"model": model, "optimizer": optimizer, "step": step}
            )

    def on_train_end(self, fabric, loss, model, optimizer, step):
        fabric.save(
            self.path + f"/final_checkpoint_step={step}_loss={loss:.4f}.ckpt",
            {"model": model, "optimizer": optimizer, "step": step}
        )


class PolicyNetwork(NatureCNN):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
            out_classes: int = 6,
    ) -> None:
        super().__init__(observation_space, features_dim, normalized_image)
        self.action_net = torch.nn.Linear(features_dim, out_classes)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.linear(self.cnn(observations))
        return self.action_net(x)


def train(fabric, model, optimizer, dataloader, num_epochs=1, log_interval=10):
    model.train()
    step = 0
    running_loss = 0.
    running_kl_loss = 0.
    mean_loss = 0.
    metric_collection = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "precision": MulticlassPrecision(num_classes=num_classes),
        "auroc": MulticlassAUROC(num_classes=num_classes),
    }).to(fabric.device)
    # training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            step += 1
            inputs, target = batch
            optimizer.zero_grad()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, target)
            kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits, dim=1), target, reduction="batchmean")
            fabric.backward(loss)
            optimizer.step()

            # Log metrics
            running_loss += loss.item()
            running_kl_loss += kl_loss.item()
            metric_collection.update(logits, target.argmax(1))
            if step % log_interval == 0:
                mean_loss = running_loss / log_interval
                mean_kl_loss = running_kl_loss / log_interval
                metrics = {"loss": mean_loss, "kl_loss": mean_kl_loss}
                # compute additional metrics
                metrics.update(metric_collection.compute())
                # log metrics
                fabric.log_dict(metrics, step)
                # reset metrics
                running_loss = 0.
                running_kl_loss = 0.
                metric_collection.reset()

        fabric.print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    # save final model
    fabric.call("on_train_end", fabric=fabric, loss=mean_loss, model=model, optimizer=optimizer, step=step)


def main():
    # Settings
    num_epochs = 10
    lr = 1e-3
    batch_size = 64
    framestack = 4
    log_interval = 20
    log_dir = "logs"
    model_dir = "models"
    name = "my-first-run"
    seed = 1
    accelerator = "cuda"
    precision = "32-true"
    epsilon = 0.01
    dataset_length = 10_000
    features_dim = 512

    # Setup logger
    tb_logger = TensorBoardLogger(root_dir=log_dir, name=name)
    wandb_logger = WandbLogger(save_dir=log_dir, name=name, project="Pong-RL", entity="iu-projects")

    # Log hyperparameters
    wandb_logger.experiment.config.update({
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "framestack": framestack,
        "seed": seed,
        "accelerator": accelerator,
        "precision": precision,
        "epsilon": epsilon,
        "dataset_length": dataset_length,
        "features_dim": features_dim,
    })

    # Add custom callback
    save_best_model = SaveBestModel(model_dir)

    # Configure Fabric
    fabric = L.Fabric(
        accelerator=accelerator,
        precision=precision,
        callbacks=[save_best_model],
        loggers=[tb_logger, wandb_logger]
    )

    # Set seed
    fabric.seed_everything(seed)

    # Instantiate objects
    obs_space = gym.spaces.Box(low=0, high=1, shape=(framestack, 84, 84), dtype=np.float32)
    with fabric.init_module():
        model = PolicyNetwork(obs_space, features_dim=features_dim, normalized_image=True, out_classes=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create dataloader
    dataset = AtariPongDataset(dataset_length=dataset_length, epsilon=epsilon, seed=seed, mode="proba")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Set float32 matmul precision to utilize tensor cores
    torch.set_float32_matmul_precision("high")

    # Set up objects
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    # Run training loop
    train(fabric, model, optimizer, dataloader, num_epochs, log_interval)

    wandb.finish()


if __name__ == "__main__":
    main()
