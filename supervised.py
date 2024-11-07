import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import NatureCNN
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision


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
    """Subclass of NatureCNN that adds a linear layer to include the action network."""
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


def train(fabric, model, optimizer, scheduler, dataloader, num_epochs=1, log_interval=10, num_classes=6, label_smoothing=0.):
    # initialize metrics
    model.train()
    step = 0
    running_loss = 0.
    running_kl_loss = 0.
    mean_loss = 0.
    metric_collection = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "precision": MulticlassPrecision(num_classes=num_classes),
    }).to(fabric.device)

    # training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # increment step
            step += 1
            # split batch
            inputs, target = batch
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            logits = model(inputs)
            # compute loss
            loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=label_smoothing)
            kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits, dim=1), target, reduction="batchmean")
            # backpropagation
            fabric.backward(loss)
            optimizer.step()
            # Update the learning rate
            scheduler.step()

            # update metrics
            running_loss += loss.item()
            running_kl_loss += kl_loss.item()
            metric_collection.update(logits, target.argmax(1))
            if step % log_interval == 0:
                # compute mean loss
                mean_loss = running_loss / log_interval
                mean_kl_loss = running_kl_loss / log_interval
                metrics = {"loss": mean_loss, "kl_loss": mean_kl_loss}
                # compute additional metrics
                metrics.update(metric_collection.compute())
                # log learning rate
                metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                # log metrics
                fabric.log_dict(metrics, step)
                # reset metrics
                running_loss = 0.
                running_kl_loss = 0.
                metric_collection.reset()

        fabric.print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    # save final model
    fabric.call("on_train_end", fabric=fabric, loss=mean_loss, model=model, optimizer=optimizer, step=step)
