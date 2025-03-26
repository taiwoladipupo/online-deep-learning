import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from torch import nn

from .models import  load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
def train(
        exp_dir: str = "logs",
        model_name: str = "Classifier",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
        alpha = 1,
        beta = 0.7,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Loss function and optimizer
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step  = 0
    training_metrics = DetectionMetric()
    validation_metrics = DetectionMetric()

    # Training loop
    for epoch in range(num_epoch):
        # clear all available metrics
        training_metrics.reset()

        model.train()

        for batch in train_data:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            img = batch["image"]
            track = batch["track"]
            depth = batch["depth"]

            optimizer.zero_grad()
            pred, pred_depth = model(img)
            pred_labels = pred.argmax(dim=1)

            logits = torch.nn.functional.one_hot(track, num_classes=3).permute(0, 2, 1).float()

            print({"img": img.shape,
                   "depth": depth.shape,
                   "track": track.shape,
                   "logits": logits.shape,})

            training_metrics.add(pred_labels, track, depth)

            loss = alpha * ce_loss(pred, logits) + beta * mse_loss(pred_depth, depth)
            loss.backward()
            optimizer.step()

            global_step += 1

            with torch.inference_mode():
                model.eval()

                for batch in train_data:
                    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in
                             batch.items()}
                    img = batch["image"]
                    track = batch["track"]
                    depth = batch["depth"]

                    pred, pred_depth = model(img)
                    pred_labels = pred.argmax(dim=1)
                    validation_metrics.add(pred_labels, track, pred_depth, depth)

            # log accuracy to tensorboard
            accuracy_metrics = validation_metrics.compute()
            epoch_train_acc = torch.tensor(training_metrics.compute()["accuracy"])
            epoch_val_acc = torch.tensor(validation_metrics.compute()["accuracy"])
            iou = torch.tensor(accuracy_metrics["iou"])
            depth_error = torch.tensor(validation_metrics.compute()["depth_error"])
            tp_depth_error = torch.tensor(validation_metrics.compute()["tp_depth_error"])

            logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
            logger.add_scalar("validation_accuracy", epoch_val_acc, global_step)
            logger.add_scalar("accuracy", accuracy_metrics["accuracy"], global_step)
            logger.add_scalar("iou", iou, global_step)
            logger.add_scalar("depth_error", depth_error, global_step)
            logger.add_scalar("tp_depth_error", tp_depth_error, global_step)

            # print on first, last, every 10th epoch
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                    f"train_acc={epoch_train_acc:.4f} "
                    f"val_acc={epoch_val_acc:.4f}"
                    f"accuracy={accuracy_metrics['accuracy']:.4f} "
                    f"depth_error={depth_error:.4f} "
                    f"tp_depth_error={tp_depth_error:.4f}"
                )
            # scheduler.step()
    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument("--exp_dir", type=str, default="logs")
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--num_epoch", type=int, default=50)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--seed", type=int, default=2024)

        # optional: additional model hyperparameters
        # parser.add_argument("--num_layers", type=int, default=3)

        # pass all arguments to train
        train(**vars(parser.parse_args()))