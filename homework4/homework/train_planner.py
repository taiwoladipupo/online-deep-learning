import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch import nn
from .metrics import PlannerMetric
from .models import load_model, save_model
from .datasets.road_dataset import load_data


def train(exp_dir: str = "logs",
          model_name: str = "mlp_planner",
          num_epoch: int = 50,
          lr: float = 1e-3,
          batch_size: int = 8,
          seed: int = 2024,
          alpha=10.0,
          **kwargs):
    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)


    loss_fn = nn.MSELoss()  # Assuming the loss function is MSE for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    for epoch in range(num_epoch):
        # Reset Metrics at beginning of each epoch
        train_metrics.reset()
        val_metrics.reset()

        for batch in train_data:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            track_left = batch["track_left"]
            track_right = batch["track_right"]
            waypoints = batch["waypoints"]
            mask = batch["waypoints_mask"]

            optimizer.zero_grad()
            # forward pass
            pred = model(track_left, track_right)

            pred_long = pred[..., 0]
            pred_lat = pred[..., 1]
            gt_long = waypoints[..., 0]
            gt_lat = waypoints[..., 1]

            # Compute the loss using L1 loss
            loss_long = F.l1_loss(pred_long, gt_long)
            loss_lat = F.l1_loss(pred_lat, gt_lat)
            loss = loss_long + alpha * loss_lat


            loss.backward()
            optimizer.step()

            # Update metrics
            train_metrics.add(pred, waypoints, mask)

            global_step += 1

        # Evaluation mode

        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                track_left = batch["track_left"]
                track_right = batch["track_right"]
                waypoints = batch["waypoints"]
                mask = batch["waypoints_mask"]

                pred = model(track_left, track_right)

                # Update metrics
                val_metrics.add(pred, waypoints, mask)

        training_metrics = train_metrics.compute()
        validation_metrics = val_metrics.compute()

        # Compute longitudinal and lateral errors
        train_l1_error = training_metrics["l1_error"]
        train_longitudinal_error = training_metrics["longitudinal_error"]
        train_lateral_error = training_metrics["lateral_error"]

        # Log metrics
        logger.add_scalar("train/l1_error", train_l1_error, global_step)
        logger.add_scalar("train/longitudinal_error", train_longitudinal_error, global_step)
        logger.add_scalar("train/lateral_error", train_lateral_error, global_step)

        val_l1_error = validation_metrics["l1_error"]
        val_longitudinal_error = validation_metrics["longitudinal_error"]
        val_lateral_error = validation_metrics["lateral_error"]

        logger.add_scalar("val/l1_error", val_l1_error, global_step)
        logger.add_scalar("val/longitudinal_error", val_longitudinal_error, global_step)
        logger.add_scalar("val/lateral_error", val_lateral_error, global_step)

        # Print metrics
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}, - "
              f"Train L1 Error: {train_l1_error:.4f} "
              f"Val L1 Error: {val_l1_error:.4f} "
              f"Train Longitudinal Error: {train_longitudinal_error:.4f} "
              f"Val Longitudinal Error: {val_longitudinal_error:.4f} "
              f"Train Lateral Error: {train_lateral_error:.4f} "
              f"Val Lateral Error: {val_lateral_error:.4f}")

    # Save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))

# print("Time to train")
