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

def train(exp_dir="logs",
          model_name: str = "cnn_planner",
          num_epoch: int = 50,
          lr: float = 1e-3,
          batch_size: int = 8,
          seed: int = 2024,
          alpha: float = 10.0,          # Lateral loss weight
          beta: float = 5.0,            # Coverage loss weight
          target_coverage: float = 1.0,   # Desired minimum lateral span (adjust based on your data scale)
          patience: int = 5,
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

    # Initialize the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Initialize the loss function
    loss_fn = nn.MSELoss()  # Using MSE loss for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    global_step = 0
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epoch):
        train_metrics.reset()
        val_metrics.reset()
        model.train()

        epoch_train_loss = 0.0
        train_batches = 0

        for batch in train_data:
            # Ensure each batch item is a tensor
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor)
                         else torch.tensor(v, device=device))
                     for k, v in batch.items()}
            image = batch["image"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            pred = model(image)
            train_metrics.add(pred, waypoints, waypoints_mask)

            # Split predictions into longitudinal and lateral components.
            # Expected shape: (B, n_waypoints, 2)
            pred_long = pred[..., 0]  # (B, n_waypoints)
            pred_lat = pred[..., 1]   # (B, n_waypoints)
            gt_long = waypoints[..., 0]
            gt_lat = waypoints[..., 1]

            loss_long = loss_fn(pred_long, gt_long)
            loss_lat = loss_fn(pred_lat, gt_lat)

            # Compute lateral span (coverage) per sample: difference between maximum and minimum lateral prediction.
            lat_max, _ = torch.max(pred_lat, dim=1)  # shape (B,)
            lat_min, _ = torch.min(pred_lat, dim=1)  # shape (B,)
            coverage = lat_max - lat_min             # (B,)
            # Coverage loss: penalize if the coverage is below target_coverage.
            coverage_loss = F.relu(target_coverage - coverage).mean()

            # Total loss
            total_loss = loss_long + alpha * loss_lat + beta * coverage_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            train_batches += 1
            global_step += 1

        avg_train_loss = epoch_train_loss / train_batches

        # Evaluate on validation set.
        epoch_val_loss = 0.0
        val_batches = 0
        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor)
                             else torch.tensor(v, device=device))
                         for k, v in batch.items()}
                image = batch["image"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                pred = model(image)
                val_metrics.add(pred, waypoints, waypoints_mask)

                pred_long = pred[..., 0]
                pred_lat = pred[..., 1]
                gt_long = waypoints[..., 0]
                gt_lat = waypoints[..., 1]
                loss_long = loss_fn(pred_long, gt_long)
                loss_lat = loss_fn(pred_lat, gt_lat)
                lat_max, _ = torch.max(pred_lat, dim=1)
                lat_min, _ = torch.min(pred_lat, dim=1)
                coverage = lat_max - lat_min
                coverage_loss = F.relu(target_coverage - coverage).mean()
                batch_loss = loss_long + alpha * loss_lat + beta * coverage_loss
                epoch_val_loss += batch_loss.item()
                val_batches += 1

        avg_val_loss = epoch_val_loss / val_batches

        training_metrics = train_metrics.compute()
        validation_metrics = val_metrics.compute()

        train_l1_error = training_metrics["l1_error"]
        train_long_error = training_metrics["longitudinal_error"]
        train_lat_error = training_metrics["lateral_error"]

        val_l1_error = validation_metrics["l1_error"]
        val_long_error = validation_metrics["longitudinal_error"]
        val_lat_error = validation_metrics["lateral_error"]

        logger.add_scalar("train/loss", avg_train_loss, global_step)
        logger.add_scalar("val/loss", avg_val_loss, global_step)
        logger.add_scalar("train/l1_error", train_l1_error, global_step)
        logger.add_scalar("train/longitudinal_error", train_long_error, global_step)
        logger.add_scalar("train/lateral_error", train_lat_error, global_step)
        logger.add_scalar("val/l1_error", val_l1_error, global_step)
        logger.add_scalar("val/longitudinal_error", val_long_error, global_step)
        logger.add_scalar("val/lateral_error", val_lat_error, global_step)

        print(f"Epoch {epoch+1}/{num_epoch}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train L1: {train_l1_error:.4f}, Val L1: {val_l1_error:.4f}, "
              f"Train Long: {train_long_error:.4f}, Val Long: {val_long_error:.4f}, "
              f"Train Lat: {train_lat_error:.4f}, Val Lat: {val_lat_error:.4f}")

        scheduler.step()

        # Early stopping based on validation loss.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=10.0, help="Lateral loss weight")
    parser.add_argument("--beta", type=float, default=5.0, help="Coverage loss weight")
    parser.add_argument("--target_coverage", type=float, default=1.0, help="Desired minimum lateral span")
    parser.add_argument("--patience", type=int, default=5)
    train(**vars(parser.parse_args()))
