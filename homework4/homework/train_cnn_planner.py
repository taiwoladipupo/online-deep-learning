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
from torchvision import transforms as T
def train(exp_dir = "logs",
          model_name: str = "cnn_planner",
          num_epoch: int = 50,
          lr: float = 1e-3,
          batch_size: int = 8,
          seed: int = 2024,
          alpha=10.0,
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

    # augmentation
    augment = T.Compose([
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.7),
        T.RandomAffine(0, translate=(0.05, 0.0)),
    ])

    # Initialize the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Initialize the loss function
    loss_fn = nn.MSELoss() # Assuming the loss function is MSE for regression tasks

    # Initialize the optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    global_step = 0
    # Initialize the metric
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    best_val_loss = float("inf")
    patience_counter = 0
    best_val_flag = False

    for epoch in range(num_epoch):
        # Reset the metrics at the start of each epoch
        train_metrics.reset()
        val_metrics.reset()
        model.train()

        epoch_train_loss = 0.0
        train_batches = 0

        for batch in train_data:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            image = batch["image"]
            image = augment(image)
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            pred = model(image)
            train_metrics.add(pred, waypoints, waypoints_mask)

            ## Ensuring the predictions and waypoints are tensors
            if isinstance(pred, list):
                pred = torch.tensor(pred, device=device)
            if isinstance(waypoints, list):
                waypoints = torch.tensor(waypoints, device=device)

            # Loss calculation
            pred_long = pred[..., 0]
            pred_lat = pred[..., 1]
            gt_long = waypoints[..., 0]
            gt_lat = waypoints[..., 1]



            loss_long = loss_fn(pred_long, gt_long)
            loss_lat = loss_fn(pred_lat, gt_lat)
            loss = loss_long + alpha * loss_lat

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1
            global_step += 1


        # Evaluate the model on the validation set
        epoch_val_loss = 0.0
        val_batches = 0

        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                image = batch["image"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                pred = model(image)

                # Update the validation metrics
                val_metrics.add(pred, waypoints, waypoints_mask)

                pred_long = pred[..., 0]
                pred_lat = pred[..., 1]
                gt_long = waypoints[..., 0]
                gt_lat = waypoints[..., 1]
                loss_long = loss_fn(pred_long, gt_long)
                loss_lat = loss_fn(pred_lat, gt_lat)
                batch_loss = loss_long + alpha * loss_lat
                epoch_val_loss += batch_loss.item()
                val_batches += 1
        avg_val_loss = epoch_val_loss / val_batches

        # Log the metrics
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
        # once both criteria are satisfied, save the model
        if not best_val_flag and val_longitudinal_error < 0.30 and val_lateral_error < 0.45:
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}.th")
            print(f"Thresholds met at epoch {epoch + 1}, saved to {log_dir} / {model_name}.th")
            best_val_flag = True
            break


        # Step the scheduler
        scheduler.step()
        patience_counter = 0
        # Early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if not best_val_flag:
        print("Best validation loss not achieved, saving the model")
        # Save the model
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


