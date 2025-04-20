import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch import nn
from torchvision import transforms as T

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "cnn_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 2024,
    alpha: float = 10.0,
    patience: int = 5,
    **model_kwargs
):
    # --- Device setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- TensorBoard ---
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now():%m%d_%H%M%S}"
    logger = tb.SummaryWriter(log_dir)

    # --- Data augmentation ---
    augment = T.Compose([
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.7),
        T.RandomAffine(0, translate=(0.05, 0.0)),
    ])

    # --- Model, data, loss, optimizer, scheduler ---
    model = load_model(model_name, **model_kwargs).to(device)
    train_loader = load_data("drive_data/train", shuffle=True,
                             batch_size=batch_size, num_workers=2)
    val_loader   = load_data("drive_data/val",   shuffle=False)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # --- Metrics & early stopping state ---
    train_metrics = PlannerMetric()
    val_metrics   = PlannerMetric()
    best_val_mse  = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epoch + 1):
        # ---- TRAIN ----
        model.train()
        train_metrics.reset()
        train_mse = 0.0
        for batch in train_loader:
            # move to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            imgs = augment(batch["image"])
            wpts, mask = batch["waypoints"], batch["waypoints_mask"]
            preds = model(imgs)
            train_metrics.add(preds, wpts, mask)

            long_p, lat_p = preds[...,0], preds[...,1]
            long_g, lat_g = wpts[...,0],    wpts[...,1]
            loss = loss_fn(long_p, long_g) + alpha * loss_fn(lat_p, lat_g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse += loss.item()

        train_mse /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_metrics.reset()
        val_mse = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                imgs = batch["image"]
                wpts, mask = batch["waypoints"], batch["waypoints_mask"]

                preds = model(imgs)
                val_metrics.add(preds, wpts, mask)

                long_p, lat_p = preds[...,0], preds[...,1]
                long_g, lat_g = wpts[...,0],    wpts[...,1]
                val_mse += (loss_fn(long_p, long_g) +
                            alpha * loss_fn(lat_p, lat_g)).item()

        val_mse /= len(val_loader)

        # ---- METRICS ----
        tm = train_metrics.compute()
        vm = val_metrics.compute()

        # ---- LOG & PRINT ----
        if epoch % 10 == 0 or epoch == num_epoch:
            print(
                f"Epoch {epoch:03d}/{num_epoch:03d} | "
                f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
                f"Train L1: {tm['l1_error']:.4f} | Val L1: {vm['l1_error']:.4f} | "
                f"Train Long: {tm['longitudinal_error']:.4f} | Val Long: {vm['longitudinal_error']:.4f} | "
                f"Train Lat: {tm['lateral_error']:.4f} | Val Lat: {vm['lateral_error']:.4f}"
            )
            # TensorBoard scalars
            logger.add_scalar("train/mse_loss", train_mse, epoch)
            logger.add_scalar("val/mse_loss",   val_mse,   epoch)
            logger.add_scalar("train/l1_error", tm["l1_error"],          epoch)
            logger.add_scalar("val/l1_error",   vm["l1_error"],          epoch)
            logger.add_scalar("train/long_error",tm["longitudinal_error"],epoch)
            logger.add_scalar("val/long_error",  vm["longitudinal_error"],epoch)
            logger.add_scalar("train/lat_error", tm["lateral_error"],     epoch)
            logger.add_scalar("val/lat_error",   vm["lateral_error"],     epoch)

        # ---- EARLY STOPPING / SAVE BEST ----
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            save_model(model)
            print(f"  -> New best MSE: {best_val_mse:.4f}, model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

        scheduler.step()

    print(f"Training complete. Best validation MSE = {best_val_mse:.4f}")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",    type=str,   default="logs")
    parser.add_argument("--model_name", type=str,   required=True)
    parser.add_argument("--num_epoch",  type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--seed",       type=int,   default=2024)
    parser.add_argument("--alpha",      type=float, default=10.0)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()

    train(**vars(args))
