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
from .models import load_model, save_model


def train(
    exp_dir: str = "logs",
    model_name: str = "cnn_planner",
    num_epoch: int = 60,
    lr: float = 1e-4,
    batch_size: int = 8,
    seed: int = 2024,
    alpha_long: float = 1.0,
    alpha_lat: float = 0.5,
    patience: int = 5,
    **model_kwargs
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging
    run_dir = Path(exp_dir) / f"{model_name}_{datetime.now():%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = tb.SummaryWriter(run_dir)

    # Data augmentation
    augment = T.Compose([
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        T.RandomAffine(0, translate=(0.05, 0.0))
    ])

    # Model
    model = load_model(model_name, **model_kwargs).to(device)

    # Data loaders
    train_loader = load_data("drive_data/train", shuffle=True,
                              batch_size=batch_size, num_workers=4)
    val_loader   = load_data("drive_data/val",   shuffle=False)

    # Loss, optimizer, scheduler
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, num_epoch + 1):
        # Training
        model.train()
        for batch in train_loader:
            img = augment(batch["image"]).to(device)
            wp  = batch["waypoints"].to(device)

            pred = model(img)
            long_pred = pred[..., 0]
            lat_pred  = pred[..., 1]
            long_gt   = wp[..., 0]
            lat_gt    = wp[..., 1]

            loss = alpha_long * mse(long_pred, long_gt) + alpha_lat * mse(lat_pred, lat_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                wp  = batch["waypoints"].to(device)

                pred = model(img)
                lp, lt = pred[..., 0], pred[..., 1]
                gl, gt = wp[..., 0], wp[..., 1]
                val_loss += (alpha_long * mse(lp, gl) + alpha_lat * mse(lt, gt)).item()
        val_loss /= len(val_loader)

        # Log
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch {epoch}/{num_epoch} - val_loss: {val_loss:.4f}")

        # Early stopping & saving
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            save_model(model)
            print("↳ New best model saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print("Training complete. Best val loss: {:.4f}".format(best_val))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir",       type=str,   default="logs")
    p.add_argument("--model_name",    type=str,   required=True)
    p.add_argument("--num_epoch",     type=int,   default=60)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--alpha_long",    type=float, default=1.0)
    p.add_argument("--alpha_lat",     type=float, default=0.5)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=2024)
    args = p.parse_args()
    train(**vars(args))
