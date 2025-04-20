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
    alpha_long: float = 1.0,
    alpha_lat: float = 0.2,
    patience: int = 5,
    **model_kwargs
):
    #—— setup ——
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else torch.device("cpu")
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now():%m%d_%H%M%S}"
    writer  = tb.SummaryWriter(log_dir)

    augment = T.Compose([
        T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.1)], p=0.7),
        T.RandomAffine(0, translate=(0.05,0.0)),
    ])

    model        = load_model(model_name, **model_kwargs).to(device)
    train_loader = load_data("drive_data/train", shuffle=True,  batch_size=batch_size, num_workers=2)
    val_loader   = load_data("drive_data/val",   shuffle=False)

    mse_loss    = nn.MSELoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                       mode="min", factor=0.5, patience=2, verbose=True)

    train_metrics = PlannerMetric()
    val_metrics   = PlannerMetric()

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, num_epoch+1):
        # —— train epoch ——
        model.train()
        train_metrics.reset()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
            imgs = augment(batch["image"])
            wpts, mask = batch["waypoints"], batch["waypoints_mask"]

            preds = model(imgs)
            train_metrics.add(preds, wpts, mask)

            long_p, lat_p = preds[...,0], preds[...,1]
            long_g, lat_g = wpts[...,0],    wpts[...,1]
            loss = alpha_long*mse_loss(long_p, long_g) + alpha_lat*mse_loss(lat_p, lat_g)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # —— validate epoch ——
        model.eval()
        val_metrics.reset()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
                imgs = batch["image"]
                wpts, mask = batch["waypoints"], batch["waypoints_mask"]

                preds = model(imgs)
                val_metrics.add(preds, wpts, mask)

                long_p, lat_p = preds[...,0], preds[...,1]
                long_g, lat_g = wpts[...,0],    wpts[...,1]
                val_loss += (alpha_long*mse_loss(long_p, long_g)
                           + alpha_lat*mse_loss(lat_p, lat_g)).item()
        val_loss /= len(val_loader)

        # —— compute & log metrics ——
        tm = train_metrics.compute()
        vm = val_metrics.compute()
        print(
            f"Epoch {epoch:02d}/{num_epoch:02d} | "
            f"Val MSE: {val_loss:.4f} | "
            f"Train Long Err: {tm['longitudinal_error']:.4f} | Val Long Err: {vm['longitudinal_error']:.4f} | "
            f"Train Lat Err: {tm['lateral_error']:.4f}     | Val Lat Err: {vm['lateral_error']:.4f}"
        )
        for tag, val in [
            ("val/mse",    val_loss),
            ("train/long", tm["longitudinal_error"]),
            ("val/long",   vm["longitudinal_error"]),
            ("train/lat",  tm["lateral_error"]),
            ("val/lat",    vm["lateral_error"]),
        ]:
            writer.add_scalar(tag, val, epoch)

        # —— early stopping & lr scheduling ——
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_model(model)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Done. best val MSE: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir",    type=str,   default="logs")
    p.add_argument("--model_name", type=str,   required=True)
    p.add_argument("--num_epoch",  type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--seed",       type=int,   default=2024)
    p.add_argument("--alpha_long", type=float, default=1.0)
    p.add_argument("--alpha_lat",  type=float, default=0.2)
    p.add_argument("--patience",   type=int,   default=5)
    args = p.parse_args()

    train(**vars(args))
