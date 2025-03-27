import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch import nn

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

# Define the fixed target resolution as expected by the grader.
TARGET_RES = (96, 128)


def train(
        exp_dir: str = "logs",
        model_name: str = "Detector",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
        alpha=1,
        beta=0.7,
        **kwargs,
):
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

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    training_metrics = DetectionMetric()
    validation_metrics = DetectionMetric()

    # Helper function: force a tensor to the fixed target resolution.
    def force_resolution(tensor, mode, target_res=TARGET_RES):
        # Ensure tensor is 4D: (B, C, H, W) for interpolation.
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        return F.interpolate(tensor, size=target_res, mode=mode, align_corners=(mode == 'bilinear')).squeeze(1)

    for epoch in range(num_epoch):
        training_metrics.reset()

        model.train()
        for batch in train_data:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            img = batch["image"]
            track = batch["track"]  # ground truth segmentation (B, H, W) or (B, 1, H, W)
            depth = batch["depth"]  # ground truth depth (B, H, W) or (B, 1, H, W)

            optimizer.zero_grad()
            pred, pred_depth = model(img)
            # Resize predictions to fixed target resolution.
            pred = F.interpolate(pred, size=TARGET_RES, mode='bilinear', align_corners=False)
            pred_labels = pred.argmax(dim=1)  # (B, 96, 128)

            # Force ground truth segmentation to TARGET_RES using nearest interpolation.
            if track.ndim == 4:
                track = track.squeeze(1)
            track = F.interpolate(track.unsqueeze(1).float(), size=TARGET_RES, mode='nearest').squeeze(1).long()
            assert pred_labels.shape == track.shape, f"Segmentation mismatch: {pred_labels.shape} vs {track.shape}"

            # Depth resizing: force both predicted and ground truth depth to TARGET_RES.
            if pred_depth.ndim == 3:
                pred_depth = pred_depth.unsqueeze(1)
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            pred_depth = F.interpolate(pred_depth, size=TARGET_RES, mode='bilinear', align_corners=False).squeeze(1)
            depth = F.interpolate(depth, size=TARGET_RES, mode='bilinear', align_corners=False).squeeze(1)
            assert pred_depth.shape == depth.shape, f"Depth mismatch: {pred_depth.shape} vs {depth.shape}"

            training_metrics.add(pred_labels, track, pred_depth, depth)

            loss = alpha * ce_loss(pred, track) + beta * mse_loss(pred_depth, depth)
            loss.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                img = batch["image"]
                track = batch["track"]
                depth = batch["depth"]

                pred, pred_depth = model(img)
                # Resize both predictions and ground truth to TARGET_RES.
                pred = F.interpolate(pred, size=TARGET_RES, mode='bilinear', align_corners=False)
                pred_labels = pred.argmax(dim=1)

                if track.ndim == 4:
                    track = track.squeeze(1)
                track = F.interpolate(track.unsqueeze(1).float(), size=TARGET_RES, mode='nearest').squeeze(1).long()
                assert pred_labels.shape == track.shape, f"Validation segmentation mismatch: {pred_labels.shape} vs {track.shape}"

                if pred_depth.ndim == 3:
                    pred_depth = pred_depth.unsqueeze(1)
                if depth.ndim == 3:
                    depth = depth.unsqueeze(1)
                pred_depth = F.interpolate(pred_depth, size=TARGET_RES, mode='bilinear', align_corners=False).squeeze(1)
                depth = F.interpolate(depth, size=TARGET_RES, mode='bilinear', align_corners=False).squeeze(1)
                assert pred_depth.shape == depth.shape, f"Validation depth mismatch: {pred_depth.shape} vs {depth.shape}"

                validation_metrics.add(pred_labels, track, pred_depth, depth)

        computed_validation_metrics = validation_metrics.compute()
        epoch_train_acc = torch.tensor(training_metrics.compute()["accuracy"])
        epoch_val_acc = torch.tensor(computed_validation_metrics["accuracy"])
        iou = torch.tensor(computed_validation_metrics["iou"])
        abs_depth_error = torch.tensor(computed_validation_metrics["abs_depth_error"])
        tp_depth_error = torch.tensor(computed_validation_metrics["tp_depth_error"])

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)
        logger.add_scalar("accuracy", computed_validation_metrics["accuracy"], global_step)
        logger.add_scalar("iou", iou, global_step)
        logger.add_scalar("abs_depth_error", abs_depth_error, global_step)
        logger.add_scalar("tp_depth_error", tp_depth_error, global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d}/{num_epoch:2d} - "
                  f"train_acc={epoch_train_acc:.4f} "
                  f"val_acc={epoch_val_acc:.4f} "
                  f"accuracy={epoch_val_acc:.4f} "
                  f"iou={iou:.4f} "
                  f"abs_depth_error={abs_depth_error:.4f} "
                  f"tp_depth_error={tp_depth_error:.4f}")

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
