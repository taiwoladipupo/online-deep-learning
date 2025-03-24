import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from pathlib import Path
from datetime import datetime
from collections import Counter
from .metrics import ConfusionMatrix
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from torch.optim.lr_scheduler import _LRScheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.total_steps - self.last_epoch) / (self.total_steps - self.warmup_steps) for base_lr in self.base_lrs]

class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=25):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Weight background lower, emphasize small classes
        self.class_weights = torch.tensor([0.1, 2.0, 2.5], dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        self.dice_loss = DiceLoss()
        self.depth_loss = nn.L1Loss()
        self.depth_weight = 0.05

        if device:
            self.to(device)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, target, depth_pred, depth_true):
        target = target.to(logits.device)
        depth_true = depth_true.to(depth_pred.device)

        if depth_true.ndim == 4 and depth_true.shape[1] == 1:
            depth_true = depth_pred.squeeze(1)  # (B, H, W)

        if depth_pred.ndim == 3 :
            depth_pred = depth_true.squeeze(1)  # (B, H, W)

        # Suppress class 0 for first few epochs
        if self.current_epoch < 5:
            mask = target != 0
            if mask.any():

                logits = logits[:, 1:]  # Remove class 0 logits
                ce = self.ce_loss(logits, target.clamp(min=1))
            else:
                ce = self.ce_loss(logits, target)
        else:
            ce = self.ce_loss(logits, target)

        dice = self.dice_loss(logits, target)
        depth = self.depth_loss(depth_pred, depth_true)

        total = 0.7 * ce + 0.3 * dice + self.depth_weight * depth
        return total



def match_shape(pred, target):
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=False)
    return pred

def warmup_scheduler(optimizer, warmup_epochs=3):
    def lr_lambda(epoch):
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(exp_dir="logs", model_name="detector", num_epoch=25, lr=5e-4,
          batch_size=16, seed=2024, transform_pipeline="default", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    loss_func = CombinedLoss(device=device, total_epochs=num_epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = warmup_scheduler(optimizer)

    train_data = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    global_step = 0
    best_miou = 0

    for epoch in range(num_epoch):
        loss_func.set_epoch(epoch)
        model.train()
        train_losses = []

        for batch in train_data:
            img = batch["image"].to(device)
            label = batch["track"].to(device)
            depth_true = batch["depth"].to(device)

            logits, depth_pred = model(img)
            loss = loss_func(logits, label, depth_pred, depth_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            logger.add_scalar("train/total_loss", loss.item(), global_step)
            global_step += 1

        scheduler.step()

        # Validation
        model.eval()
        val_losses, depth_errors = [], []
        confusion_matrix = ConfusionMatrix(num_classes=3)

        with torch.no_grad():
            for batch in val_data:
                img = batch["image"].to(device)
                label = batch["track"].to(device)
                depth_true = batch["depth"].to(device)

                logits, depth_pred = model(img)
                loss = loss_func(logits, label, depth_pred, depth_true)
                val_losses.append(loss.item())

                pred = logits.argmax(dim=1)
                confusion_matrix.add(pred, label)

                depth_errors.append(F.l1_loss(depth_pred, depth_true).item())

            avg_probs = F.softmax(logits, dim=1).mean(dim=(0, 2, 3))
            pred_classes = logits.argmax(dim=1)
            total = pred_classes.numel()
            pred_dist = {int(k): f"{(v/total)*100:.2f}%" for k, v in zip(*torch.unique(pred_classes, return_counts=True))}

            print("Average class probabilities:", avg_probs)
            print("Predicted distribution:", pred_dist)
            print("Pred unique:", torch.unique(pred_classes))
            print("Target unique:", torch.unique(label))

            miou = confusion_matrix.compute()
            mean_depth_mae = np.mean(depth_errors)
            print("miou:", miou["iou"])
            print("mean_depth_mae:", mean_depth_mae)

            for i, iou in enumerate(np.diag(confusion_matrix.matrix) /
                                    (confusion_matrix.matrix.sum(1) + confusion_matrix.matrix.sum(0) - np.diag(confusion_matrix.matrix) + 1e-6)):
                print(f"Class {i} IoU: {iou:.3f}")

            if miou["iou"] > best_miou:
                best_miou = miou["iou"]
                torch.save(model.state_dict(), log_dir / "best_model.th")

            logger.add_scalar("val/miou", miou["iou"], epoch)
            logger.add_scalar("val/seg_accuracy", miou["accuracy"], epoch)
            logger.add_scalar("val/depth_mae", mean_depth_mae, epoch)

            print(f"Epoch {epoch+1:2d}/{num_epoch:2d} - Train loss: {np.mean(train_losses):.4f}, Val loss: {np.mean(val_losses):.4f}")
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
    parser.add_argument("--transform_pipeline", type=str, default="default")

    args = parser.parse_args()
    train(**vars(parser.parse_args()))