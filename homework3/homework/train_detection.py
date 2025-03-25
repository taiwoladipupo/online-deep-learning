import argparse
import numpy as np
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from .metrics import ConfusionMatrix
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovász extension with respect to sorted errors.
    """
    gts = gt_sorted.sum()
    if gts == 0:
        return torch.zeros_like(gt_sorted)
    grad = (gts - gt_sorted.cumsum(0)) / gts
    return grad


def lovasz_softmax_flat(probas, labels):
    """
    Computes the Lovász-Softmax loss for a flat tensor of predictions and labels.

    Args:
        probas: [P, C] prediction probabilities
        labels: [P] ground truth labels
    Returns:
        Scalar loss value.
    """
    losses = []
    C = probas.size(1)
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if losses:
        return torch.mean(torch.stack(losses))
    return torch.tensor(0.0, device=probas.device)


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        """
        Lovász-Softmax loss (simplified implementation).

        Args:
            per_image (bool): whether to compute the loss per image.
            ignore: label to ignore.
        """
        super(LovaszSoftmaxLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        Args:
            logits: tensor of shape (B, C, H, W)
            labels: tensor of shape (B, H, W)
        Returns:
            Scalar loss value.
        """
        probas = torch.softmax(logits, dim=1)
        loss = lovasz_softmax_flat(probas.view(-1, logits.size(1)), labels.view(-1))
        return loss


# --- Combined Loss using Lovász-Softmax for segmentation ---

class LovaszCombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=25, seg_loss_weight=1.0, depth_loss_weight=0.0):
        """
        Combined loss for segmentation and optional depth regression.
        Uses Lovász-Softmax loss for segmentation and L1 loss for depth.

        Args:
            device (torch.device): device to run the loss on.
            total_epochs (int): total training epochs.
            seg_loss_weight (float): weight for segmentation loss.
            depth_loss_weight (float): weight for depth regression loss.
        """
        super(LovaszCombinedLoss, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.seg_loss_weight = seg_loss_weight
        self.depth_loss_weight = depth_loss_weight

        self.seg_loss = LovaszSoftmaxLoss()
        self.depth_loss = nn.L1Loss()

        if device:
            self.to(device)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, target, depth_pred, depth_true):
        device = logits.device
        target = target.to(device)
        depth_true = depth_true.to(device)

        # Ensure logits match target spatial dimensions
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        # Compute Lovász-Softmax loss for segmentation
        seg_loss_val = self.seg_loss(logits, target)

        # Process depth predictions
        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred.squeeze(1)
        if depth_true.ndim == 4 and depth_true.shape[1] == 1:
            depth_true = depth_true.squeeze(1)
        if depth_pred.shape[-2:] != depth_true.shape[-2:]:
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                       align_corners=False).squeeze(1)
        depth_loss_val = self.depth_loss(depth_pred, depth_true)

        return self.seg_loss_weight * seg_loss_val + self.depth_loss_weight * depth_loss_val

def visualize_sample(data_loader, num_samples=1):
    """
    Visualizes one sample (image and segmentation mask) from the provided data loader.
    """
    for i, batch in enumerate(data_loader):
        # Take the first sample in the batch
        image = batch["image"][0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        mask = batch["track"][0].cpu().numpy()  # (H, W)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        # Adjust if your images are not in uint8
        plt.imshow(image.astype(np.uint8))
        plt.subplot(1, 2, 2)
        plt.title("Segmentation Mask")
        plt.imshow(mask, cmap='jet')
        plt.colorbar()
        plt.show()

        if i + 1 >= num_samples:
            break


def print_label_histogram(data_loader):
    """
    Computes and prints the overall label distribution across the dataset.
    """
    all_labels = []
    for batch in data_loader:
        labels = batch["track"].view(-1)
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Overall label distribution:", distribution)

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



def match_shape(pred, target):
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=False)
    return pred

def warmup_scheduler(optimizer, warmup_epochs=3):
    def lr_lambda(epoch):
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_class_weights(data_loader):
    """
    Calculate class weights based on the inverse frequency of each class in the dataset.

    Args:
        data_loader: DataLoader object for the training dataset.

    Returns:
        class_weights: Tensor containing the weight for each class.
    """
    all_labels = []
    for batch in data_loader:
        labels = batch["track"].view(-1)
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    total_counts = np.sum(counts)
    class_weights = total_counts / (len(unique) * counts)
    return torch.tensor(class_weights, dtype=torch.float32)

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import numpy as np

def train(exp_dir="logs", model_name="detector", num_epoch=60, lr=1e-3,
          batch_size=16, seed=2024, transform_pipeline="default", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)

    train_data = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # Calculate class weights
    class_weights = calculate_class_weights(train_data)
    loss_func = LovaszCombinedLoss(
        device=device,
        total_epochs=num_epoch,
        seg_loss_weight=1.0,
        depth_loss_weight=0.0
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    print("Verifying training labels...")
    visualize_sample(train_data, num_samples=1)
    print_label_histogram(train_data)

    global_step = 0
    best_miou = 0
    val_losses = []

    for epoch in range(num_epoch):
        loss_func.set_epoch(epoch)
        model.train()
        train_losses = []

        for batch in train_data:
            img = batch["image"].to(device).float()
            label = batch["track"].to(device)
            depth_true = batch["depth"].to(device)

            logits, depth_pred = model(img)
            if label.ndim == 4 and label.shape[1] == 1:
                label = label.squeeze(1)
            if logits.shape[2:] != label.shape[1:]:
                logits = F.interpolate(logits, size=label.shape[1:], mode='bilinear', align_corners=False)

            # Ensure depth predictions are resized to match depth labels
            if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

            loss = loss_func(logits, label, depth_pred, depth_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            logger.add_scalar("train/total_loss", loss.item(), global_step)
            global_step += 1

        # Validation
        model.eval()
        val_losses, depth_errors = [], []
        confusion_matrix = ConfusionMatrix(num_classes=3)

        with torch.no_grad():
            for batch in val_data:
                img = batch["image"].to(device).float()
                label = batch["track"].to(device)
                depth_true = batch["depth"].to(device)

                logits, depth_pred = model(img)
                if logits.shape[2:] != label.shape[1:]:
                    logits = F.interpolate(logits, size=label.shape[1:], mode='bilinear', align_corners=False)

                # Ensure depth predictions are resized to match depth labels
                if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                    depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

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

        scheduler.step(np.mean(val_losses))

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