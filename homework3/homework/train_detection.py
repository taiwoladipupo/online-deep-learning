import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

# Local modules
from .metrics import ConfusionMatrix
from .models import load_model, save_model
from .datasets.road_dataset import load_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=100, seg_loss_weight=1.0, depth_loss_weight=0.0, ce_weight=0.6, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.seg_loss_weight = seg_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ce_weight = ce_weight

        self.lovasz_loss = LovaszSoftmaxLoss()
        self.focal_loss = FocalLoss(logits=True)
        self.dice_loss = DiceLoss()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 20.0, 20.0], dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.depth_loss = nn.L1Loss()

        if device:
            self.to(device)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, target, depth_pred, depth_true):
        device = logits.device
        target = target.to(device)
        depth_true = depth_true.to(device)

        if logits.shape[2:] != target.shape[1:]:
            logits = nn.functional.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        lovasz_loss_val = self.lovasz_loss(logits, target)
        ce_loss_val = self.ce_loss(logits, target)
        focal_loss_val = self.focal_loss(logits, target)
        dice_loss_val = self.dice_loss(logits, target)

        seg_loss_val = (1 - self.ce_weight) * (lovasz_loss_val + focal_loss_val + dice_loss_val) + self.ce_weight * ce_loss_val

        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred.squeeze(1)
        if depth_true.ndim == 4 and depth_true.shape[1] == 1:
            depth_true = depth_true.squeeze(1)
        if depth_pred.shape[-2:] != depth_true.shape[-2:]:
            depth_pred = nn.functional.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        depth_loss_val = self.depth_loss(depth_pred, depth_true)

        return self.seg_loss_weight * seg_loss_val + self.depth_loss_weight * depth_loss_val

def compute_sample_weights(dataset):
    weights = []
    for i in range(len(dataset)):
        mask = torch.tensor(dataset[i]["track"]).float()
        total_pixels = mask.numel()
        rare_pixels = ((mask == 1) | (mask == 2)).float().sum().item()
        rare_ratio = rare_pixels / total_pixels
        weight = 1.0 / (rare_ratio + 1e-6)
        weights.append(weight)
    return weights

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self):
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] Variable, logits at each pixel (between -\infty and +\infty)
            labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        """
        logits = logits.permute(0, 2, 3, 1).contiguous()
        labels = labels.view(-1)
        logits = logits.view(-1, logits.size(-1))
        return self.lovasz_softmax_flat(logits, labels)

    def lovasz_softmax_flat(self, logits, labels):
        """
        Multi-class Lovasz-Softmax loss
        Args:
            logits: [P, C] Variable, logits at each prediction (between -\infty and +\infty)
            labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        """
        C = logits.size(1)
        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - logits[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)))
        return sum(losses) / len(losses)

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if gt_sorted.numel() > 1:  # cover case when gt_sorted is empty
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
# Data Augmentation Transforms
def get_train_transforms():
    return T.Compose([
        T.Resize((192, 256)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomCrop((192, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize((192, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

# Visualization and Histogram
def visualize_sample(data_loader, num_samples=1):
    for i, batch in enumerate(data_loader):
        image = batch["image"][0].permute(1, 2, 0).cpu().numpy()
        mask = batch["track"][0].cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image.astype(np.uint8))
        plt.subplot(1, 2, 2)
        plt.title("Segmentation Mask")
        plt.imshow(mask, cmap='jet')
        plt.colorbar()
        plt.show()
        if i + 1 >= num_samples:
            break

def print_label_histogram(data_loader):
    all_labels = []
    for batch in data_loader:
        labels = batch["track"].view(-1)
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Overall label distribution:", distribution)

# Warmup Scheduler and Helpers
class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.total_steps - self.last_epoch) / (self.total_steps - self.warmup_steps) for base_lr
                    in self.base_lrs]

def warmup_scheduler_fn(optimizer, warmup_epochs=3):
    def lr_lambda(epoch):
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def match_shape(pred, target):
    if pred.shape[2:] != target.shape[1:]:
        pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=False)
    return pred

def calculate_class_weights(data_loader):
    all_labels = []
    for batch in data_loader:
        labels = batch["track"].view(-1)
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    freq = counts / np.sum(counts)
    median_freq = np.median(freq)
    class_weights = median_freq / freq
    weight_dict = {cls: w for cls, w in zip(unique, class_weights)}
    return torch.tensor([weight_dict.get(i, 1.0) for i in range(3)], dtype=torch.float32)

# Prediction Function for Inference
def predict(model, image_path, device):
    transform = T.Compose([
        T.Resize((192, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        seg_logits, raw_depth = model(input_tensor)
        seg_logits = F.interpolate(seg_logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        pred_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()
        depth = raw_depth  # Already normalized via Sigmoid
        depth = F.interpolate(depth.unsqueeze(1), size=(96, 64), mode='bilinear', align_corners=False).squeeze(1)
        depth_pred = depth[0].cpu().numpy()
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(np.array(image))
    plt.subplot(1, 3, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred_mask, cmap='jet')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Predicted Depth")
    plt.imshow(depth_pred, cmap='gray')
    plt.colorbar()
    plt.show()
    return pred_mask, depth_pred

# Main Training Loop with WeightedRandomSampler and Extended Training
def train(exp_dir="logs", model_name="detector", num_epoch=100, lr=1e-4,  # lowered lr to 1e-4
          batch_size=16, seed=2024, transform_pipeline="default", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load training dataset as a Dataset (return_dataloader=False) to compute sample weights.
    train_dataset = load_data("drive_data/train", transform_pipeline="aug",
                              return_dataloader=False, shuffle=False, batch_size=1, num_workers=2)
    sample_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_data = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    # Load validation data normally.
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    print("Verifying training labels...")
    visualize_sample(train_data, num_samples=1)
    print_label_histogram(train_data)

    model = load_model(model_name, **kwargs).to(device)

    # Calculate class weights for the loss function.
    class_weights = torch.tensor([1.0, 20.0, 20.0], dtype=torch.float32).to(device)
    loss_func = CombinedLoss(
        device=device,
        total_epochs=num_epoch,
        seg_loss_weight=1.0,
        depth_loss_weight=0.0,
        ce_weight=0.6,  # increased weight on cross-entropy component
        class_weights=class_weights
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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
            if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                           align_corners=False).squeeze(1)

            loss = loss_func(logits, label, depth_pred, depth_true)

            optimizer.zero_grad()
            loss.backward()
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                    depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                               align_corners=False).squeeze(1)

                loss = loss_func(logits, label, depth_pred, depth_true)
                val_losses.append(loss.item())

                pred = logits.argmax(dim=1)
                confusion_matrix.add(pred, label)
                depth_errors.append(F.l1_loss(depth_pred, depth_true).item())

            avg_probs = F.softmax(logits, dim=1).mean(dim=(0, 2, 3))
            pred_classes = logits.argmax(dim=1)
            total = pred_classes.numel()
            pred_dist = {int(k): f"{(v / total) * 100:.2f}%" for k, v in
                         zip(*torch.unique(pred_classes, return_counts=True))}

            print(f"\nEpoch {epoch + 1}/{num_epoch}")
            print("Average class probabilities:", avg_probs)
            print("Predicted distribution:", pred_dist)
            print("Pred unique:", torch.unique(pred_classes))
            print("Target unique:", torch.unique(label))

            miou = confusion_matrix.compute()
            mean_depth_mae = np.mean(depth_errors)
            print("mIoU:", miou["iou"])
            print("mean_depth_mae:", mean_depth_mae)
            for i, iou in enumerate(np.diag(confusion_matrix.matrix) /
                                    (confusion_matrix.matrix.sum(1) + confusion_matrix.matrix.sum(0) - np.diag(
                                        confusion_matrix.matrix) + 1e-6)):
                print(f"Class {i} IoU: {iou:.3f}")

            if miou["iou"] > best_miou:
                best_miou = miou["iou"]
                torch.save(model.state_dict(), log_dir / "best_model.th")
                print(">>> Best model updated <<<")

            logger.add_scalar("val/miou", miou["iou"], epoch)
            logger.add_scalar("val/seg_accuracy", miou["accuracy"], epoch)
            logger.add_scalar("val/depth_mae", mean_depth_mae, epoch)

            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d} - Train loss: {np.mean(train_losses):.4f}, Val loss: {np.mean(val_losses):.4f}")

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