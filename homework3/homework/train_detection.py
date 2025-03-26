import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
import numpy as np
import torch.utils.tensorboard as tb
import torchvision.transforms as T
from PIL import Image  # For specifying interpolation in transforms

# Local modules
from .metrics import ConfusionMatrix
from .models import load_model, save_model
from .datasets.road_dataset import load_data


#############################################
# Loss Functions
#############################################
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()
        tversky_index = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky_index


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()
        tversky_index = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        return focal_tversky_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=1.5, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs shape: [N, C, H, W]
        if self.logits:
            if targets.ndim == inputs.ndim:
                if targets.shape[1] != 1:
                    if targets.size(-1) != 1 and torch.all(targets == targets[..., 0:1].expand_as(targets)):
                        targets = targets[..., 0]
                    else:
                        raise ValueError(f"Unexpected target shape: {targets.shape}. "
                                         "Expected [N, H, W] or [N, 1, H, W] when logits=True.")
            if targets.ndim == 3:
                targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
            elif targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
                targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
            else:
                raise ValueError(f"Unexpected target shape: {targets.shape}. "
                                 "Expected [N, H, W] or [N, 1, H, W] when logits=True.")
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=100, seg_loss_weight=1.0, depth_loss_weight=0.0,
                 ce_weight=1.0, dice_weight=0.0, tversky_weight=0.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.seg_loss_weight = seg_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

        self.focal_loss = FocalLoss(alpha=0.7, gamma=1.5, logits=True)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)

        if class_weights is None:
            class_weights = torch.tensor([1.0, 50.0, 30.0], dtype=torch.float32)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if len(class_weights) != 3:
            raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes (3).")
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

        # Use raw logits divided by temperature (do not apply softmax before loss)
        temperature = 1.5
        scaled_logits = logits / temperature

        if scaled_logits.shape[2:] != target.shape[1:]:
            scaled_logits = F.interpolate(scaled_logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        ce_loss_val = self.ce_loss(scaled_logits, target)
        focal_loss_val = self.focal_loss(scaled_logits, target)
        seg_loss_val = self.ce_weight * ce_loss_val + (1 - self.ce_weight) * focal_loss_val

        if self.dice_weight > 0:
            one_hot_target = F.one_hot(target, num_classes=scaled_logits.size(1)).permute(0, 3, 1, 2).float()
            dice_loss_val = self.dice_loss(scaled_logits, one_hot_target)
            seg_loss_val = seg_loss_val + self.dice_weight * dice_loss_val

        if self.tversky_weight > 0:
            one_hot_target = F.one_hot(target, num_classes=scaled_logits.size(1)).permute(0, 3, 1, 2).float()
            tversky_loss_val = self.tversky_loss(scaled_logits, one_hot_target)
            seg_loss_val = seg_loss_val + self.tversky_weight * tversky_loss_val

        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred.squeeze(1)
        if depth_true.ndim == 4 and depth_true.shape[1] == 1:
            depth_true = depth_true.squeeze(1)
        if depth_pred.shape[-2:] != depth_true.shape[-2:]:
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                       align_corners=False).squeeze(1)
        depth_loss_val = self.depth_loss(depth_pred, depth_true)

        return self.seg_loss_weight * seg_loss_val + self.depth_loss_weight * depth_loss_val




def get_val_transforms():
    return T.Compose([
        T.Resize((96, 128), interpolation=Image.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



#############################################
# Sample Weights
#############################################
def compute_sample_weights(dataset):
    sample_weights = []
    epsilon = 1e-6
    for sample in dataset:
        mask = sample['track']  # Assume shape (H, W) with integer labels {0,1,2}
        total_pixels = mask.size
        rare_pixels = np.sum((mask == 1) | (mask == 2))
        rare_ratio = rare_pixels / total_pixels
        weight = 1.0 / (rare_ratio + epsilon)
        sample_weights.append(weight)
    return sample_weights


#############################################
# Main Training Loop
#############################################
def train(exp_dir="logs", model_name="detector", num_epoch=100, lr=1e-4,
          batch_size=64, seed=2024, transform_pipeline="default", oversample=False, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Use the custom transform for training if specified

    train_dataset = load_data("drive_data/train", transform_pipeline="aug",
                              return_dataloader=False, shuffle=False, batch_size=1, num_workers=2)
    sample_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_data = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    val_dataset = load_data("drive_data/val", transform_pipeline="default", shuffle=False)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, **kwargs).to(device)

    class_weights = torch.tensor([1.0, 100.0, 90.0], dtype=torch.float32).to(device)
    print("Calculated class weights:", class_weights)

    loss_func = CombinedLoss(
        device=device,
        total_epochs=num_epoch,
        seg_loss_weight=1.0,
        depth_loss_weight=0.0,
        ce_weight=0.4,
        dice_weight=0.3,
        tversky_weight=0.3,
        class_weights=class_weights,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    global_step = 0
    best_miou = 0

    for epoch in range(num_epoch):
        loss_func.set_epoch(epoch)
        model.train()
        epoch_train_losses = []
        for batch in train_data:
            img = batch["image"].to(device).float()
            label = batch["track"].to(device).long()
            depth_true = batch["depth"].to(device)

            # Ensure label is in [N, H, W]
            if label.ndim == 4:
                redundant = True
                for i in range(1, label.shape[-1]):
                    if not torch.equal(label[..., 0], label[..., i]):
                        redundant = False
                        break
                if redundant:
                    label = label[..., 0]
                else:
                    raise ValueError(f"Label has unexpected shape {label.shape}.")

            # Get raw logits from the model
            logits, depth_pred = model(img)
            # Apply temperature scaling to raw logits (do not apply softmax)
            temperature = 1.5
            scaled_logits = logits / temperature
            if scaled_logits.shape[2:] != label.shape[1:]:
                scaled_logits = F.interpolate(scaled_logits, size=label.shape[1:], mode='bilinear', align_corners=False)
            if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                           align_corners=False).squeeze(1)

            loss = loss_func(scaled_logits, label, depth_pred, depth_true)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_train_losses.append(loss.item())
            logger.add_scalar("train/total_loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        epoch_val_losses = []
        depth_errors = []
        confusion_matrix = ConfusionMatrix(num_classes=3)
        with torch.no_grad():
            for batch in val_data:
                img = batch["image"].to(device).float()
                label = batch["track"].to(device).long()
                depth_true = batch["depth"].to(device)
                if label.ndim == 4:
                    redundant = True
                    for i in range(1, label.shape[-1]):
                        if not torch.equal(label[..., 0], label[..., i]):
                            redundant = False
                            break
                    if redundant:
                        label = label[..., 0]
                    else:
                        raise ValueError(f"Label has unexpected shape {label.shape}.")

                logits, depth_pred = model(img)
                scaled_logits = logits / temperature
                if scaled_logits.shape[2:] != label.shape[1:]:
                    scaled_logits = F.interpolate(scaled_logits, size=label.shape[1:], mode='bilinear',
                                                  align_corners=False)
                if depth_pred.shape[-2:] != depth_true.shape[-2:]:
                    depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear',
                                               align_corners=False).squeeze(1)

                loss = loss_func(scaled_logits, label, depth_pred, depth_true)
                epoch_val_losses.append(loss.item())
                pred = scaled_logits.argmax(dim=1)
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
                f"Epoch {epoch + 1:2d}/{num_epoch:2d} - Train loss: {np.mean(epoch_train_losses):.4f}, Val loss: {np.mean(epoch_val_losses):.4f}")
        scheduler.step(np.mean(epoch_val_losses))
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default")
    parser.add_argument("--oversample", action="store_true", help="Enable oversampling of minority classes")
    args = parser.parse_args()
    train(**vars(args))
