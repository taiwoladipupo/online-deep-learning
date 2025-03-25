import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
import numpy as np
import torch.utils.tensorboard as tb
import torchvision.transforms as T

# Local modules
from .metrics import ConfusionMatrix
from .models import load_model, save_model
from .datasets.road_dataset import load_data

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
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=100, seg_loss_weight=1.0, depth_loss_weight=0.0, ce_weight=1.0, dice_weight=0.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.seg_loss_weight = seg_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss(logits=True)
        self.dice_loss = DiceLoss()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 150.0, 200.0], dtype=torch.float32)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
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
            logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        ce_loss_val = self.ce_loss(logits, target)
        seg_loss_val = self.ce_weight * ce_loss_val

        if self.dice_weight > 0:
            one_hot_target = F.one_hot(target, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
            dice_loss_val = self.dice_loss(logits, one_hot_target)
            seg_loss_val = seg_loss_val + self.dice_weight * dice_loss_val

        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred.squeeze(1)
        if depth_true.ndim == 4 and depth_true.shape[1] == 1:
            depth_true = depth_true.squeeze(1)
        if depth_pred.shape[-2:] != depth_true.shape[-2:]:
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=depth_true.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        depth_loss_val = self.depth_loss(depth_pred, depth_true)

        return self.seg_loss_weight * seg_loss_val + self.depth_loss_weight * depth_loss_val

# Data Augmentation Transforms
def get_train_transforms():
    return T.Compose([
        T.Resize((96, 128)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop((96, 128), scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize((96, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def compute_sample_weights(dataset):
    # Flatten the 'track' arrays and convert to integers
    flattened_tracks = [int(item) for sample in dataset for item in sample['track'].flatten()]

    # Compute class counts
    class_counts = np.bincount(flattened_tracks)
    total_samples = len(flattened_tracks)

    # Compute class weights
    class_weights = total_samples / (len(class_counts) * class_counts)

    # Compute sample weights
    sample_weights = [class_weights[int(item)] for sample in dataset for item in sample['track'].flatten()]

    return sample_weights
# Main Training Loop
def train(exp_dir="logs", model_name="detector", num_epoch=100, lr=1e-4,
          batch_size=64, seed=2024, transform_pipeline="default", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    train_dataset = load_data("drive_data/train", transform_pipeline="aug",
                              return_dataloader=False, shuffle=False, batch_size=1, num_workers=2)

    # Debugging: Print track values for a few samples from the training dataset
    print("Training dataset track values:")
    for i in range(5):
        print(f"Sample {i} track value: {train_dataset[i]['track']}")

    sample_weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_data = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    print(f"Loaded {len(train_dataset)} training samples.")
    print(
        f"Sample data shape: {train_dataset[0]['image'].shape}, {train_dataset[0]['track'].shape}, {train_dataset[0]['depth'].shape}")

    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # Debugging: Print track values for a few samples from the validation dataset
    print("Validation dataset track values:")
    for i in range(5):
        print(f"Sample {i} track value: {val_data[i]['track']}")

    print(f"Loaded {len(val_data)} validation samples.")
    print(
        f"Sample data shape: {val_data[0]['image'].shape}, {val_data[0]['track'].shape}, {val_data[0]['depth'].shape}")

    model = load_model(model_name, **kwargs).to(device)

    class_weights = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32).to(device)
    print("Calculated class weights:", class_weights)

    loss_func = CombinedLoss(
        device=device,
        total_epochs=num_epoch,
        seg_loss_weight=1.0,
        depth_loss_weight=0.0,
        ce_weight=0.9,
        dice_weight=0.1,
        class_weights=class_weights
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

            logits, depth_pred = model(img)
            normalized_logits = F.softmax(logits, dim=1)

            if label.ndim == 4 and label.shape[1] == 1:
                label = label.squeeze(1)

            temperature = 1.5
            scaled_logits = normalized_logits / temperature
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

                logits, depth_pred = model(img)
                normalized_logits = F.softmax(logits, dim=1)
                temperature = 1.5
                scaled_logits = normalized_logits / temperature
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
    args = parser.parse_args()
    train(**vars(args))