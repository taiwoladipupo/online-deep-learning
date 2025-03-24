import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.cuda import device

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import ConfusionMatrix

class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=25):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0

        counts = torch.tensor([0.1, 2.0, 3.0], dtype=torch.float32)
        frequency = counts / counts.sum()
        class_weights = torch.log(1 / (frequency + 1e-6))
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        self.class_weights = class_weights.to(device)

        self.seg_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.depth_loss = nn.L1Loss()
        self.tversky_loss = TverskyLoss()
        self.dice_loss = DiceLoss()
        self.depth_weight = 0.05

        self.total_epochs = total_epochs
        self.current_epoch = 0

        if device:
            self.to(device)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, target, depth_pred, depth_true):
        assert logits.ndim == 4 and logits.shape[1] == 3, f"Expected shape (B, 3, H, W), got {logits.shape}"
        assert target.ndim == 3, f"Expected shape (B, H, W), got {target.shape}"

        target = target.to(logits.device)
        depth_true = depth_true.to(depth_pred.device)
        # Suppression for background
        suppression = max(0.0, 1.5 - (self.current_epoch / self.total_epochs))
        logits[:, 0, :, :] -= suppression * 0.4

        # Encourage foreground
        boost = max(0.0, 1.0 - self.current_epoch / (self.total_epochs / 2))
        logits[:, 1, :, :] += boost * 2.0
        logits[:, 2, :, :] += boost * 2.0

        # Dynamic CE class weights
        weights = torch.tensor([1.0, 4.0, 6.0], device=logits.device)
        weights[0] = max(0.1, 1.0 - self.current_epoch / self.total_epochs)
        seg_loss_fn = nn.CrossEntropyLoss(weight=weights)

        seg_loss = seg_loss_fn(logits, target) # using dynamic CE
        dice = self.dice_loss(logits, target)
        depth = self.depth_loss(depth_pred, depth_true)

        total = seg_loss + 0.7 * dice + self.depth_weight * depth

        return total

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

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        TP = (probs * target_one_hot).sum(dim=(0, 2, 3))
        FP = (probs * (1 - target_one_hot)).sum(dim=(0, 2, 3))
        FN = ((1 - probs) * target_one_hot).sum(dim=(0, 2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()

def warmup_schedulerr(optimizer, warmup_epochs=3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 16,
        seed: int = 2024,
        transform_pipeline: str = "default",
        **kwargs,
):
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

    train_data = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    loss_func = CombinedLoss(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = warmup_schedulerr(optimizer)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        loss_func.set_epoch(epoch)

        pixel_counter = Counter()
        for batch in train_data:
            img = batch['image'].to(device)
            label = batch['track'].to(device)
            depth_true = batch['depth'].to(device)

            unique, counts = torch.unique(label, return_counts=True)
            for u, c in zip(unique.tolist(), counts.tolist()):
                pixel_counter[u] += c

            logits, depth_pred = model(img)
            loss_val = loss_func(logits, label, depth_pred, depth_true)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            metrics["train_acc"].append(loss_val.item())
            global_step += 1
            logger.add_scalar("train/total_loss", loss_val.item(), global_step)

        confusion_matrix = ConfusionMatrix(num_classes=3)
        depth_errors = []

        with torch.inference_mode():
            model.eval()
            for batch in val_data:
                img = batch['image'].to(device)
                label = batch['track'].to(device)
                depth_true = batch['depth'].to(device)

                logits, depth_pred = model(img)
                val_loss = loss_func(logits, label, depth_pred, depth_true)
                metrics["val_acc"].append(val_loss.item())

                pred = logits.argmax(dim=1)
                confusion_matrix.add(pred, label)

                depth_errors.append(torch.abs(depth_pred - depth_true).mean().item())

        miou = confusion_matrix.compute()
        mean_depth_mae = sum(depth_errors) / len(depth_errors)

        print("miou:", miou["iou"])
        print("mean_depth_mae:", mean_depth_mae)


        if hasattr(confusion_matrix, "matrix"):
            matrix = confusion_matrix.matrix
            tp = np.diag(matrix)
            fp = matrix.sum(axis=0) - tp
            fn = matrix.sum(axis=1) - tp
            denom = np.add(tp, np.add(fp, fn))

            iou_per_class = np.divide(tp, denom, out=np.zeros_like(tp, dtype=np.float32), where=denom != 0)

            for i, class_iou in enumerate(iou_per_class):
                print(f"Class {i} IoU: {class_iou:.3f}")
        else:
            print("Confusion matrix data not available.")

        confusion_matrix.reset()
        print(f"mIou: {miou}")
        logger.add_scalar("val/miou", miou["iou"], epoch)
        logger.add_scalar("val/seg_accuracy", miou["accuracy"], epoch)

        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train/accuracy', epoch_train_acc, epoch)
        logger.add_scalar('val/accuracy', epoch_val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
        scheduler.step()

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

    train(**vars(parser.parse_args()))