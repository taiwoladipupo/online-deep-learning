import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.cuda import device

from .models import  load_model, save_model
# couldn't find the prescribe datasets in read me, so I will use the following datasets
from .datasets.road_dataset import load_data
# from .utils import load_data
from .metrics import ConfusionMatrix


class CombinedLoss(nn.Module):
    def __init__(self, device=None):
        super(CombinedLoss, self).__init__()
        counts = torch.tensor([0.001, 2.0, 3.0], dtype=torch.float32)
        frequency = counts / counts.sum()
        class_weights = torch.log(1 / (frequency + 1e-6))
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)

        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.l1_loss = nn.L1Loss()
        self.tversky_loss = TverskyLoss(alpha=0.5, beta=0.7, smooth=1e-6)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=torch.tensor([0.05, 1.5, 3.0], device=device), gamma=2)

        self.depth_weight = 0.05

        if device:
            self.to(device)

    def forward(self, logits: torch.Tensor, target: torch.LongTensor, depth_pred, depth_true) -> torch.Tensor:
        # logits[:, 0, :, :] -= 0.5
        if self.current_epoch < 5:
            suppression = 1.5 -0.2 * self.current_epoch
        elif self.current_epoch < 10:
            suppression = 1.5 - 0.1 * self.current_epoch
        else:
            suppression = 0.0
        logits[:, 0, :, :] -= suppression

        with torch.no_grad():
            pred = torch.softmax(logits, dim=1)
            bg_ratio = (pred == 0).float().mean()
        penalty = torch.tensor(0.0, device=logits.device)
        if bg_ratio > 0.99:
            penalty = (bg_ratio - 0.99) * 10.0
        segmentation_loss = 0.7 * self.focal_loss(logits * 2.0, target) + 0.3 * self.dice_loss(logits, target)
        depth_loss = self.l1_loss(depth_pred, depth_true)
        tversky_loss = self.tversky_loss(logits, target)
        dice_loss = self.dice_loss(logits, target)
        # Penalty for overconfident background predictions
        # probs = torch.softmax(logits, dim=1)
        # background_conf = probs[:, 0, :, :].mean()

        return segmentation_loss  + self.depth_weight * depth_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)           # (B, C, H, W)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[1])  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()                        # (B, C, H, W)

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice[1:].mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = nn.CrossEntropyLoss(weight=None, reduction='none')(logits, target)
        pt = torch.exp(-ce_loss)

        #Applying class weights per-pixel
        alpha_factor = self.alpha[target]

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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


def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
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

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()



    # train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    # val_data = load_data("classification_data/val", shuffle=False)


    train_data = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size,
                           num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # create loss function and optimizer
    loss_func = CombinedLoss(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()
        loss_func.current_epoch = epoch

        pixel_counter = Counter()
        for batch in train_data:
            # print(batch.keys())
            img = batch['image'].to(device)
            label = batch['track'].to(device)
            depth_true = batch['depth'].to(device)
            #print("labels:",torch.unique(label))
            unique, counts = torch.unique(label, return_counts=True)
            for u,c in zip(unique.tolist(), counts.tolist()):
                pixel_counter[u] += c

            # Training step
            logits, depth_pred = model(img) #  the model returns logits and depth
            loss_val = loss_func(logits, label, depth_pred, depth_true)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # calculate training accuracy
            # pred = torch.argmax(pred, dim=1)
            # train_acc = (pred == label).float().mean().item()
            metrics["train_acc"].append(loss_val.item())

            global_step += 1
            logger.add_scalar("train/total_loss", loss_val.item(), global_step)

            # Print shapes min/ma values
            if global_step % 10 == 0:
                print(f"Step {global_step}:")
                # print(f"  img shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")
                # print(f"  logits shape: {logits.shape}, min: {logits.min().item()}, max: {logits.max().item()}")
                # print(
                #     f"  depth_pred shape: {depth_pred.shape}, min: {depth_pred.min().item()}, max: {depth_pred.max().item()}")
                # pred = logits.argmax(dim=1)
                # unique_classes = torch.unique(pred)
                # print(f"  Predicted classes in batch: {unique_classes.tolist()}")
                # print(f"  Class counts: {torch.bincount(pred.view(-1)).cpu().numpy()}")
            #print(f"  Pixel counts: {pixel_counter}")
            # Initialize confusion matrix
        confusion_matrix = ConfusionMatrix(num_classes=3)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            pred_classes = logits.argmax(dim=1)
            class_counts = torch.bincount(pred_classes.view(-1), minlength=3)
            print(f"Predicted Class counts: {class_counts}")

            for batch in val_data:
                img = batch['image'].to(device)
                label = batch['track'].to(device)
                depth_true = batch['depth'].to(device)

                # compute validation accuracy
                logits, depth_pred = model(img)
                val_loss = loss_func(logits, label, depth_pred, depth_true)
                metrics["val_acc"].append(val_loss.item())

                # confusion matrix
                pred = logits.argmax(dim=1)
                confusion_matrix.add(pred, label)
        # calculate mIou
        miou = confusion_matrix.compute()

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
        #
        #
        # #understaanding which class is affecting iou
        # if isinstance(miou['iou'], list):
        #
        #     # print(f"miou: {miou['iou']: .3f}")
        #     for i, class_iou in enumerate(miou['iou']):
        #         print(f"Class {i} IoU: {class_iou:.3f}")
        # else:
        #     print(f"miou: {miou['iou']: .3f}")
        confusion_matrix.reset()
        print(f"mIou: {miou}")
        logger.add_scalar("val/miou", miou["iou"], epoch)
        logger.add_scalar("val/seg_accuracy", miou["accuracy"], epoch)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train/accuracy', epoch_train_acc, epoch)
        logger.add_scalar('val/accuracy', epoch_val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
        scheduler.step()

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
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

    # optional: additional model hyperparameters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))