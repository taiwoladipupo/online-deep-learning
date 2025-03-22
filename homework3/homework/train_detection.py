import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch import nn

from .models import  load_model, save_model
# couldn't find the prescribe datasets in read me, so I will use the following datasets
from .datasets.road_dataset import load_data
# from .utils import load_data
from .metrics import ConfusionMatrix


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.l1_loss = nn.L1Loss()
        self.dice_loss = DiceLoss()
        self.depth_weight = 0.3 # weight for depth loss
        self.dice_weight = 1.0 # weight for dice loss

    def forward(self, logits: torch.Tensor, target: torch.LongTensor, depth_pred, depth_true) -> torch.Tensor:
        """
        Combined loss function for segmentation and depth prediction
        Args:
            logits: tensor (b, c, h,w) logits, where c is the number of classes
            target: tensor (b,h,w) labels
            depth_pred: tensor (b,h,w) predicted depth
            depth_true: tensor (b,h,w) true depth

        Returns:
            tensor, segmantation loss + regression loss

        """
        segmentation_loss = self.ce_loss(logits, target)
        depth_loss = self.l1_loss(depth_pred, depth_true)
        dice_loss = self.dice_loss(logits, target)
        return segmentation_loss + self.dice_weight * dice_loss + self.depth_weight * depth_loss

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
        return 1 - dice.mean()



def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
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


    train_data = load_data("drive_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size,
                           num_workers=2)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False)

    # create loss function and optimizer
    loss_func = CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            # print(batch.keys())
            img = batch['image'].to(device)
            label = batch['track'].to(device)
            depth_true = batch['depth'].to(device)

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
                print(f"  img shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")
                print(f"  logits shape: {logits.shape}, min: {logits.min().item()}, max: {logits.max().item()}")
                print(f"  depth_pred shape: {depth_pred.shape}, min: {depth_pred.min().item()}, max: {depth_pred.max().item()}")
            # Initialize confusion matrix
            confusion_matrix = ConfusionMatrix(num_classes=3)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

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
                confusion_matrix.update(pred, label)
        # calculate mIou
        miou = confusion_matrix.compute_mean_iou()
        print(f"miou: {miou}")

        logger.add_scalar("val/miou", miou, epoch)

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
        #scheduler.step()

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

    # optional: additional model hyperparameters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))