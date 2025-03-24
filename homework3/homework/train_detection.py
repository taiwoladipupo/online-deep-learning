import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, device=None, total_epochs=25):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Adjust these weights based on class imbalance stats
        class_weights = torch.tensor([1.0, 3.0, 4.0], dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
        if depth_true.ndim == 4:
            depth_true = depth_true.squeeze(1)

        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        depth = self.depth_loss(depth_pred, depth_true)

        total_loss = 0.7 * ce + 0.3 * dice + self.depth_weight * depth
        return total_loss


def warmup_scheduler(optimizer, warmup_epochs=3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        lr: float = 5e-4,
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
    scheduler = warmup_scheduler(optimizer)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()
        loss_func.set_epoch(epoch)
        loss_func.total_epochs = num_epoch

        pixel_counter = Counter()
        for batch in train_data:

            img = batch['image'].to(device)
            label = batch['track'].to(device)
            depth_true = batch['depth'].to(device)
            # print("Unique target labels in batch:", torch.unique(label))

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
            probs = F.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=(0,2,3))
            print("Average class probabilities:", avg_probs)

        if epoch % 5 == 0 :  # just one batch to reduce clutter
            import torchvision.utils as vutils

            # Normalize masks to [0, 1] and repeat channels to match img_vis
            pred_mask = logits.argmax(dim=1).unsqueeze(1).float() / 2.0  # (B, 1, H, W)
            true_mask = label.unsqueeze(1).float() / 2.0  # (B, 1, H, W)
            pred_mask = pred_mask.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            true_mask = true_mask.repeat(1, 3, 1, 1)  # (B, 3, H, W)

            img_vis = img[:, :3, :, :]  # (B, 3, H, W)

            # Concatenate along the batch dimension
            grid = vutils.make_grid(torch.cat([img_vis, true_mask, pred_mask], dim=0), nrow=batch_size)
            logger.add_image("val/sample_image_pred_gt", grid, global_step=global_step)

        print("Pred unique:", torch.unique(pred))
        print("Target unique:", torch.unique(label))

        miou = confusion_matrix.compute()
        mean_depth_mae = sum(depth_errors) / len(depth_errors)

        best_miou = 0
        best_model = None
        if miou["iou"] > best_miou:
            best_miou = miou["iou"]
            best_model = model

        if best_model:
            torch.save(best_model, log_dir / "best_model.th")

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