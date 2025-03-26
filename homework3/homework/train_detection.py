import argparse
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.tensorboard as tb

from torch import nn

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

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

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # Loss function and optimizer
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    global_step = 0
    training_metrics = DetectionMetric()
    validation_metrics = DetectionMetric()

    # Training loop
    for epoch in range(num_epoch):
        # clear all available metrics
        training_metrics.reset()

        model.train()

        for batch in train_data:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            img = batch["image"]
            track = batch["track"]
            depth = batch["depth"]

            optimizer.zero_grad()
            pred, pred_depth = model(img)


            # Resizing Pred
            if pred.shape[2:] != track.shape[1:]:
                target_size = tuple(int(x) for x in track.shape[1:])
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            pred_labels = pred.argmax(dim=1)

            if track.dim() == 4:
                track = track.squeeze(1)
            #print("Before forced resize: pred_labels:", pred_labels.shape, "track", track.shape)
            if track.shape != pred_labels.shape:
                #print("Before resizing, track shape:", track.shape, "pred labels shape", pred_labels.shape)
                target_size =tuple(int(x) for x in pred_labels.shape[-2:])
                track = F.interpolate(track.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1).long()
                #print("After resizing, track shape:", track.shape)
            assert pred_labels.shape == track.shape


            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            if pred_depth.ndim == 3:
                pred_depth = pred_depth.unsqueeze(1)


            # Check if spatial dimensions differ
            if pred_depth.shape[-2:] != depth.shape[-2:]:
                print("Before interpolation: pred_depth shape =", pred_depth.shape, "depth shape =", depth.shape)
                target_size = tuple(int(x) for x in depth.shape[-2:])  # e.g., (H, W) from ground truth depth
                # Upsample pred_depth to match ground truth depth resolution.
                pred_depth = F.interpolate(pred_depth,
                                           size=target_size,
                                           mode='bilinear',
                                           align_corners=False)
                print("After interpolation: pred_depth shape =", pred_depth.shape)


            # Squeeze them back if necessary
            if pred_depth.ndim == 4:
                pred_depth = pred_depth.squeeze(1)
            if depth.ndim == 4:
                depth = depth.squeeze(1)
            print("Before metric add: pred_depth shape =", pred_depth.shape, "depth shape =", depth.shape)
            assert pred_depth.shape == depth.shape, "Shape mismatch: pred_depth {} vs depth {}".format(pred_depth.shape, depth.shape)

            # # Squeeze them back
            # pred_depth = pred_depth.squeeze(1)
            # depth = depth.squeeze(1)


            logits = torch.nn.functional.one_hot(track, num_classes=3).permute(0, 3, 1,2).float()

            # print({"img": img.shape,
            #        "depth": depth.shape,
            #        "track": track.shape,
            #        "target_indices": target_indices.shape,})

            # Ensure the tensors have compatible dimensions
            # Ensure the tensors have compatible dimensions
            # Ensure the tensors have compatible dimensions
            # if pred_depth.shape[2] != depth.shape[2]:
            #     if depth.dim() == 3:  # If depth has only 3 dimensions, add a dimension
            #         depth = depth.unsqueeze(1)
            #     if pred_depth.dim() == 3:  # If pred_depth has only 3 dimensions, add a dimension
            #         pred_depth = pred_depth.unsqueeze(1)
            #     pred_depth = F.interpolate(pred_depth, size=(depth.shape[2], depth.shape[3]), mode='bilinear',
            #                                align_corners=False)
            training_metrics.add(pred_labels, track,pred_depth, depth)

            loss = alpha * ce_loss(pred, logits) + beta * mse_loss(pred_depth, depth)
            loss.backward()
            optimizer.step()

            global_step += 1

        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                img = batch["image"]
                track = batch["track"]
                depth = batch["depth"]

                pred, pred_depth = model(img)

                if pred.shape[2:] != track.shape[1:]:
                    target_size = tuple((int(x) for x in track.shape[1:]))
                    pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                pred_labels = pred.argmax(dim=1)

                if track.dim() == 4:
                    track = track.squeeze(1)

                if track.shape != pred_labels.shape:
                   #print("Before resizing, track shape:", track.shape, "pred labels shape",pred_labels.shape)
                    target_size =tuple((int(x) for x in pred_labels.shape[-2:]))
                    track =F.interpolate(track.unsqueeze(1).float(), size=target_size, mode='nearest').long()
                    #print("After resizing, track shape:", track.shape)

                assert pred_labels.shape == track.shape
                # Resizing depth

                if depth.ndim == 3:
                    depth = depth.unsqueeze(1)
                if pred_depth.ndim == 3:
                    pred_depth = pred_depth.unsqueeze(1)


                # Check if spatial dimensions differ
                if pred_depth.shape[-2:] != depth.shape[-2:]:
                    target_size = tuple(depth.shape[-2:])  # e.g., (H, W) from ground truth depth
                    # Upsample pred_depth to match ground truth depth resolution.
                    pred_depth = F.interpolate(pred_depth,
                                               size=target_size,
                                               mode='bilinear',
                                               align_corners=False)
                # Squeeze them back if necessary
                if pred_depth.ndim == 4:
                    pred_depth = pred_depth.squeeze(1)
                if depth.ndim == 4:
                    depth = depth.squeeze(1)
                assert pred_depth.shape == depth.shape

                # pred_labels = pred.argmax(dim=1)
                validation_metrics.add(pred_labels, track, pred_depth, depth)

        # log accuracy to tensorboard
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

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"accuracy={epoch_val_acc:.4f} "
                f"iou={iou:.4f} "
                f"abs_depth_error={abs_depth_error:.4f} "
                f"tp_depth_error={tp_depth_error:.4f}"
            )

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