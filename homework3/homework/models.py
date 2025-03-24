from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Detector(nn.Module):
    def __init__(self, num_classes=3):
        """
        Detector model using MobileNetV3-Small as the encoder.
        Assumes input images are (B, 3, 192, 256).
        Outputs:
         - Segmentation logits: (B, num_classes, 96, 128)
         - Depth map: (B, 96, 128)
        """
        super(Detector, self).__init__()
        # Load MobileNetV3-Small pretrained on ImageNet
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder = mobilenet.features
        # For an input of (192,256), the encoder output is roughly (B, 576, 6, 8)

        # Decoder: Upsample gradually from (B,576,6,8) to (B,16,96,128)
        self.decoder = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=3, padding=1),  # (B,128,6,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,128,12,16)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,64,24,32)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,32,48,64)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,16,96,128)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Segmentation head: outputs logits for 3 classes.
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        # Depth head: outputs a single-channel depth map.
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 192, 256)
        features = self.encoder(x)  # (B,576, approx 6,8)
        up = self.decoder(features)  # (B,16,96,128)
        seg_logits = self.seg_head(up)  # (B, num_classes, 96, 128)
        depth = self.depth_head(up)  # (B,1,96,128)
        depth = depth.squeeze(1)  # (B,96,128)
        return seg_logits, depth

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class RandomChannelDropout(nn.Module):
    def __init__(self, channels=0.2):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        if not self.training or self.channels <= 0:
            return x

        keep_prob = torch.rand(x.shape[1], device=x.device) > self.channels
        mask = keep_prob.float()[None, :, None, None]

        return x * mask



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Detector(nn.Module):
    def __init__(self, num_classes=3):
        """
        Detector model using MobileNetV2 as the encoder.
        Assumes input images are (B, 3, 192, 256).
        Outputs:
         - Segmentation logits: (B, num_classes, 96, 128)
         - Depth map: (B, 96, 128)
        """
        super(Detector, self).__init__()
        # Load pretrained MobileNetV2 and use its feature extractor as encoder.
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = mobilenet.features  # This produces output of shape (B, 1280, H/32, W/32)
        # For input (192,256), encoder output will be roughly (B,1280,6,8)

        # Decoder: Upsample gradually from (B,1280,6,8) to (B,32,96,128)
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,256,12,16)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,128,24,32)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,64,48,64)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B,32,96,128)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Segmentation head: output logits for each class.
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        # Depth head: output a single-channel depth map.
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 192, 256)
        features = self.encoder(x)  # (B,1280, ~6, ~8)
        up = self.decoder(features)  # (B,32,96,128)
        seg_logits = self.seg_head(up)  # (B, num_classes, 96, 128)
        depth = self.depth_head(up)  # (B, 1, 96, 128)
        depth = depth.squeeze(1)  # (B, 96, 128)
        return seg_logits, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
