from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))


        # Convolutional layers first block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(64)

        # Second block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(128)

        # Third block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(256)

        # Final block
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        # Adding Activation and max pooling layers
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3) # To reduce overfitting


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # First convolutional block
        z = self.maxPool(self.relu(self.batch1(self.conv1(z))))

        # Second convolutional block
        z = self.maxPool(self.relu(self.batch2(self.conv2(z))))

        # third convolutional block
        z = self.maxPool(self.relu(self.batch3(self.conv3(z))))

        # We then apply global average pooling to reduce spatial dimensions to 1x1
        z = self.global_pool(z)
        z =self.dropout(z)

        # TODO: replace with actual forward pass
        # Apply the final convolutional layer
        logits = self.conv4(z).squeeze(-1).squeeze(-1)

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


# --- Downsampling Block ---
# --- Residual Downsampling Block ---
class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A downsampling block with residual connection.
        It uses two 3x3 convolutions.
        The first convolution does not downsample (stride=1) and the second does (stride=2).
        If in_channels != out_channels, a 1x1 convolution is used on the shortcut to match dimensions.
        """
        super(ResidualDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Create a shortcut path if the dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out


# --- Upsampling Block with Skip Connection ---
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsampling block that first upsamples using ConvTranspose2d,
        then concatenates with the skip connection, and processes with conv layers.
        """
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x_up = self.upconv(x)
        # If spatial dimensions differ due to rounding, interpolate to match skip size.
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along the channel dimension
        x_cat = torch.cat([x_up, skip], dim=1)
        out = self.conv(x_cat)
        return out


# --- Detector Model ---
class Detector(nn.Module):
    def __init__(self, num_classes=3):
        """
        Detector model with an encoder-decoder architecture using skip connections.
        The network processes image features and branches into two heads:
         - Segmentation head: outputs logits for each class.
         - Depth head: outputs a single-channel depth map (scaled to [0, 1] via Sigmoid).
        Input: (B, 3, H, W)
        Outputs:
         - Segmentation logits: (B, num_classes, H, W)
         - Depth: (B, H, W)
        """
        super(Detector, self).__init__()
        # Encoder using Residual Downsampling Blocks
        self.down1 = ResidualDownBlock(3, 16)  # (B, 3, H, W) -> (B, 16, H/2, W/2)
        self.down2 = ResidualDownBlock(16, 32)  # -> (B, 32, H/4, W/4)
        self.down3 = ResidualDownBlock(32, 64)  # -> (B, 64, H/8, W/8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling Blocks with Skip Connections)
        self.up1 = UpBlock(128, 64)  # Upsample from H/8 to H/4, use d3 skip connection
        self.up2 = UpBlock(64, 32)  # Upsample from H/4 to H/2, use d2 skip connection
        self.up3 = UpBlock(32, 16)  # Upsample from H/2 to H, use d1 skip connection

        # Segmentation head: outputs logits for each class.
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        # Depth head: outputs a single channel and uses Sigmoid to scale outputs in [0, 1]
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path
        d1 = self.down1(x)  # (B, 16, H/2, W/2)
        d2 = self.down2(d1)  # (B, 32, H/4, W/4)
        d3 = self.down3(d2)  # (B, 64, H/8, W/8)

        # Bottleneck
        bn = self.bottleneck(d3)  # (B, 128, H/8, W/8)

        # Decoder path with skip connections
        u1 = self.up1(bn, d3)  # (B, 64, H/4, W/4)
        u2 = self.up2(u1, d2)  # (B, 32, H/2, W/2)
        u3 = self.up3(u2, d1)  # (B, 16, H, W)

        # Heads for segmentation and depth prediction
        seg_logits = self.seg_head(u3)  # (B, num_classes, H, W)
        depth = self.depth_head(u3).squeeze(1)  # (B, H, W)
        return seg_logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference: returns both predicted segmentation (class labels) and depth.
        Args:
            x (torch.FloatTensor): input image with shape (B, 3, h, w)
        Returns:
            tuple:
              - pred: predicted class labels with shape (B, h, w)
              - depth: predicted depth in [0, 1] with shape (B, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        depth = raw_depth  # Already scaled by Sigmoid
        return pred, depth
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.

        Args:
            x (torch.FloatTensor): image with shape (B, 3, h, w) and values in [0, 1]

        Returns:
            tuple:
              - pred: class labels {0, 1, 2} with shape (B, h, w)
              - depth: normalized depth [0, 1] with shape (B, h, w)
        """
        logits, raw_depth = self(x)
        # Print shapes for debugging
        # print("Logits shape:", logits.shape, "Raw depth shape:", raw_depth.shape)
        pred = logits.argmax(dim=1)
        # Optionally, post-process depth if needed; here raw_depth is already scaled via Sigmoid.
        depth = raw_depth
        return pred, depth

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
