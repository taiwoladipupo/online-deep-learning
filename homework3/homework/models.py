from pathlib import Path

import torch
import torch.nn as nn
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class Detector(nn.Module):
    def __init__(self, num_classes=3):
        """
        Detector model.
        Assumes input images are (B, 3, 192, 256) so that the output matches label size (B, 96, 128).
        Outputs:
         - Segmentation logits: (B, num_classes, 96, 128)
         - Depth map: (B, 96, 128)
        """
        super(Detector, self).__init__()
        # Downsampling Block 1: from (B,3,192,256) -> (B,16,96,128)
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B,16,96,128)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Downsampling Block 2: from (B,16,96,128) -> (B,32,48,64)
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B,32,48,64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Upsampling Block 1: upsample from (B,32,48,64) -> (B,16,96,128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # (B,16,96,128)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Skip connection: concatenate up1 output with down1 features (both now (B,16,96,128))
        # After concatenation, channels = 32.
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (B,16,96,128)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Segmentation head: outputs logits (B, num_classes, 96, 128)
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        # Depth head: outputs depth (B, 1, 96, 128) -> squeeze to (B,96,128)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 192, 256)
        d1 = self.down1(x)       # (B,16,96,128)
        d2 = self.down2(d1)      # (B,32,48,64)
        u1 = self.up1(d2)        # (B,16,96,128)
        # Concatenate skip connection from d1 with upsampled features along channel dimension.
        u1_cat = torch.cat([u1, d1], dim=1)  # (B,32,96,128)
        u1_out = self.conv_up1(u1_cat)        # (B,16,96,128)
        logits = self.seg_head(u1_out)        # (B,3,96,128)
        depth = self.depth_head(u1_out)       # (B,1,96,128)
        depth = depth.squeeze(1)              # (B,96,128)
        return logits, depth


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
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
