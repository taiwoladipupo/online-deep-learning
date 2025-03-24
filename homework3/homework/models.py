from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

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




class Detector(nn.Module):
    def __init__(self, num_classes=3):
        """
        Detector model with a pretrained ResNet34 encoder.
        Assumes input images are (B, 3, 192, 256).
        The encoder downsamples the image while the decoder upsamples
        with skip connections to produce:
         - Segmentation logits: (B, num_classes, 96, 128)
         - Depth map: (B, 96, 128)
        """
        super(Detector, self).__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=True)
        # Encoder: use conv1, bn1, relu from ResNet (no maxpool) to preserve more resolution.
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Output: (B,64,96,128)
        # Then include maxpool and layer1, layer2, layer3, layer4.
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)           # (B,64,48,64)
        self.layer2 = resnet.layer2                                           # (B,128,24,32)
        self.layer3 = resnet.layer3                                           # (B,256,12,16)
        self.layer4 = resnet.layer4                                           # (B,512,6,8)

        # Decoder: use transposed convolutions with skip connections.
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)      # (B,256,12,16)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)      # (B,128,24,32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128+128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)       # (B,64,48,64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)        # (B,64,96,128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Final segmentation and depth heads.
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)   # (B,64,96,128)
        x1 = self.layer1(x0)  # (B,64,48,64)
        x2 = self.layer2(x1)  # (B,128,24,32)
        x3 = self.layer3(x2)  # (B,256,12,16)
        x4 = self.layer4(x3)  # (B,512,6,8)
        # Decoder with skip connections
        d4 = self.up4(x4)     # (B,256,12,16)
        d4 = torch.cat([d4, x3], dim=1)  # (B,512,12,16)
        d4 = self.conv4(d4)   # (B,256,12,16)
        d3 = self.up3(d4)     # (B,128,24,32)
        d3 = torch.cat([d3, x2], dim=1)  # (B,256,24,32)
        d3 = self.conv3(d3)   # (B,128,24,32)
        d2 = self.up2(d3)     # (B,64,48,64)
        d2 = torch.cat([d2, x1], dim=1)  # (B,128,48,64)
        d2 = self.conv2(d2)   # (B,64,48,64)
        d1 = self.up1(d2)     # (B,64,96,128)
        d1 = torch.cat([d1, x0], dim=1)  # (B,128,96,128)
        d1 = self.conv1(d1)   # (B,64,96,128)
        seg_logits = self.seg_head(d1)  # (B, num_classes, 96,128)
        depth = self.depth_head(d1)     # (B,1,96,128)
        depth = depth.squeeze(1)        # (B,96,128)
        return seg_logits, depth

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
