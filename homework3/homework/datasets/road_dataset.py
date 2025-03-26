from pathlib import Path
from typing import Union
from imblearn.over_sampling import SMOTE
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import road_transforms
from .road_utils import Track
from .road_transforms import RandomRotation
from PIL import Image

class RoadDataset(Dataset):
    """
    SuperTux dataset for road detection
    """

    def __init__(
        self,
        episode_path: str,
        transform_pipeline: str = "default",
    ):
        super().__init__()

        self.episode_path = Path(episode_path)

        info = np.load(self.episode_path / "info.npz", allow_pickle=True)

        self.track = Track(**info["track"].item())
        self.frames: dict[str, np.ndarray] = {k: np.stack(v) for k, v in info["frames"].item().items()}
        self.transform = self.get_transform(transform_pipeline)

    def get_transform(self, transform_pipeline: str):
        if transform_pipeline == "default":
            xform = road_transforms.Compose([
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
                road_transforms.TrackProcessor(self.track),
            ])
        elif transform_pipeline == "aug":
            xform = road_transforms.Compose([
                # Load the raw data
                road_transforms.ImageLoader(self.episode_path),
                road_transforms.DepthLoader(self.episode_path),
                road_transforms.TrackProcessor(self.track),
                # Apply spatial augmentations consistently to image, depth, and track:
                road_transforms.RandomHorizontalFlip(p=0.5),
                road_transforms.RandomRotation(degrees=15),
                # Apply color jitter only to the image (do not modify the mask)
                road_transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                # Resize the image and mask separately with appropriate interpolation:
                road_transforms.Resize(
                    (96, 128),
                    resample=Image.Resampling.BILINEAR
                ),
                road_transforms.Resize(
                    (96, 128),
                    resample=Image.Resampling.NEAREST
                ),
            ])
        return xform
        return xform

    def __len__(self):
        return len(self.frames["location"])

    def __getitem__(self, idx: int):
        """
        Returns:
            dict: sample data with keys "image", "depth", "track"
        """
        sample = {"_idx": idx, "_frames": self.frames}
        sample = self.transform(sample)

        # remove private keys
        for key in list(sample.keys()):
            if key.startswith("_"):
                sample.pop(key)

        return sample



def oversample_minority_classes(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
    oversample: bool = False
) -> Union[DataLoader, Dataset]:
    dataset_path = Path(dataset_path)
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]

    if not scenes and dataset_path.is_dir():
        scenes = [dataset_path]

    datasets = []
    for episode_path in sorted(scenes):
        datasets.append(RoadDataset(episode_path, transform_pipeline=transform_pipeline))
    dataset = ConcatDataset(datasets)

    if oversample:
        X = [sample['image'] for sample in dataset]
        y = [sample['track'] for sample in dataset]
        X_res, y_res = oversample_minority_classes(X, y)
        dataset = [(X_res[i], y_res[i]) for i in range(len(X_res))]

    print(f"Loaded {len(dataset)} samples from {len(datasets)} episodes")

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )