# worker/local_dataset.py
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import torch


class LocalDataset(Dataset):
    """
    Worker-side HAM10000 dataset
    """

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)

        csv_files = list(self.root.glob("HAM10000_metadata.csv"))
        if not csv_files:
            raise RuntimeError("HAM10000_metadata.csv not found")

        self.metadata = pd.read_csv(csv_files[0])

        self.class_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }

        self.images_dir = self.root / "images"
        if not self.images_dir.exists():
            raise RuntimeError("images/ directory missing")

        # --------------------------
        # Medical-safe augmentation
        # --------------------------
        self.transform = transform or transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = []

        for img_path in self.images_dir.glob("*.jpg"):
            image_id = img_path.stem
            row = self.metadata[self.metadata["image_id"] == image_id]

            if row.empty:
                continue

            label_str = row.iloc[0]["dx"]
            if label_str not in self.class_map:
                continue

            self.samples.append(
                (img_path, self.class_map[label_str])
            )

        if not self.samples:
            raise RuntimeError("No valid samples found")

        self.class_counts = Counter(label for _, label in self.samples)

    def get_class_distribution(self):
        inverse_class_map = {
            idx: label for label, idx in self.class_map.items()
        }
        return {
            inverse_class_map[label]: count
            for label, count in sorted(self.class_counts.items())
        }

    def get_sample_weights(self):
        total_samples = len(self.samples)
        num_classes = len(self.class_counts)

        class_weights = {
            label: total_samples / (num_classes * count)
            for label, count in self.class_counts.items()
            if count > 0
        }

        return torch.tensor(
            [class_weights[label] for _, label in self.samples],
            dtype=torch.double
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
