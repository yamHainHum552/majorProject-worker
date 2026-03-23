# worker/local_dataset.py
from pathlib import Path
from collections import Counter
import math
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
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.02
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
        self.inverse_class_map = {
            idx: label for label, idx in self.class_map.items()
        }

    def get_class_distribution(self):
        return {
            self.inverse_class_map[label]: self.class_counts.get(label, 0)
            for label in sorted(self.inverse_class_map)
        }

    def describe_distribution(self, title):
        distribution = self.get_class_distribution()
        summary = ", ".join(
            f"{label}={count}" for label, count in distribution.items()
        )
        return f"{title}: {summary}"

    def get_limited_upsampling_plan(self, max_upsample_factor=3.0):
        majority_count = max(self.class_counts.values())
        target_counts = {}

        for label in sorted(self.inverse_class_map):
            original_count = self.class_counts.get(label, 0)
            if original_count == 0:
                target_counts[label] = 0
                continue

            capped_target = math.ceil(original_count * max_upsample_factor)
            target_counts[label] = min(majority_count, capped_target)

        weights = []
        for _, label in self.samples:
            original_count = self.class_counts[label]
            target_count = target_counts[label]
            weights.append(target_count / original_count)

        return {
            "sample_weights": torch.tensor(weights, dtype=torch.double),
            "target_counts": target_counts,
            "num_samples": int(sum(target_counts.values())),
            "max_upsample_factor": max_upsample_factor,
        }

    def get_sample_weights(self, max_upsample_factor=3.0):
        plan = self.get_limited_upsampling_plan(
            max_upsample_factor=max_upsample_factor
        )

        return plan["sample_weights"]

    def get_class_weights(self):
        total_samples = len(self.samples)
        num_classes = len(self.class_map)

        weights = torch.ones(num_classes, dtype=torch.float32)
        for label, count in self.class_counts.items():
            if count > 0:
                weights[label] = total_samples / (num_classes * count)

        return weights

    def format_target_distribution(self, target_counts):
        return {
            self.inverse_class_map[label]: target_counts.get(label, 0)
            for label in sorted(self.inverse_class_map)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
