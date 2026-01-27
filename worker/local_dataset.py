# worker/local_dataset.py
from torch.utils.data import Dataset


class LocalDataset(Dataset):
    """
    Worker-side dataset.
    Receives already-split shard data from coordinator.
    """

    def __init__(self, shard_data: dict):
        self.x = shard_data["x"]
        self.y = shard_data["y"]

        if len(self.x) != len(self.y):
            raise ValueError("Feature/label size mismatch")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
