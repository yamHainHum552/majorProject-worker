# worker/data_manager.py
import zipfile
import io
import os
from pathlib import Path


class WorkerDataManager:
    def __init__(self, base_dir="worker_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def has_data(self):
        images_dir = os.path.join(self.base_dir, "images")
        csv_path = os.path.join(self.base_dir, "HAM10000_metadata.csv")

        return (
            os.path.isdir(images_dir) and
            os.path.isfile(csv_path) and
            len(os.listdir(images_dir)) > 0
        )

    def store_zip(self, zip_bytes: bytes):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(self.base_dir)
