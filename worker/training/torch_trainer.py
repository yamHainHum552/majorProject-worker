# worker/training/torch_trainer.py
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TorchTrainer:
    """
    Worker-side trainer
    - Performs local fine-tuning
    - Returns UPDATED MODEL WEIGHTS (FedAvg style)
    """

    def train_one_round(self, model_adapter, dataset, config):
        model = model_adapter.model
        model.train()

        batch_size = config.get("batch_size", 16)
        learning_rate = config.get("learning_rate", 3e-4)
        balance_minority_classes = config.get(
            "balance_minority_classes", True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        criterion = torch.nn.CrossEntropyLoss()

        sampler = None
        shuffle = True
        if balance_minority_classes:
            sample_weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False

            print(
                "[Worker] Using minority-class upsampling with "
                f"class distribution: {dataset.get_class_distribution()}"
            )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(loader, desc="Training", unit="batch"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = (correct / total) * 100.0
        scheduler.step(avg_loss)

        print(
            f"[Worker] Round completed | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        updated_weights = model_adapter.get_weights()

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        return updated_weights, metrics
