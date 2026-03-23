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
        fine_tune_mode = config.get("fine_tune_mode", "partial")
        freeze_blocks = config.get("freeze_blocks", 3)
        if hasattr(model_adapter, "configure_fine_tuning"):
            model_adapter.configure_fine_tuning(
                mode=fine_tune_mode,
                freeze_blocks=freeze_blocks
            )

        model = model_adapter.model
        model.train()

        batch_size = config.get("batch_size", 8)
        learning_rate = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 1e-4)
        label_smoothing = config.get("label_smoothing", 0.1)
        balance_minority_classes = config.get(
            "balance_minority_classes", True
        )
        max_upsample_factor = config.get("max_upsample_factor", 3.0)
        use_class_weighted_loss = config.get(
            "use_class_weighted_loss", True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        class_weights = None
        if use_class_weighted_loss:
            class_weights = dataset.get_class_weights().to(device)
            print(
                "[Worker] Using class-weighted loss with weights: "
                f"{class_weights.tolist()}"
            )

        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

        print(
            "[Worker] "
            + dataset.describe_distribution(
                "Raw class distribution before augmentation/upsampling"
            )
        )
        print(
            "[Worker] Using medically conservative augmentation: "
            "resize, horizontal flip, small rotation, small translation/"
            "scale, and mild color jitter"
        )

        sampler = None
        shuffle = True
        if balance_minority_classes:
            upsampling_plan = dataset.get_limited_upsampling_plan(
                max_upsample_factor=max_upsample_factor
            )
            sample_weights = upsampling_plan["sample_weights"]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=upsampling_plan["num_samples"],
                replacement=True
            )
            shuffle = False

            print(
                "[Worker] Limited minority upsampling enabled "
                f"(max factor {max_upsample_factor}x). "
                "Planned effective class distribution per round: "
                f"{dataset.format_target_distribution(upsampling_plan['target_counts'])}"
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
            f"Accuracy: {accuracy:.2f}% | "
            f"LR: {learning_rate:.6f} | "
            f"WD: {weight_decay:.6f}"
        )

        updated_weights = model_adapter.get_weights()

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        return updated_weights, metrics
