# worker/training/torch_trainer.py
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter


class TorchTrainer:
    """
    Worker-side trainer
    - Performs local fine-tuning
    - Returns UPDATED MODEL WEIGHTS (not gradients)
    """

    def train_one_round(self, model_adapter, dataset, config):
        # -------------------------
        # Model
        # -------------------------
        model = model_adapter.model
        model.train()

        # -------------------------
        # Training config
        # -------------------------
        batch_size = config.get("batch_size", 16)
        learning_rate = config.get("learning_rate", 1e-4)

        # -------------------------
        # Device
        # -------------------------
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)

        # -------------------------
        # Optimizer (ONLY trainable params)
        # -------------------------
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # -------------------------
        # LR Scheduler
        # -------------------------
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        # -------------------------
        # Class-weighted loss (HAM10000 imbalance)
        # -------------------------
        labels = [label for _, label in dataset.samples]
        counts = Counter(labels)

        class_weights = torch.tensor(
            [1.0 / counts[i] for i in range(7)],
            dtype=torch.float,
            device=device
        )

        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1
        )

        # -------------------------
        # DataLoader
        # -------------------------
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=use_cuda
        )

        # -------------------------
        # AMP (CUDA ONLY, version-safe)
        # -------------------------
        if use_cuda:
            scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = torch.cuda.amp.autocast
        else:
            scaler = None
            autocast_ctx = None

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            loader,
            desc="Training",
            unit="batch",
            leave=True
        )

        # =========================
        # TRAIN LOOP
        # =========================
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda:
                with autocast_ctx():
                    outputs = model(x)
                    loss = criterion(outputs, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
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

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # -------------------------
        # Metrics
        # -------------------------
        avg_loss = total_loss / len(loader)
        accuracy = (correct / total) * 100.0

        scheduler.step(avg_loss)

        print(
            f"[Worker] Round completed | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        # -------------------------
        # 🔑 RETURN UPDATED WEIGHTS
        # -------------------------
        updated_weights = model_adapter.get_weights()

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        return updated_weights, metrics
