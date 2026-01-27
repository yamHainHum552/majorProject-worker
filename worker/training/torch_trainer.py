# worker/training/torch_trainer.py
import torch
from torch.utils.data import DataLoader
from common.ml_interfaces.trainer import Trainer


class TorchTrainer(Trainer):
    def train_one_round(self, model_adapter, dataset, training_config):
        model = model_adapter.model
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config["lr"]
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        loader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=True
        )

        optimizer.zero_grad()

        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

        return model_adapter.get_gradients()
