# common/models/torch_resnet_model.py

from common.ml_interfaces.model_adapter import ModelAdapter
import torch
import torch.nn as nn
import torchvision.models as models


class TorchResNetModel(ModelAdapter):
    def __init__(self, num_classes=7, lr=0.001):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _build_model(self, num_classes):
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze early layers
        for name, param in model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1")):
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def init_model(self):
        """Set model to training mode"""
        self.model.train()

    def get_weights(self):
        """Return model weights for server aggregation"""
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        """Load weights received from server"""
        self.model.load_state_dict(weights)
        self.model.to(self.device)

    def get_gradients(self):
        """Return gradients after backprop (for distributed training)"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu()
        return gradients
