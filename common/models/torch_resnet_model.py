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
        self.configure_fine_tuning()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _build_model(self, num_classes):
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def configure_fine_tuning(self, mode="partial", freeze_blocks=3):
        if mode not in {"head_only", "partial", "full"}:
            raise ValueError(
                f"Unsupported fine_tune_mode '{mode}'. "
                "Use 'head_only', 'partial', or 'full'."
            )

        for param in self.model.features.parameters():
            param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        if mode == "head_only":
            for param in self.model.features.parameters():
                param.requires_grad = False
        elif mode == "partial":
            freeze_blocks = max(0, min(int(freeze_blocks), len(self.model.features)))
            for block_idx, block in enumerate(self.model.features):
                if block_idx < freeze_blocks:
                    for param in block.parameters():
                        param.requires_grad = False

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[Worker] Fine-tuning mode: {mode} | "
            f"Frozen feature blocks: "
            f"{freeze_blocks if mode == 'partial' else ('all' if mode == 'head_only' else 0)} | "
            f"Trainable params: {trainable}/{total}"
        )

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
