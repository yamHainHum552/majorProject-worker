import torch.nn as nn
from common.ml_interfaces.model_adapter import ModelAdapter


class TorchCNNModel(ModelAdapter):
    """
    Shared CNN model for MNIST.
    Used by both coordinator and worker.
    """

    def __init__(self):
        self.model = None

    def init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 32, 10)
        )

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_gradients(self):
        return {
            name: param.grad.clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
