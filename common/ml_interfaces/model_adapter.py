# common/ml_interfaces/model_adapter.py

from abc import ABC, abstractmethod


class ModelAdapter(ABC):

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @abstractmethod
    def get_gradients(self):
        pass
