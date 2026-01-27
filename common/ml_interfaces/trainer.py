# common/ml_interfaces/trainer.py
from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    """
    Worker-side training abstraction.
    """

    @abstractmethod
    def train_one_round(
        self,
        model_adapter: Any,
        dataset: Any,
        training_config: dict
    ) -> Any:
        """
        Perform one round of local training.
        """
        pass
