from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train_one_round(self, model, dataset, training_config):
        pass
