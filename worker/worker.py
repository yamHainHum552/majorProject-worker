# worker/worker.py
import time
from worker.communicator import WorkerCommunicator
from worker.training.torch_trainer import TorchTrainer
from common.models.torch_cnn_model import TorchCNNModel
from worker.local_dataset import LocalDataset
from common.serialization import deserialize, serialize


class Worker:
    def __init__(self, worker_id, coordinator_host, coordinator_port):
        self.worker_id = worker_id
        self.communicator = WorkerCommunicator(
            coordinator_host, coordinator_port
        )
        self.running = True

        self.trainer = TorchTrainer()
        self.model_adapter = TorchCNNModel()
        self.model_adapter.init_model()

    def register(self):
        self.communicator.register_worker({"worker_id": self.worker_id})

    def listen(self):
        print(f"[Worker {self.worker_id}] Waiting for tasks...")
        while self.running:
            command = self.communicator.wait_for_command(self.worker_id)
            print(f"[Worker {self.worker_id}] Command received:",
                  command.get("type"))
            self.handle_command(command)

    def handle_command(self, command: dict):
        if command.get("type") == "train":
            print(f"[Worker {self.worker_id}] Training started")
            self.communicator.update_status(self.worker_id, "busy")
            self.run_training_round(command)
            self.communicator.update_status(self.worker_id, "idle")
            print(f"[Worker {self.worker_id}] Training finished")

   # worker.py

    def run_training_round(self, command: dict):
        global_weights = deserialize(command["global_weights"])
        training_config = command["training_config"]

        self.model_adapter.set_weights(global_weights)

        shard_data = deserialize(command["shard_data"])
        dataset = LocalDataset(shard_data)

        gradients = self.trainer.train_one_round(
            self.model_adapter,
            dataset,
            training_config
        )

        self.communicator.send_gradients({
            "worker_id": self.worker_id,
            "gradients": serialize(gradients)
        })
