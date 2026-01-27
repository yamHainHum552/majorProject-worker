from worker.communicator import WorkerCommunicator
from worker.training.torch_trainer import TorchTrainer
from common.models.torch_cnn_model import TorchCNNModel
from worker.local_dataset import LocalDataset
from common.serialization import deserialize, serialize
import time


class Worker:
    def __init__(self, worker_id, coordinator_host, coordinator_port, machine_info):
        self.worker_id = worker_id
        self.communicator = WorkerCommunicator(
            coordinator_host, coordinator_port
        )
        self.running = True

        # ML components
        self.trainer = TorchTrainer()
        self.model_adapter = TorchCNNModel()
        self.model_adapter.init_model()

    def register(self):
        self.communicator.register_worker({"worker_id": self.worker_id})

    def listen(self):
        print(f"[Worker {self.worker_id}] Waiting for tasks...")
        while self.running:
            command = self.communicator.wait_for_command(self.worker_id)
            self.handle_command(command)
            time.sleep(1)

    def handle_command(self, command: dict):
        if command.get("type") == "train":
            print(f"[Worker {self.worker_id}] Training round started")
            self.communicator.update_status(self.worker_id, "busy")

            self.run_training_round(command)

            self.communicator.update_status(self.worker_id, "idle")
            print(f"[Worker {self.worker_id}] Training round finished")

    def run_training_round(self, command: dict):
        """
        Expected command payload:
        {
            type: "train",
            shard_indices: bytes,
            global_weights: bytes,
            training_config: dict
        }
        """

        # Deserialize inputs
        shard_indices = deserialize(command["shard_indices"])
        global_weights = deserialize(command["global_weights"])
        training_config = command["training_config"]

        # Load model weights
        self.model_adapter.set_weights(global_weights)

        # Build local dataset shard
        dataset = LocalDataset(shard_indices)

        # Train locally
        gradients = self.trainer.train_one_round(
            self.model_adapter,
            dataset,
            training_config
        )

        # Send gradients back to coordinator
        self.communicator.send_gradients({
            "worker_id": self.worker_id,
            "gradients": serialize(gradients)
        })
