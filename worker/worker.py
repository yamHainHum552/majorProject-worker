# worker/worker.py

import time
import base64
from tqdm import tqdm

from worker.communicator import WorkerCommunicator
from worker.training.torch_trainer import TorchTrainer
from worker.local_dataset import LocalDataset
from worker.data_manager import WorkerDataManager

from common.models.torch_resnet_model import TorchResNetModel
from common.serialization import deserialize, serialize
from common.utils import get_machine_power


class Worker:

    def __init__(self, worker_id, coordinator_host, coordinator_port):

        self.worker_id = worker_id

        # Dataset state
        self.data_buffer = bytearray()
        self._dataset = None

        # Training state
        self.current_round = 0
        self.running = True

        # System info
        self.machine_info = get_machine_power()

        # Communication
        self.communicator = WorkerCommunicator(
            coordinator_host,
            coordinator_port
        )

        # Local storage
        self.data_manager = WorkerDataManager()

        # Detect existing dataset
        self.dataset_initialized = self.data_manager.has_data()

        if self.dataset_initialized:
            print(
                f"[Worker {self.worker_id}] Existing dataset detected. "
                f"Skipping bootstrap."
            )

        # ML components
        self.trainer = TorchTrainer()

        self.model_adapter = TorchResNetModel(num_classes=7)
        self.model_adapter.init_model()

        # Model sanity check
        trainable = sum(
            p.numel()
            for p in self.model_adapter.model.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in self.model_adapter.model.parameters()
        )

        print(
            f"[Worker {self.worker_id}] Model initialized | "
            f"Trainable params: {trainable}/{total}"
        )

    # ------------------------------------------------
    # REGISTER WORKER
    # ------------------------------------------------
    def register(self):

        info = self.machine_info

        print(
            f"\n[Worker {self.worker_id}] === SYSTEM POWER REPORT ===\n"
            f"Platform : {info['platform']}\n"
            f"CPU : {info['cpu']['cores']} cores @ {info['cpu']['max_ghz']} GHz\n"
            f"RAM : {info['ram']['available_gb']} GB available\n"
            f"GPU : {info['gpu']['gpu_type']} ({info['gpu']['backend']})\n"
            f"TOTAL POWER SCORE: {info['scores']['total']}\n"
        )

        while True:

            try:

                self.communicator.register_worker({

                    "worker_id": self.worker_id,

                    "power": info["scores"]["total"],

                    "has_data": self.data_manager.has_data()

                })

                print(f"[Worker {self.worker_id}] Registered successfully.")

                break

            except Exception as e:

                print("[Worker] Coordinator unreachable:", e)

                print("[Worker] Retrying in 5 seconds...")

                time.sleep(5)

    # ------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------
    def listen(self):

        print(f"[Worker {self.worker_id}] Waiting for tasks...")

        while self.running:

            command = self.communicator.wait_for_command(self.worker_id)

            self.handle_command(command)

    # ------------------------------------------------
    # COMMAND HANDLER
    # ------------------------------------------------
    def handle_command(self, command: dict):

        cmd = command.get("cmd")

        if cmd == "WAIT":

            time.sleep(0.2)

            return

        if cmd == "INIT_DATA_START":

            self._handle_data_init()

            return

        if cmd == "TRAIN":

            self.run_training_round(command)

            return

        if cmd == "STOP_TRAINING":

            print(
                f"[Worker {self.worker_id}] STOP received. Shutting down."
            )

            self.running = False

            return

    # ------------------------------------------------
    # DATA INITIALIZATION (ZIP STREAM)
    # ------------------------------------------------
    def _handle_data_init(self):

        if self.dataset_initialized:
            return

        print(f"[Worker {self.worker_id}] Receiving dataset...")

        self.data_buffer = bytearray()

        with tqdm(
            unit="B",
            unit_scale=True,
            desc="Dataset Transfer",
            leave=True
        ) as pbar:

            while True:

                try:
                    resp = self.communicator.fetch_data_chunk(self.worker_id)
                except Exception as e:
                    raise RuntimeError(
                        "Dataset transfer interrupted by coordinator: "
                        f"{e}"
                    ) from e

                if resp is None:
                    break

                chunk = base64.b64decode(resp["chunk"])

                self.data_buffer.extend(chunk)

                pbar.update(len(chunk))

        if len(self.data_buffer) == 0:
            raise RuntimeError(
                "No dataset bytes received from coordinator"
            )

        # Extract dataset
        self.data_manager.store_zip(bytes(self.data_buffer))

        self.dataset_initialized = True

        self.data_buffer.clear()

        print(
            f"[Worker {self.worker_id}] Dataset stored successfully."
        )

        self.communicator.send_data_ready(self.worker_id)

    # ------------------------------------------------
    # TRAINING ROUND
    # ------------------------------------------------
    def run_training_round(self, command: dict):

        if not self.dataset_initialized:

            print(
                f"[Worker {self.worker_id}] Dataset not ready. Skipping training."
            )

            return

        round_id = command.get("round_id", 0)

        if round_id == self.current_round:

            print(
                f"[Worker {self.worker_id}] Duplicate round {round_id} ignored"
            )

            return

        if round_id != self.current_round + 1:

            print(
                f"[Worker {self.worker_id}] Round mismatch. "
                f"Expected {self.current_round + 1}, got {round_id}"
            )

            return

        self.current_round = round_id

        # ------------------------------------------------
        # LOAD GLOBAL MODEL
        # ------------------------------------------------
        global_weights = deserialize(command["global_weights"])

        training_config = command["training_config"]

        self.model_adapter.set_weights(global_weights)

        # ------------------------------------------------
        # LOAD DATASET
        # ------------------------------------------------
        if self._dataset is None:

            self._dataset = LocalDataset(
                root_dir=self.data_manager.base_dir
            )

        # ------------------------------------------------
        # LOCAL TRAINING
        # ------------------------------------------------
        updated_weights, metrics = self.trainer.train_one_round(

            self.model_adapter,

            self._dataset,

            training_config

        )

        print(
            f"[Worker {self.worker_id}] "
            f"TRAIN → Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.2f}%"
        )

        # ------------------------------------------------
        # SEND MODEL UPDATE
        # ------------------------------------------------
        try:

            self.communicator.send_gradients({

                "worker_id": self.worker_id,

                "round_id": round_id,

                "weights": serialize(updated_weights),

                "metrics": metrics

            })

        except Exception as e:

            print(
                f"[Worker {self.worker_id}] Coordinator unreachable:", e
            )

            self.running = False
