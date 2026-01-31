# worker/worker.py
import time
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

        # --------------------------
        # Dataset state
        # --------------------------
        self.dataset_initialized = False
        self.data_buffer = bytearray()
        self._dataset = None  # cache dataset after first load

        # --------------------------
        # Training state
        # --------------------------
        self.current_round = 0
        self.running = True

        # --------------------------
        # System info
        # --------------------------
        self.machine_info = get_machine_power()

        # --------------------------
        # Communication
        # --------------------------
        self.communicator = WorkerCommunicator(
            coordinator_host, coordinator_port
        )

        # --------------------------
        # Local storage
        # --------------------------
        self.data_manager = WorkerDataManager()

        # --------------------------
        # ML components
        # --------------------------
        self.trainer = TorchTrainer()

        # 🔑 Model adapter (pretrained ResNet18)
        self.model_adapter = TorchResNetModel(num_classes=7)
        self.model_adapter.init_model()

        # --------------------------
        # Sanity check (optional, safe)
        # --------------------------
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

    # ==================================================
    # REGISTRATION
    # ==================================================
    def register(self):
        info = self.machine_info

        print(
            f"\n[Worker {self.worker_id}] === SYSTEM POWER REPORT ===\n"
            f"CPU : {info['cpu_cores']} cores @ {info['cpu_ghz']} GHz\n"
            f"RAM : {info['available_ram_gb']} GB available\n"
            f"GPU : {info['gpu']['gpu_type']}\n"
            f"TOTAL POWER SCORE: {info['scores']['total']}\n"
        )

        self.communicator.register_worker({
            "worker_id": self.worker_id,
            "power": info["scores"]["total"],
            "has_data": self.data_manager.has_data()
        })

    # ==================================================
    # MAIN LOOP
    # ==================================================
    def listen(self):
        print(f"[Worker {self.worker_id}] Waiting for tasks...")
        while self.running:
            command = self.communicator.wait_for_command(self.worker_id)
            self.handle_command(command)

    # ==================================================
    # COMMAND HANDLER
    # ==================================================
    def handle_command(self, command: dict):
        cmd = command.get("cmd")

        if cmd == "WAIT":
            time.sleep(1)
            return

        # --------------------------
        # DATA BOOTSTRAP (ZIP ONCE)
        # --------------------------
        if cmd == "INIT_DATA_START":
            self._handle_data_init()
            return

        # --------------------------
        # TRAIN
        # --------------------------
        if cmd == "TRAIN":
            self.run_training_round(command)
            return

        # --------------------------
        # STOP
        # --------------------------
        if cmd == "STOP_TRAINING":
            print(f"[Worker {self.worker_id}] STOP received. Shutting down.")
            self.running = False
            return

    # ==================================================
    # DATA INITIALIZATION
    # ==================================================
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
                resp = self.communicator.fetch_data_chunk(self.worker_id)

                # ✅ End of stream
                if resp is None:
                    break

                chunk = deserialize(resp["chunk"])
                self.data_buffer.extend(chunk)
                pbar.update(len(chunk))

        # ZIP integrity + extract
        try:
            self.data_manager.store_zip(bytes(self.data_buffer))
        except Exception as e:
            raise RuntimeError("Dataset ZIP corrupted") from e

        self.dataset_initialized = True
        self.data_buffer.clear()

        print(f"[Worker {self.worker_id}] Dataset stored successfully.")
        self.communicator.send_data_ready(self.worker_id)

    # ==================================================
    # TRAINING ROUND
    # ==================================================
    def run_training_round(self, command: dict):
        round_id = command.get("round_id", 0)

        # 🔒 Ignore stale or duplicate rounds
        if round_id <= self.current_round:
            print(
                f"[Worker {self.worker_id}] Ignoring stale round {round_id}"
            )
            return

        self.current_round = round_id

        # --------------------------
        # Load global weights
        # --------------------------
        global_weights = deserialize(command["global_weights"])
        training_config = command["training_config"]

        self.model_adapter.set_weights(global_weights)

        # --------------------------
        # Load dataset ONCE
        # --------------------------
        if self._dataset is None:
            self._dataset = LocalDataset(
                root_dir=self.data_manager.base_dir
            )

        # --------------------------
        # Train locally
        # --------------------------
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

        # --------------------------
        # SEND UPDATED WEIGHTS
        # (weight-based distributed learning)
        # --------------------------
        try:
            self.communicator.send_gradients({
                "worker_id": self.worker_id,
                "weights": serialize(updated_weights),
                "metrics": metrics
            })
        except Exception:
            print(
                f"[Worker {self.worker_id}] Coordinator unreachable. Exiting."
            )
            self.running = False
