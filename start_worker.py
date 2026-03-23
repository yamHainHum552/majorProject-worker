# start_worker.py

import os
import socket
import sys
import yaml
from worker.worker import Worker


def load_config():
    try:
        with open("config/worker_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        if "worker_id" not in cfg:
            raise ValueError("worker_id missing in worker_config.yaml")

        if "coordinator" not in cfg:
            raise ValueError(
                "coordinator section missing in worker_config.yaml")

        if "host" not in cfg["coordinator"]:
            raise ValueError("coordinator.host missing in worker_config.yaml")

        if "port" not in cfg["coordinator"]:
            raise ValueError("coordinator.port missing in worker_config.yaml")

        cfg["worker_id"] = resolve_worker_id(cfg["worker_id"])

        return cfg

    except Exception as e:
        print("[Worker] Failed to load configuration:", e)
        sys.exit(1)


def resolve_worker_id(config_worker_id):
    env_worker_id = os.getenv("WORKER_ID", "").strip()
    if env_worker_id:
        print(f"[Worker] Using worker ID from WORKER_ID: {env_worker_id}")
        return env_worker_id

    worker_id = str(config_worker_id).strip()
    if not worker_id:
        raise ValueError("worker_id must not be empty")

    # Treat the checked-in value as a shared placeholder and derive a
    # machine-specific ID so copied configs do not overwrite each other.
    if worker_id == "worker_1":
        hostname = socket.gethostname().strip().lower().replace(" ", "-")
        resolved_worker_id = f"worker_{hostname}"
        print(
            "[Worker] Placeholder worker_id 'worker_1' detected. "
            f"Using machine-specific ID: {resolved_worker_id}"
        )
        return resolved_worker_id

    return worker_id


def main():

    print("\n[Worker] Starting worker node...\n")

    cfg = load_config()

    worker_id = cfg["worker_id"]
    host = cfg["coordinator"]["host"]
    port = cfg["coordinator"]["port"]

    print(f"[Worker] Worker ID: {worker_id}")
    print(f"[Worker] Coordinator Host: {host}")
    print(f"[Worker] Coordinator Port: {port}\n")

    try:
        worker = Worker(
            worker_id=worker_id,
            coordinator_host=host,
            coordinator_port=port,
        )

        worker.register()

        worker.listen()

    except KeyboardInterrupt:
        print("\n[Worker] Shutdown requested. Exiting.")
        sys.exit(0)

    except Exception as e:
        print("[Worker] Fatal error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
