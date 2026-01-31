# start_worker.py
import yaml
from worker.worker import Worker


def load_config():
    with open("config/worker_config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    print("[Worker] Starting worker node...")

    cfg = load_config()

    worker = Worker(
        worker_id=cfg["worker_id"],
        coordinator_host=cfg["coordinator"]["host"],
        coordinator_port=cfg["coordinator"]["port"],
    )

    # ✅ REGISTER ONCE (this prints power + sends power)
    worker.register()

    # ✅ START LISTENING
    worker.listen()


if __name__ == "__main__":
    main()
