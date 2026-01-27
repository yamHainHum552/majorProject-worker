import yaml
from worker.worker import Worker
from common.utils import get_machine_info


def load_config():
    with open("config/worker_config.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("[Worker] Starting worker node...")
    cfg = load_config()

    worker = Worker(
        worker_id=cfg["worker_id"],
        coordinator_host=cfg["coordinator"]["host"],
        coordinator_port=cfg["coordinator"]["port"],
        machine_info=get_machine_info(),
    )

    worker.register()
    worker.listen()


if __name__ == "__main__":
    main()
