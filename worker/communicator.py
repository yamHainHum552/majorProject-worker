import json
import urllib.request


class WorkerCommunicator:
    def __init__(self, coordinator_host, coordinator_port):
        self.base_url = f"http://{coordinator_host}:{coordinator_port}"

    def register_worker(self, payload: dict):
        req = urllib.request.Request(
            f"{self.base_url}/register",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req):
            return {"status": "ok"}

    def wait_for_command(self, worker_id: str):
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/next-command?worker_id={worker_id}"
            ) as res:
                return json.loads(res.read().decode())
        except:
            return {"type": "idle"}

    def update_status(self, worker_id: str, state: str):
        req = urllib.request.Request(
            f"{self.base_url}/update-status",
            data=json.dumps({"worker_id": worker_id, "state": state}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req)
