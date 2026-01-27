# worker/communicator.py
import json
import urllib.request


class WorkerCommunicator:
    def __init__(self, coordinator_host, coordinator_port):
        self.base_url = f"http://{coordinator_host}:{coordinator_port}"

    def register_worker(self, payload: dict):
        self._post("/register", payload)

    def wait_for_command(self, worker_id: str):
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/next-command?worker_id={worker_id}",
                timeout=60  # 🔴 IMPORTANT
            ) as res:
                data = res.read()
                print(f"[Worker {worker_id}] Received {len(data)} bytes")
                return json.loads(data.decode())
        except Exception as e:
            print(f"[Worker {worker_id}] Poll error:", str(e))
            return {"type": "idle"}

    def update_status(self, worker_id: str, state: str):
        self._post("/update-status", {
            "worker_id": worker_id,
            "state": state
        })

    def send_gradients(self, payload: dict):
        self._post("/submit-gradients", payload)

    def _post(self, endpoint: str, payload: dict):
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req)
