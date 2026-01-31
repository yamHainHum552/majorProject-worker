# worker/communicator.py
import json
import urllib.request
import urllib.error


class WorkerCommunicator:
    def __init__(self, coordinator_host, coordinator_port):
        self.base_url = f"http://{coordinator_host}:{coordinator_port}"

    def register_worker(self, payload: dict):
        self._post("/register", payload)

    def wait_for_command(self, worker_id: str):
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/next-command?worker_id={worker_id}",
                timeout=60
            ) as res:
                return json.loads(res.read().decode())
        except Exception:
            return {"cmd": "WAIT"}

    # ==================================================
    # 🔑 CRITICAL FIX: SAFE DATA CHUNK FETCH
    # ==================================================
    def fetch_data_chunk(self, worker_id: str):
        try:
            req = urllib.request.Request(
                f"{self.base_url}/next-data-chunk?worker_id={worker_id}",
                method="GET"
            )

            with urllib.request.urlopen(req, timeout=60) as res:
                # ✅ HTTP 204 = END OF STREAM
                if res.status == 204:
                    return None

                body = res.read()
                if not body:
                    return None

                return json.loads(body.decode())

        except urllib.error.HTTPError as e:
            if e.code == 204:
                return None
            return None
        except Exception:
            return None

    def send_data_ready(self, worker_id: str):
        self._post("/data-ready", {"worker_id": worker_id})

    def send_gradients(self, payload: dict):
        try:
            self._post("/submit-gradients", payload)
        except Exception:
            # Coordinator may already be shutting down
            print(f"[Worker] Coordinator unreachable while sending gradients.")

    def _post(self, endpoint: str, payload: dict):
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=60)
