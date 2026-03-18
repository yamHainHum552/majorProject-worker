import json
import urllib.request
import urllib.error
import time
import gzip
import base64


class WorkerCommunicator:

    def __init__(self, coordinator_host, coordinator_port):
        self.base_url = f"http://{coordinator_host}:{coordinator_port}"

    # ------------------------------------------------
    # REGISTER WORKER
    # ------------------------------------------------
    def register_worker(self, payload: dict):
        self._post("/register", payload)

    # ------------------------------------------------
    # POLL COMMAND
    # ------------------------------------------------
    def wait_for_command(self, worker_id: str):

        try:
            with urllib.request.urlopen(
                f"{self.base_url}/next-command?worker_id={worker_id}",
                timeout=60,
            ) as res:

                body = res.read()

                if not body:
                    return {"cmd": "WAIT"}

                return json.loads(body.decode())

        except Exception:
            # Coordinator may be busy aggregating
            time.sleep(2)
            return {"cmd": "WAIT"}

    # ------------------------------------------------
    # DATA STREAM
    # ------------------------------------------------
    def fetch_data_chunk(self, worker_id: str):

        try:

            req = urllib.request.Request(
                f"{self.base_url}/next-data-chunk?worker_id={worker_id}",
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=60) as res:

                if res.status == 204:
                    return None

                body = res.read()

                if not body:
                    return None

                return json.loads(body.decode())

        except urllib.error.HTTPError as e:

            if e.code == 204:
                return None

            print("[Worker] Data stream HTTP error:", e)
            return None

        except Exception as e:
            print("[Worker] Data stream error:", e)
            return None

    # ------------------------------------------------
    # DATA READY
    # ------------------------------------------------
    def send_data_ready(self, worker_id: str):
        self._post("/data-ready", {"worker_id": worker_id})

    # ------------------------------------------------
    # SEND MODEL UPDATE (COMPRESSED)
    # ------------------------------------------------
    def send_gradients(self, payload: dict):

        retries = 3

        for attempt in range(retries):

            try:

                # compress serialized weights
                raw = payload["weights"].encode()
                compressed = gzip.compress(raw)

                payload["weights"] = base64.b64encode(compressed).decode()

                self._post("/submit-gradients", payload)

                return

            except Exception as e:

                print(
                    f"[Worker] Send attempt {attempt+1}/{retries} failed:",
                    e
                )

                time.sleep(3)

        raise RuntimeError("Failed to send model update")

    # ------------------------------------------------
    # GENERIC POST
    # ------------------------------------------------
    def _post(self, endpoint: str, payload: dict):

        data = json.dumps(payload).encode()

        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=600) as res:

            if res.status != 200:
                raise RuntimeError(
                    f"Coordinator returned status {res.status}"
                )
