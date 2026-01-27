# common/serialization.py
import pickle
import base64


def serialize(obj) -> str:
    """
    Serialize Python object to base64 string (JSON-safe).
    """
    raw_bytes = pickle.dumps(obj)
    return base64.b64encode(raw_bytes).decode("utf-8")


def deserialize(blob):
    """
    Deserialize from base64 string or raw bytes.
    """
    if isinstance(blob, str):
        blob = base64.b64decode(blob.encode("utf-8"))
    return pickle.loads(blob)
