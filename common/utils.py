# common/utils.py
import platform
import psutil


def get_machine_info():
    """
    Collects basic hardware info for power scoring.
    """
    return {
        "os": platform.system(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
