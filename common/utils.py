import platform
import os


def get_machine_info():
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "cpu_count": os.cpu_count(),
    }
