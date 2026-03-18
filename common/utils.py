import psutil
import platform
import torch


def get_gpu_info():
    """
    Detect GPU across:
    - NVIDIA (CUDA)
    - Apple Silicon (MPS)
    - CPU-only systems
    """

    system = platform.system()

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)

            name_upper = gpu_name.upper()

            if "RTX" in name_upper:
                gpu_type = "RTX"
            elif "GTX" in name_upper:
                gpu_type = "GTX"
            else:
                gpu_type = "CUDA"

            return {
                "has_gpu": True,
                "gpu_name": gpu_name,
                "vram_gb": vram_gb,
                "gpu_type": gpu_type,
                "backend": "CUDA"
            }
        except Exception:
            pass

    # Apple Silicon GPU (MPS)

    if system == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            chip = platform.processor() or "Apple Silicon"
            return {
                "has_gpu": True,
                "gpu_name": chip,
                "vram_gb": 0,
                "gpu_type": "APPLE",
                "backend": "MPS"
            }

    # No GPU

    return {
        "has_gpu": False,
        "gpu_name": None,
        "vram_gb": 0,
        "gpu_type": "NONE",
        "backend": "CPU"
    }


# MACHINE POWER SCORING

def get_machine_power():
    """
    Cross-platform machine power score for
    power-aware distributed ML.
    """

    # CPU
    cpu_cores = psutil.cpu_count(logical=True) or 4
    cpu_freq = psutil.cpu_freq()
    cpu_ghz = round(cpu_freq.max / 1000, 2) if cpu_freq else 2.0

    cpu_brand = platform.processor().lower()
    cpu_multiplier = 1.4 if (
        "apple" in cpu_brand or "arm" in cpu_brand) else 1.0

    cpu_score = cpu_cores * cpu_ghz * cpu_multiplier

    # RAM

    ram = psutil.virtual_memory()
    total_ram_gb = round(ram.total / (1024 ** 3), 2)
    available_ram_gb = round(ram.available / (1024 ** 3), 2)

    ram_score = available_ram_gb * 0.4

    # GPU
    gpu = get_gpu_info()
    gpu_score = 0.0

    if gpu["has_gpu"]:
        if gpu["gpu_type"] == "RTX":
            gpu_score = 25 + gpu["vram_gb"] * 3.5
        elif gpu["gpu_type"] == "GTX":
            gpu_score = 15 + gpu["vram_gb"] * 2.5
        elif gpu["gpu_type"] == "APPLE":
            gpu_score = 18
        else:
            gpu_score = 12 + gpu["vram_gb"] * 2.0

    # FINAL

    total_power = round(cpu_score + ram_score + gpu_score, 2)

    return {
        "platform": platform.system(),
        "cpu": {
            "cores": cpu_cores,
            "max_ghz": cpu_ghz,
            "score": round(cpu_score, 2)
        },
        "ram": {
            "total_gb": total_ram_gb,
            "available_gb": available_ram_gb,
            "score": round(ram_score, 2)
        },
        "gpu": gpu,
        "scores": {
            "cpu": round(cpu_score, 2),
            "ram": round(ram_score, 2),
            "gpu": round(gpu_score, 2),
            "total": total_power
        }
    }
