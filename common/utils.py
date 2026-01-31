# common/utils.py

import psutil
import torch
import subprocess
import re


def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "has_gpu": False,
            "gpu_name": None,
            "vram_gb": 0,
            "gpu_type": "NONE"
        }

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
            "gpu_type": gpu_type
        }

    except Exception:
        return {
            "has_gpu": True,
            "gpu_name": "Unknown GPU",
            "vram_gb": 0,
            "gpu_type": "CUDA"
        }


def get_machine_power():
    # ===== CPU =====
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    cpu_ghz = round(cpu_freq.max / 1000, 2) if cpu_freq else 2.0
    cpu_score = cpu_cores * cpu_ghz

    # ===== RAM =====
    ram = psutil.virtual_memory()
    available_ram_gb = round(ram.available / (1024 ** 3), 2)
    ram_score = available_ram_gb * 0.5

    # ===== GPU =====
    gpu = get_gpu_info()
    gpu_score = 0

    if gpu["has_gpu"]:
        if gpu["gpu_type"] == "RTX":
            gpu_score = 20 + gpu["vram_gb"] * 3
        elif gpu["gpu_type"] == "GTX":
            gpu_score = 10 + gpu["vram_gb"] * 2
        else:
            gpu_score = 8 + gpu["vram_gb"] * 1.5

    total_power = round(cpu_score + ram_score + gpu_score, 2)

    return {
        "cpu_cores": cpu_cores,
        "cpu_ghz": cpu_ghz,
        "available_ram_gb": available_ram_gb,
        "gpu": gpu,
        "scores": {
            "cpu": round(cpu_score, 2),
            "ram": round(ram_score, 2),
            "gpu": round(gpu_score, 2),
            "total": total_power
        }
    }
