"""Optional, non-allocating accelerator probe; CPU is always the safe default."""

import importlib.util


def inspect_hardware():
    result = {"selected_device": "cpu", "cuda_detected": False, "cuda_usable": False}
    if importlib.util.find_spec("torch") is None:
        result["note"] = "PyTorch is not installed; CPU fallback."
        return result

    try:
        import torch

        result["pytorch_version"] = torch.__version__
        result["cuda_build"] = torch.version.cuda
        if not torch.cuda.is_available():
            result["note"] = "CUDA unavailable in this PyTorch environment; CPU fallback."
            return result

        result["cuda_detected"] = True
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        result.update({
            "gpu_name": properties.name,
            "compute_capability": f"{major}.{minor}",
            "total_memory_bytes": properties.total_memory,
        })
        try:
            free, total = torch.cuda.mem_get_info(index)
            result.update({"free_memory_bytes": free, "total_memory_bytes": total})
        except (AttributeError, RuntimeError):
            pass
        arch = f"sm_{major}{minor}"
        supported_arches = torch.cuda.get_arch_list()
        result["cuda_usable"] = arch in supported_arches
        result["note"] = (
            "CUDA detected; architecture appears in this PyTorch build. "
            "CPU remains selected until a workload verifies the device."
            if result["cuda_usable"] else
            f"CUDA detected, but {arch} is not listed by this PyTorch build; CPU fallback."
        )
    except Exception as exc:
        result["note"] = f"Accelerator probe failed ({type(exc).__name__}); CPU fallback."
    return result
