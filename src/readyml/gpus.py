"""Datacenter GPUs: memory in GiB, architecture family, CUDA compute capability.

Compute capability is from developer.nvidia.com/cuda/gpus. Memory is the
advertised figure; usable memory is a little lower.
"""
GPUS = {
    "t4": (16, "turing", "7.5"),
    "l4": (24, "ada", "8.9"),
    "a10g": (24, "ampere", "8.6"),
    "a100-40": (40, "ampere", "8.0"),
    "a100-80": (80, "ampere", "8.0"),
    "l40s": (48, "ada", "8.9"),
    "h100": (80, "hopper", "9.0"),
    "h200": (141, "hopper", "9.0"),
    "b200": (180, "blackwell", "10.0"),
}
ALIASES = {"a10": "a10g", "a100": "a100-80"}


def lookup(name):
    key = ALIASES.get(name.lower(), name.lower())
    if key not in GPUS:
        raise KeyError(f"unknown GPU '{name}'. Known: {', '.join(sorted(GPUS))}")
    vram, family, capability = GPUS[key]
    return {"name": key, "vram_gib": vram, "family": family, "capability": capability}
