"""Resolve `name:version` into what that runtime release supports."""
import json
from pathlib import Path

from . import vllm

DATA = Path(__file__).parent / "data"


class RuntimeSpecError(Exception):
    pass


def version_key(version):
    return tuple(int(part) for part in version.split("."))


def _index(name):
    path = DATA / f"{name}_support.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_vllm(version, fetch=vllm.fetch_source):
    return {
        "name": "vllm",
        "label": "vLLM",
        "version": version,
        "source": vllm.RAW.format(version=version, path=vllm.REGISTRY),
        "registry": vllm.parse_registry(fetch(version, vllm.REGISTRY)),
        "methods": vllm.parse_quant_methods(fetch(version, vllm.QUANT)),
        "fallback": True,
        "index": _index("vllm"),
    }


def load_sglang(version):
    index = _index("sglang")
    archs = (index or {}).get("by_version", {}).get(version)
    if archs is None:
        known = (index or {}).get("versions") or []
        raise RuntimeSpecError(f"SGLang {version} is not in the bundled index ({known[0]} to {known[-1]})" if known else "no SGLang index bundled")
    fallback = {a for a in archs if a.startswith("Transformers")}
    return {
        "name": "sglang",
        "label": "SGLang",
        "version": version,
        "source": f"https://github.com/sgl-project/sglang/tree/v{version}/python/sglang/srt/models",
        "registry": {"native": set(archs) - fallback, "transformers": set(), "removed": {}, "plugin": {}},
        "methods": None,
        "fallback": bool(fallback),
        "index": index,
    }


def load(spec):
    name, _, version = spec.partition(":")
    if not version or name not in ("vllm", "sglang"):
        raise RuntimeSpecError("--runtime must be vllm:<version> or sglang:<version>, for example vllm:0.34.0")
    try:
        return load_vllm(version) if name == "vllm" else load_sglang(version)
    except vllm.RuntimeSourceError as e:
        raise RuntimeSpecError(str(e)) from e


def support(arch):
    """First and last release of each runtime that loads `arch` natively, from the bundled indexes."""
    out = {}
    for name, label in (("vllm", "vLLM"), ("sglang", "SGLang")):
        index = _index(name) or {}
        seen = (index.get("native") or {}).get(arch)
        out[label] = {"seen": seen, "latest": (index.get("versions") or [None])[-1], "oldest": (index.get("versions") or [None])[0]}
    return out
