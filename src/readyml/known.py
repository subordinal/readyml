"""Match a planned launch against known startup failures reported upstream."""
import json
import re
from pathlib import Path

from .runtimes import version_key

DATA = Path(__file__).parent / "data" / "known_issues.json"
BOUND = re.compile(r"(>=|<=|>|<|==)?\s*v?(\d+\.\d+(?:\.\d+)?)")


def load():
    return json.loads(DATA.read_text()) if DATA.exists() else []


def _key(version):
    m = re.match(r"\d+(?:\.\d+)*", version)
    return version_key(m.group(0)) + (0,) * (3 - m.group(0).count(".") - 1) if m else None


def version_matches(spec, version):
    """True, False, or None when the issue's wording cannot be read as a version range."""
    have = _key(version)
    bounds = BOUND.findall(spec or "")
    if not bounds or have is None:
        return None
    if len(bounds) == 2 and not bounds[0][0] and not bounds[1][0] and re.search(r"\d\s*(?:-|to)\s*v?\d", spec):
        return _key(bounds[0][1]) <= have <= _key(bounds[1][1])
    results = []
    for op, v in bounds:
        want = _key(v)
        results.append({">=": have >= want, "<=": have <= want, ">": have > want, "<": have < want}.get(op, have[: len(v.split("."))] == want[: len(v.split("."))]))
    return all(results) if any(op for op, _ in bounds) else any(results)


def model_quantization(info):
    """Quantization tags for a model, in the vocabulary the records use."""
    config = info.get("config") or {}
    qc = config.get("quantization_config") or {}
    tags = set()
    method = (qc.get("quant_method") or "").lower()
    if method:
        tags.add(method)
    if qc.get("load_in_4bit") or qc.get("load_in_8bit"):
        tags.add("bitsandbytes")
    if any(s["rfilename"].endswith(".gguf") for s in info.get("siblings", [])):
        tags.add("gguf")
    dtypes = " ".join((info.get("safetensors") or {}).get("parameters") or {}).upper()
    text = (json.dumps(qc) + " " + info.get("id", "")).lower()
    if "F8_" in dtypes or "fp8" in text:
        tags.add("fp8")
    for name in ("nvfp4", "mxfp4", "awq", "gptq"):
        if name in text:
            tags.add(name)
    if "I8" in dtypes.split() or "int8" in text or "w8a8" in text and "fp8" not in tags:
        tags.add("int8")
    return tags


def match(records, info, rt, gpu):
    archs = set((info.get("config") or {}).get("architectures") or [])
    model_id = info.get("id", "").lower()
    quant = model_quantization(info)
    hits = []
    for r in records:
        if r["runtime"] != rt["name"]:
            continue
        by_model = model_id in {m.lower() for m in r.get("models") or []}
        by_arch = bool(archs & set(r.get("architectures") or []))
        by_quant = bool(quant & set(r.get("quantization") or []))
        if r.get("models") or r.get("architectures"):
            if not (by_model or by_arch):
                continue
        elif not by_quant:
            continue
        if r.get("quantization") and not by_quant:
            continue
        if r.get("gpu_families") and gpu["family"] not in r["gpu_families"]:
            continue
        versions = r.get("versions") or {}
        fixed = versions.get("fixed_in")
        if fixed and _key(fixed) and _key(rt["version"]) and _key(rt["version"]) >= _key(fixed):
            continue
        affected = version_matches(versions.get("affected"), rt["version"])
        if affected is False:
            continue
        hits.append({"record": r, "version_confirmed": affected is True, "exact_model": by_model})
    return hits
