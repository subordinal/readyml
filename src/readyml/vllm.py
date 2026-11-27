"""What a vLLM release supports, read from its source at the release tag."""
import ast
import json
import urllib.error
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/vllm-project/vllm/v{version}/{path}"
REGISTRY = "vllm/model_executor/models/registry.py"
QUANT = "vllm/model_executor/layers/quantization/__init__.py"
MATRIX_DOCS = "https://docs.vllm.ai/en/v{version}/features/quantization/index.html"

# Quantization support by GPU family, from the vLLM v0.34.0 docs. No release
# publishes this in machine-readable form, so it is transcribed by hand.
MATRIX_VERSION = "0.34.0"
FAMILIES = ("volta", "turing", "ampere", "ada", "hopper")
MATRIX = {
    "awq": (False, True, True, True, True),
    "gptq": (True, True, True, True, True),
    "int8_w8a8": (False, True, True, True, True),
    "fp8_w8a8": (False, False, False, True, True),
    "bitsandbytes": (True, True, True, True, True),
    "gguf": (True, True, True, True, True),
}


class RuntimeSourceError(Exception):
    pass


def version_key(version):
    return tuple(int(part) for part in version.split("."))


def support_index():
    path = Path(__file__).parent / "data" / "vllm_support.json"
    return json.loads(path.read_text()) if path.exists() else None


def fetch_source(version, path):
    url = RAW.format(version=version, path=path)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeSourceError(f"no vLLM release tagged v{version} ({e.code} for {path})") from e
    except urllib.error.URLError as e:
        raise RuntimeSourceError(f"cannot reach GitHub: {e.reason}") from e


def _module_dicts(source):
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
            out[target.id] = value
    return out


def parse_registry(source):
    """Architectures the release loads natively, through Transformers, or dropped."""
    dicts = _module_dicts(source)

    def keys(name):
        return {k.value for k in dicts[name].keys if isinstance(k, ast.Constant)}

    def str_values(name):
        d = dicts.get(name)
        if d is None:
            return {}
        return {
            k.value: v.value
            for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
        }

    if "_VLLM_MODELS" not in dicts:
        raise RuntimeSourceError("registry layout not recognised: no _VLLM_MODELS")
    merged = [k for k, v in zip(dicts["_VLLM_MODELS"].keys, dicts["_VLLM_MODELS"].values) if k is None]
    if not merged:
        names = ["_VLLM_MODELS"]
    else:
        names = [v.id for v in dicts["_VLLM_MODELS"].values if isinstance(v, ast.Name)]
    native, via_transformers = set(), set()
    for name in names:
        (via_transformers if name.startswith("_TRANSFORMERS") else native).update(keys(name))
    return {
        "native": native,
        "transformers": via_transformers,
        "removed": str_values("_PREVIOUSLY_SUPPORTED_MODELS"),
        "plugin": str_values("_OOT_SUPPORTED_MODELS"),
    }


def parse_quant_methods(source):
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "QuantizationMethods"
            and isinstance(node.value, ast.Subscript)
        ):
            sl = node.value.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            return {e.value for e in elts if isinstance(e, ast.Constant)}
    raise RuntimeSourceError("quantization layout not recognised: no QuantizationMethods")
