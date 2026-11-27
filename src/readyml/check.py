"""Static preflight: findings from model metadata, a runtime release and a GPU."""
from . import vllm
from .hub import weight_files
from .runtimes import version_key

GIB = 1024**3
PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"
KV_BUDGET = 0.9  # share of GPU memory assumed usable; vLLM's long-standing gpu_memory_utilization default
TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "tokenizer_config.json"}


def finding(level, check, message, source=None):
    return {"level": level, "check": check, "message": message, "source": source}


def quant_row(info):
    """Map a model's quantization to a matrix row. Returns (row, label, certain)."""
    config = info.get("config") or {}
    qc = config.get("quantization_config") or {}
    method = (qc.get("quant_method") or "").lower()
    if any(s["rfilename"].endswith(".gguf") for s in info.get("siblings", [])) and not method:
        return "gguf", "GGUF", True
    if not qc:
        return None, None, True
    if method in ("awq", "gptq", "bitsandbytes"):
        return method, method, True
    if qc.get("load_in_4bit") or qc.get("load_in_8bit"):
        return "bitsandbytes", "bitsandbytes", True
    if method == "compressed-tensors":
        groups = list((qc.get("config_groups") or {}).values())
        weights = [g.get("weights") or {} for g in groups]
        acts = [g.get("input_activations") for g in groups]
        bits = {w.get("num_bits") for w in weights}
        dtypes = (info.get("safetensors") or {}).get("parameters") or {}
        is_float = any(w.get("type") == "float" for w in weights) or any(k.startswith("F8") for k in dtypes)
        if bits == {8} and is_float:
            return "fp8_w8a8", "FP8 (compressed-tensors)", all(acts)
        if bits == {8} and all(acts):
            return "int8_w8a8", "INT8 W8A8 (compressed-tensors)", True
    return "unknown", method or "unrecognised", False


def check_architecture(info, rt):
    archs = (info.get("config") or {}).get("architectures") or []
    registry, label, src = rt["registry"], f"{rt['label']} {rt['version']}", rt["source"]
    if not archs:
        return [finding(WARN, "architecture", "config.json names no architecture; cannot check support", src)]
    out = []
    for arch in archs:
        if arch in registry["native"]:
            out.append(finding(PASS, "architecture", f"{arch} is in the {label} model registry", src))
        elif arch in registry["transformers"]:
            out.append(finding(WARN, "architecture", f"{arch} runs through the Transformers backend in {label}, not a native implementation", src))
        elif arch in registry["removed"]:
            out.append(finding(FAIL, "architecture", f"{arch} was removed from {rt['label']}; last supported in {registry['removed'][arch]}", src))
        elif arch in registry["plugin"]:
            out.append(finding(WARN, "architecture", f"{arch} needs an out-of-tree plugin: {registry['plugin'][arch]}", src))
        else:
            out.append(_unlisted(arch, rt))
    return out


def _unlisted(arch, rt):
    label, version, src = rt["label"], rt["version"], rt["source"]
    seen = ((rt.get("index") or {}).get("native") or {}).get(arch)
    if seen and version_key(seen["first"]) > version_key(version):
        return finding(FAIL, "architecture", f"{arch} is not in {label} {version}; native support starts in {seen['first']}", src)
    if seen and version_key(seen["last"]) < version_key(version):
        return finding(FAIL, "architecture", f"{arch} is not in {label} {version}; last release with native support is {seen['last']}", src)
    if not rt["fallback"]:
        return finding(FAIL, "architecture", f"{arch} is not in the {label} {version} registry, and this release has no Transformers fallback", src)
    return finding(WARN, "architecture", f"{arch} is not in the {label} {version} registry; it will only load if the Transformers fallback can resolve it", src)


def check_quantization(info, rt, gpu):
    row, label, certain = quant_row(info)
    if row is None:
        return [finding(PASS, "quantization", "model is not quantized")]
    if rt["name"] != "vllm":
        return [finding(INFO, "quantization", f"{label}: quantization support on {rt['label']} is not checked yet")]
    version, methods = rt["version"], rt["methods"]
    docs = vllm.MATRIX_DOCS.format(version=vllm.MATRIX_VERSION)
    if row == "unknown":
        level = INFO if label in methods else WARN
        known = "is a vLLM quantization method" if label in methods else f"is not in vLLM {version}'s quantization methods"
        return [finding(level, "quantization", f"'{label}' {known}; hardware support for it is not checked yet")]
    if gpu["family"] not in vllm.FAMILIES:
        return [finding(WARN, "quantization", f"{label}: the vLLM docs matrix has no {gpu['family']} column, so support on {gpu['name']} is unknown", docs)]
    supported = vllm.MATRIX[row][vllm.FAMILIES.index(gpu["family"])]
    note = "" if version == vllm.MATRIX_VERSION else f" (matrix is from {vllm.MATRIX_VERSION}, not {version})"
    if supported:
        return [finding(PASS, "quantization", f"{label} is supported on {gpu['family']} GPUs{note}", docs)]
    if not certain:
        return [finding(WARN, "quantization", f"{label}: vLLM lists FP8 W8A8 as unsupported on {gpu['family']} GPUs. This model quantizes weights only, so it may load through another kernel path{note}", docs)]
    return [finding(FAIL, "quantization", f"{label} is listed as unsupported on {gpu['family']} GPUs ({gpu['name']}){note}", docs)]


def check_memory(info, gpu, gpu_count):
    files = weight_files(info)
    total = sum(size for _, size in files)
    if not total:
        return [finding(WARN, "memory", "no weight files with a known size; cannot check fit")]
    capacity = gpu["vram_gib"] * gpu_count * GIB
    share = total / capacity
    size = f"{total / GIB:.1f} GiB of weights on {gpu_count} x {gpu['name']} ({gpu['vram_gib'] * gpu_count} GiB)"
    if share > 1:
        return [finding(FAIL, "memory", f"{size}: the weights alone do not fit")]
    if share > 0.9:
        return [finding(WARN, "memory", f"{size}: {share:.0%} used by weights, little left for the KV cache")]
    return [finding(PASS, "memory", f"{size}: {share:.0%} used by weights")]


def kv_bytes_per_token(config):
    """KV cache bytes per token for a standard attention stack, or None when the config does not describe one."""
    c = config if config.get("num_hidden_layers") else config.get("text_config") or {}
    layers, heads, hidden = c.get("num_hidden_layers"), c.get("num_attention_heads"), c.get("hidden_size")
    if not (layers and heads and hidden):
        return None, None
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or hidden // heads
    width = 4 if (c.get("torch_dtype") or config.get("torch_dtype")) == "float32" else 2
    return 2 * layers * kv_heads * head_dim * width, c.get("max_position_embeddings")


def check_context(info, config, gpu, gpu_count, max_model_len):
    if not config:
        return []
    per_token, native_len = kv_bytes_per_token(config)
    length = max_model_len or native_len
    weights = sum(size for _, size in weight_files(info))
    if not (per_token and length and weights):
        return [finding(INFO, "context", "KV cache is not estimated: the config does not describe a standard attention stack")]
    budget = gpu["vram_gib"] * gpu_count * GIB * KV_BUDGET - weights
    need = length * per_token
    fits = int(budget // per_token) if budget > 0 else 0
    basis = f"estimate, {per_token / 1024:.0f} KiB per token, {KV_BUDGET:.0%} of GPU memory usable"
    if need > budget:
        return [finding(WARN, "context", f"one {length:,}-token sequence needs about {need / GIB:.1f} GiB of KV cache; about {max(budget, 0) / GIB:.1f} GiB is left after weights. Set --max-model-len to {fits:,} or less ({basis})")]
    return [finding(PASS, "context", f"KV cache for {length:,} tokens needs about {need / GIB:.1f} GiB; about {budget / GIB:.1f} GiB is left after weights, room for {fits:,} cached tokens ({basis})")]


def check_artifact(info, pinned):
    out = []
    files = {s["rfilename"] for s in info.get("siblings", [])}
    if "config.json" not in files and not any(f.endswith(".gguf") for f in files):
        out.append(finding(FAIL, "artifact", "repo has no config.json"))
    if not files & TOKENIZER_FILES and not any(f.endswith(".gguf") for f in files):
        out.append(finding(WARN, "artifact", "repo has no tokenizer files; the server may fail after the full load"))
    total = sum(size for _, size in weight_files(info))
    out.append(finding(INFO, "artifact", f"download is {total / GIB:.1f} GiB per replica with no shared cache"))
    if info.get("gated"):
        out.append(finding(INFO, "artifact", "repo is gated; every node needs a token with access"))
    if not pinned and info.get("sha"):
        out.append(finding(WARN, "artifact", f"revision is not pinned; pin to {info['sha'][:12]} so every rollout loads the same weights"))
    return out


def run(info, rt, gpu, gpu_count, pinned, config=None, max_model_len=None):
    findings = (
        check_architecture(info, rt)
        + check_quantization(info, rt, gpu)
        + check_memory(info, gpu, gpu_count)
        + check_context(info, config, gpu, gpu_count, max_model_len)
        + check_artifact(info, pinned)
    )
    levels = {f["level"] for f in findings}
    verdict = "NOT READY" if FAIL in levels else "READY WITH WARNINGS" if WARN in levels else "READY"
    return {"verdict": verdict, "findings": findings}
