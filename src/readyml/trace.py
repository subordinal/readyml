"""Turn a vLLM startup log into a timeline: where the time between start and ready went."""
import re
from datetime import datetime

STAMP = re.compile(r"\b(?:INFO|WARNING|ERROR|DEBUG) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")

# Phase boundaries, in startup order: the first line matching each pattern starts that phase.
MARKERS = [
    ("engine start", re.compile(r"Initializing a V\d+ LLM engine")),
    ("model load", re.compile(r"Loading model from scratch|Starting to load model")),
    ("engine init", re.compile(r"Model loading took")),
    ("server setup", re.compile(r"init engine \(profile, create kv cache, warmup model\) took")),
    ("ready", re.compile(r"Application startup complete|Starting vLLM server on")),
]
PHASES = ["process start", "engine start", "model load", "engine init", "server setup"]
NOTES = {
    "process start": "imports and argument parsing, before the engine starts",
    "engine start": "engine process spawn and device init",
    "model load": "download, model init and weight load",
    "engine init": "compile, warmup, CUDA graph capture and KV cache allocation",
    "server setup": "API server start",
}

# Durations vLLM reports itself. Summed when a line repeats (one per rank or per graph).
REPORTED = {
    "download": re.compile(r"Time spent downloading weights for .*?: ([\d.]+) seconds"),
    "prefetch": re.compile(r"Prefetching checkpoint files into page cache finished in ([\d.]+)s"),
    "model init": re.compile(r"Initializing model took ([\d.]+) seconds"),
    "weight load": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "compile": re.compile(r"Compiling a graph for .*? takes ([\d.]+) s"),
    "compile cache hit": re.compile(r"Directly load the compiled graph.*?took ([\d.]+) s"),
    "warmup": re.compile(r"Initial profiling/warmup run took ([\d.]+) s"),
    "graph capture": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
}
VERSION = re.compile(r"LLM engine \(v([^)]+)\)")
MEMORY = re.compile(r"Model loading took ([\d.]+) GiB")


def _stamp(line):
    m = STAMP.search(line)
    if not m:
        return None
    month, day, hour, minute, second = map(int, m.groups())
    return datetime(2000, month, day, hour, minute, second)


def parse(lines):
    first = last = None
    starts, reported = {}, {}
    version = memory = None
    for line in lines:
        t = _stamp(line)
        if t:
            first = first or t
            last = t
        for name, pattern in MARKERS:
            if name not in starts and pattern.search(line) and last:
                starts[name] = last
        for name, pattern in REPORTED.items():
            m = pattern.search(line)
            if m:
                reported[name] = round(reported.get(name, 0) + float(m.group(1)), 2)
        version = version or (VERSION.search(line) or [None, None])[1]
        memory = memory or (MEMORY.search(line) or [None, None])[1]
    if first is None:
        return None
    starts["process start"] = first
    order = PHASES + ["ready"]
    phases = []
    for i, name in enumerate(PHASES):
        if name not in starts:
            continue
        end = next((starts[n] for n in order[i + 1:] if n in starts), None)
        if end is None:
            phases.append({"phase": name, "seconds": None, "note": NOTES[name]})
            break
        phases.append({"phase": name, "seconds": (end - starts[name]).total_seconds(), "note": NOTES[name]})
    ready = "ready" in starts
    # A log from an offline or embedded engine has no API server lines; the engine is up once init finishes.
    engine_ready = "server setup" in starts
    end = starts["ready"] if ready else starts["server setup"] if engine_ready else last
    total = (end - first).total_seconds()
    if engine_ready and not ready:
        phases = [p for p in phases if p["phase"] != "server setup"]
    return {
        "ready": ready,
        "engine_ready": engine_ready,
        "total_seconds": total,
        "version": version,
        "model_memory_gib": float(memory) if memory else None,
        "phases": phases,
        "reported": reported,
        "hints": hints(phases, reported, total, ready or engine_ready),
    }


def hints(phases, reported, total, ready):
    out = []
    if not ready:
        stuck = phases[-1]["phase"] if phases else "process start"
        out.append(f"The log ends before the server is ready. The last phase reached is '{stuck}'.")
    if total <= 0:
        return out
    by_name = {p["phase"]: p["seconds"] for p in phases if p["seconds"] is not None}
    compile_s = reported.get("compile", 0)
    if compile_s and "compile cache hit" not in reported and compile_s / total > 0.1:
        out.append(f"Graph compilation took {compile_s:.0f} s with no cache hit. Persist VLLM_CACHE_ROOT across restarts to reuse compiled graphs.")
    if reported.get("download", 0) / total > 0.2:
        out.append(f"Downloading weights took {reported['download']:.0f} s. Pre-stage the model on the node or a shared cache.")
    load = by_name.get("model load", 0)
    if load / total > 0.5 and reported.get("weight load", 0) / max(load, 1) > 0.5:
        out.append(f"Weight loading took {reported['weight load']:.0f} s. Check the storage the weights are read from; network filesystems are slow.")
    init = by_name.get("engine init", 0)
    explained = compile_s + reported.get("warmup", 0) + reported.get("graph capture", 0)
    if init / total > 0.5 and explained < 0.5 * init:
        out.append(f"Engine init took {init:.0f} s, but the log only accounts for {explained:.0f} s of it (compile, warmup, graph capture). The rest is not broken out at INFO level.")
    return out


def render(result):
    if result is None:
        return "no timestamped vLLM log lines found"
    total = result["total_seconds"]
    head = "READY" if result["ready"] else "ENGINE READY (no API server in this log)" if result["engine_ready"] else "NOT READY"
    lines = [f"{head} after {fmt(total)}" + (f"  (vLLM {result['version']})" if result["version"] else ""), ""]
    for p in result["phases"]:
        if p["seconds"] is None:
            lines.append(f"  {p['phase']:<14} {'did not finish':>9}        {p['note']}")
        else:
            share = p["seconds"] / total if total else 0
            lines.append(f"  {p['phase']:<14} {fmt(p['seconds']):>9}  {share:>4.0%}  {p['note']}")
    if result["reported"]:
        lines += ["", "  reported by vLLM:"]
        lines += [f"    {name:<18} {fmt(sec):>9}" for name, sec in result["reported"].items()]
    if result["hints"]:
        lines += [""] + [f"  - {h}" for h in result["hints"]]
    return "\n".join(lines)


def fmt(seconds):
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes}m {sec:02d}s" if minutes else f"{sec}s"
