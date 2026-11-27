"""Build src/readyml/data/vllm_support.json: per architecture, the vLLM releases that load it natively.

Releases whose registry is missing or laid out differently are listed under "skipped".

Run from the repo root: PYTHONPATH=src python3 tools/build_support.py
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from readyml import vllm

OUT = Path("src/readyml/data/vllm_support.json")


def release_versions():
    raw = subprocess.run(
        ["gh", "api", "--paginate", "repos/vllm-project/vllm/releases", "--jq", ".[].tag_name"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    found = [t[1:] for t in raw if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return sorted(found, key=vllm.version_key)


def parse(version):
    try:
        return version, vllm.parse_registry(vllm.fetch_source(version, vllm.REGISTRY))
    except vllm.RuntimeSourceError as e:
        return version, str(e)


def main():
    versions = release_versions()
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(parse, versions))
    parsed, skipped, archs = [], [], {}
    for version, registry in results:
        if isinstance(registry, str):
            skipped.append(version)
            continue
        parsed.append(version)
        for arch in registry["native"]:
            archs.setdefault(arch, []).append(version)
    index = {
        "versions": parsed,
        "skipped": skipped,
        "native": {a: {"first": v[0], "last": v[-1], "count": len(v)} for a, v in sorted(archs.items())},
    }
    OUT.write_text(json.dumps(index, indent=1) + "\n")
    print(f"{len(parsed)} releases parsed, {len(skipped)} skipped, {len(archs)} architectures", file=sys.stderr)


if __name__ == "__main__":
    main()
