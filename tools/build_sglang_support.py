"""Build src/readyml/data/sglang_support.json from a local SGLang clone.

SGLang registers every class named in a model file's module-level EntryClass,
so the supported architectures of a release are those names at its tag.

Run: python3 tools/build_sglang_support.py /path/to/sglang
(clone with: git clone --filter=blob:limit=300k --no-checkout https://github.com/sgl-project/sglang.git)
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("src/readyml/data/sglang_support.json")
MODELS = "python/sglang/srt/models"


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def entry_classes(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "EntryClass":
            value = node.value
            elts = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            return [e.id for e in elts if isinstance(e, ast.Name)]
    return []


def version_key(v):
    return tuple(int(p) for p in v.split("."))


def main(repo):
    tags = [t[1:] for t in git(repo, "tag").split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    tags.sort(key=version_key)
    versions, skipped, by_arch, per_version = [], [], {}, {}
    for v in tags:
        files = [f for f in git(repo, "ls-tree", "-r", "--name-only", f"v{v}", MODELS).split("\n") if f.endswith(".py")]
        archs = set()
        for f in files:
            archs.update(entry_classes(git(repo, "show", f"v{v}:{f}")))
        if not archs:
            skipped.append(v)
            continue
        versions.append(v)
        per_version[v] = sorted(archs)
        for a in archs:
            by_arch.setdefault(a, []).append(v)
    index = {
        "versions": versions,
        "skipped": skipped,
        "native": {a: {"first": vs[0], "last": vs[-1], "count": len(vs)} for a, vs in sorted(by_arch.items())},
        "by_version": per_version,
    }
    OUT.write_text(json.dumps(index, separators=(",", ":")) + "\n")
    print(f"{len(versions)} releases parsed, {len(skipped)} skipped, {len(by_arch)} architectures", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
