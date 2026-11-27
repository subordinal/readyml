import argparse
import json
import sys

from . import check, gpus, hub, vllm

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


def cmd_check(args):
    runtime, _, version = args.runtime.partition(":")
    if runtime != "vllm" or not version:
        sys.exit("--runtime must be vllm:<version>, for example vllm:0.34.0. Other runtimes are not supported yet.")
    try:
        gpu = gpus.lookup(args.gpu)
        info = hub.fetch_model(args.model.removeprefix("hf://"), args.revision)
        registry = vllm.parse_registry(vllm.fetch_source(version, vllm.REGISTRY))
        methods = vllm.parse_quant_methods(vllm.fetch_source(version, vllm.QUANT))
    except (KeyError, hub.HubError, vllm.RuntimeSourceError) as e:
        sys.exit(f"readyml: {e.args[0]}")
    result = check.run(info, registry, methods, gpu, args.gpus, version, pinned=bool(args.revision))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['verdict']}  {info['id']} on vllm {version}, {args.gpus} x {gpu['name']}\n")
        for f in result["findings"]:
            print(f"  {MARK[f['level']]}  {f['check']:<13} {f['message']}")
    return 1 if result["verdict"] == "NOT READY" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="readyml", description="Preflight check for self-hosted inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a model against a runtime release and a GPU, without launching anything")
    c.add_argument("--model", required=True, help="Hugging Face model id")
    c.add_argument("--revision", help="commit SHA, branch or tag to check")
    c.add_argument("--runtime", required=True, help="vllm:<version>")
    c.add_argument("--gpu", required=True, help="GPU type, for example h100, a100-80, l4")
    c.add_argument("--gpus", type=int, default=1, help="number of GPUs (default 1)")
    c.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    sys.exit(cmd_check(args))


if __name__ == "__main__":
    main()
