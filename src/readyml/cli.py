import argparse
import json
import sys

from . import check, gpus, hub, runtimes, trace

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


def cmd_check(args):
    try:
        gpu = gpus.lookup(args.gpu)
        rt = runtimes.load(args.runtime)
        model = args.model.removeprefix("hf://")
        info = hub.fetch_model(model, args.revision)
    except (KeyError, hub.HubError, runtimes.RuntimeSpecError) as e:
        sys.exit(f"readyml: {e.args[0]}")
    config = hub.fetch_config(model, args.revision)
    result = check.run(info, rt, gpu, args.gpus, bool(args.revision), config, args.max_model_len)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['verdict']}  {info['id']} on {rt['label']} {rt['version']}, {args.gpus} x {gpu['name']}\n")
        for f in result["findings"]:
            print(f"  {MARK[f['level']]}  {f['check']:<13} {f['message']}")
    return 1 if result["verdict"] == "NOT READY" else 0


def cmd_versions(args):
    try:
        info = hub.fetch_model(args.model.removeprefix("hf://"), args.revision)
    except hub.HubError as e:
        sys.exit(f"readyml: {e.args[0]}")
    archs = (info.get("config") or {}).get("architectures") or []
    if not archs:
        sys.exit(f"readyml: {info['id']} names no architecture in config.json")
    result = {arch: runtimes.support(arch) for arch in archs}
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    for arch, by_runtime in result.items():
        print(f"{info['id']}  ({arch})\n")
        for label, r in by_runtime.items():
            seen = r["seen"]
            if not seen:
                print(f"  {label:<7} no release from {r['oldest']} to {r['latest']} loads it natively")
            else:
                first = f"{seen['first']} or earlier" if seen["first"] == r["oldest"] else seen["first"]
                if seen["last"] == r["latest"]:
                    print(f"  {label:<7} {first} and later (checked up to {r['latest']})")
                else:
                    print(f"  {label:<7} {first} to {seen['last']}, then removed")
    return 0


def cmd_trace(args):
    with (sys.stdin if args.log == "-" else open(args.log, errors="replace")) as f:
        result = trace.parse(f)
    print(json.dumps(result, indent=2) if args.json else trace.render(result))
    return 0 if result and result["ready"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="readyml", description="Preflight check for self-hosted inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check a model against a runtime release and a GPU, without launching anything")
    c.add_argument("--model", required=True, help="Hugging Face model id")
    c.add_argument("--revision", help="commit SHA, branch or tag to check")
    c.add_argument("--runtime", required=True, help="vllm:<version> or sglang:<version>")
    c.add_argument("--gpu", required=True, help="GPU type, for example h100, a100-80, l4")
    c.add_argument("--gpus", type=int, default=1, help="number of GPUs (default 1)")
    c.add_argument("--max-model-len", type=int, help="context length you will serve (default: the model's maximum)")
    c.add_argument("--json", action="store_true", help="print JSON")
    c.set_defaults(func=cmd_check)
    v = sub.add_parser("versions", help="list which vLLM and SGLang releases load a model's architecture natively")
    v.add_argument("--model", required=True, help="Hugging Face model id")
    v.add_argument("--revision", help="commit SHA, branch or tag")
    v.add_argument("--json", action="store_true", help="print JSON")
    v.set_defaults(func=cmd_versions)
    t = sub.add_parser("trace", help="explain where startup time went, from a vLLM server log")
    t.add_argument("log", help="path to a vLLM log file, or - for stdin (kubectl logs <pod> | readyml trace -)")
    t.add_argument("--json", action="store_true", help="print JSON")
    t.set_defaults(func=cmd_trace)
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
