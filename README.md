# readyml

Will this model start on this runtime and this GPU? `readyml` answers before you launch anything, and explains a slow start after.

It downloads no weights and needs no GPU. It has no dependencies outside the Python standard library.

```
pip install git+https://github.com/subordinal/readyml
```

## check: will it start?

```
$ readyml check --model Qwen/Qwen3-8B --runtime vllm:0.8.0 --gpu l4

NOT READY  Qwen/Qwen3-8B on vLLM 0.8.0, 1 x l4

  FAIL  architecture  Qwen3ForCausalLM is not in vLLM 0.8.0; native support starts in 0.8.4
  ok    quantization  model is not quantized
  ok    memory        15.3 GiB of weights on 1 x l4 (24 GiB): 64% used by weights
  ok    context       KV cache for 40,960 tokens needs about 5.6 GiB; about 6.3 GiB is left after weights, ...
  info  artifact      download is 15.3 GiB per replica with no shared cache
  WARN  artifact      revision is not pinned; pin to b968826d9c46 so every rollout loads the same weights
```

Exit code is 1 when the verdict is NOT READY, so it can gate CI. `--json` prints the findings as JSON.

| Check | What it reads |
|---|---|
| Architecture is supported in that release, and which release added or dropped it | vLLM's model registry parsed at each release tag (55 releases, 0.6.3 to 0.34.0); SGLang's model files at each tag (67 releases, 0.1.10 to 0.5.23) |
| Quantization works on that GPU family | the quantization table in the vLLM v0.34.0 docs (vLLM only) |
| Weights fit in GPU memory | file sizes from the Hugging Face Hub |
| Context length fits | a KV cache estimate from config.json; `--max-model-len` sets the length |
| Known startup failures that match this model, release and GPU | 68 records from vLLM, SGLang and KServe issues, in `src/readyml/data/known_issues.json` |
| Repo is complete and the revision is pinned | the Hugging Face Hub |

Runtimes: `vllm:<version>` and `sglang:<version>`. GPUs: `t4 l4 a10g a100-40 a100-80 l40s h100 h200 b200`. Use `--gpus N` for more than one.

### Known issues

Most startup failures are not configuration mistakes. They are engine bugs that repeat exactly for a given model, release, GPU and set of flags. `check` matches your launch against failures already reported upstream:

```
$ readyml check --model Qwen/Qwen3.8-Flash-Next-FP8 --runtime vllm:0.34.0 --gpu a100-80 --gpus 4
  ...
  WARN  known issue   The Triton fused-MoE kernel is selected without a compute-capability check and uses
                      the fp8e4nv dtype, which SM80 does not support. ... Workaround: Pass --moe-backend marlin
                      https://github.com/vllm-project/vllm/issues/54318
```

Each record names the issue it came from. A record matches only when it names your model or its architecture. Records were extracted from issue text and 16 of the 68 are marked medium confidence; only two have a confirmed fixed-in release. Corrections and new records are welcome as pull requests to the JSON file.

## versions: which release do I need?

```
$ readyml versions --model Qwen/Qwen-7B-Chat

Qwen/Qwen-7B-Chat  (QWenLMHeadModel)

  vLLM    0.6.3 or earlier to 0.23.0, then removed
  SGLang  every indexed release (0.1.10 to 0.5.23)
```

## trace: where did startup time go?

```
$ kubectl logs my-vllm-pod | readyml trace -

READY after 50m 44s  (vLLM 0.34.0rc1.dev398+g153dc0464)

  process start        18s    1%  imports and argument parsing, before the engine starts
  engine start          7s    0%  engine process spawn and device init
  model load            4s    0%  download, model init and weight load
  engine init      50m 05s   99%  compile, warmup, CUDA graph capture and KV cache allocation
  server setup         10s    0%  API server start
```

It reads the timestamps and the durations vLLM logs itself, and adds a hint when one cause dominates. This example is a real log from a vLLM issue and is one of the test fixtures.

## Limits

- It reads configuration and logs. It cannot see engine bugs that are not in its known-issues file.
- The quantization table comes from one vLLM release and has no Blackwell column. SGLang quantization is not checked.
- The KV cache figure is an estimate from the standard attention formula. It is not calibrated against measured runs, and it is skipped for models whose config does not describe a standard attention stack.
- `trace` reads vLLM logs only, and has been run against one real log so far. Phases vLLM does not log separately (distributed init, download with streaming loaders) are not split out.
- Private and gated models need `HF_TOKEN`.

## Develop

```
PYTHONPATH=src python -m unittest discover -s tests
```

The release indexes are rebuilt with `tools/build_support.py` (vLLM) and `tools/build_sglang_support.py` (SGLang).

Apache-2.0. Built by [Subordinal](https://github.com/subordinal). Early development.
