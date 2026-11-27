# readyml

Will this model start on this runtime and this GPU? `readyml check` answers before you launch anything.

```
$ readyml check --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 --runtime vllm:0.34.0 --gpu a100-80

READY WITH WARNINGS  RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 on vllm 0.34.0, 1 x a100-80

  ok    architecture  LlamaForCausalLM is in the vLLM 0.34.0 model registry
  WARN  quantization  FP8 (compressed-tensors): vLLM lists FP8 W8A8 as unsupported on ampere GPUs. ...
  ok    memory        8.5 GiB of weights on 1 x a100-80 (80 GiB): 11% used by weights. ...
  info  artifact      download is 8.5 GiB per replica with no shared cache
  WARN  artifact      revision is not pinned; pin to 870a1177ed08 so every rollout loads the same weights
```

It downloads no weights and needs no GPU. Exit code is 1 when the verdict is NOT READY, so it can gate CI.

## What it checks today

| Check | Source |
|---|---|
| Architecture is supported in that vLLM release | the model registry in vLLM's source at the release tag |
| Quantization works on that GPU family | the quantization matrix in the vLLM v0.34.0 docs |
| Weights fit in GPU memory | file sizes from the Hugging Face Hub |
| Repo has config and tokenizer files, revision is pinned | the Hugging Face Hub |

## What it does not do yet

- KV cache and context length are not part of the memory check.
- The quantization matrix is transcribed from one vLLM release (0.34.0) and has no Blackwell column.
- Only vLLM. SGLang and KServe are planned.
- It reads configuration only. It cannot see engine bugs or anything that depends on the running cluster.

## Install and test

```
pip install -e .
PYTHONPATH=src python -m unittest discover -s tests
```

No dependencies outside the Python standard library. Set `HF_TOKEN` for gated models.

Status: early development.
