import copy
import json
import unittest
from pathlib import Path

from readyml import check, gpus, runtimes, vllm

FIX = Path(__file__).parent / "fixtures"
REGISTRY = vllm.parse_registry((FIX / "vllm-0.34.0-registry.py").read_text())
METHODS = vllm.parse_quant_methods((FIX / "vllm-0.34.0-quantization.py").read_text())
MODEL = json.loads((FIX / "llama-3.1-8b-fp8.json").read_text())


def fixture_source(version, path):
    name = "registry" if path == vllm.REGISTRY else "quantization"
    return (FIX / f"vllm-0.34.0-{name}.py").read_text()


RT = runtimes.load_vllm("0.34.0", fetch=fixture_source)


def run(model=MODEL, gpu="h100", count=1, pinned=True, rt=RT):
    return check.run(model, rt, gpus.lookup(gpu), count, pinned)


def levels(result, name):
    return [f["level"] for f in result["findings"] if f["check"] == name]


class ParseTest(unittest.TestCase):
    def test_registry_sets(self):
        self.assertIn("LlamaForCausalLM", REGISTRY["native"])
        self.assertIn("GemmaForCausalLM", REGISTRY["transformers"])
        self.assertNotIn("GemmaForCausalLM", REGISTRY["native"])
        self.assertEqual(REGISTRY["removed"]["QWenLMHeadModel"], "0.23.0")
        self.assertIn("BartModel", REGISTRY["plugin"])

    def test_quant_methods(self):
        self.assertIn("compressed-tensors", METHODS)
        self.assertIn("awq", METHODS)

    def test_quant_methods_old_list_layout(self):
        src = 'from typing import List\nQUANTIZATION_METHODS: List[str] = ["aqlm", "awq"]\n'
        self.assertEqual(vllm.parse_quant_methods(src), {"aqlm", "awq"})

    def test_unknown_layout_is_an_error(self):
        with self.assertRaises(vllm.RuntimeSourceError):
            vllm.parse_registry("X = {}")


class CheckTest(unittest.TestCase):
    def test_ready_on_hopper(self):
        self.assertEqual(run()["verdict"], "READY")

    def test_unpinned_revision_warns(self):
        self.assertEqual(run(pinned=False)["verdict"], "READY WITH WARNINGS")

    def test_weight_only_fp8_on_ampere_warns(self):
        self.assertEqual(levels(run(gpu="a100-80"), "quantization"), ["warn"])

    def test_w8a8_fp8_on_ampere_fails(self):
        m = copy.deepcopy(MODEL)
        m["config"]["quantization_config"]["config_groups"]["group_0"]["input_activations"] = {"num_bits": 8}
        result = run(m, gpu="a100-80")
        self.assertEqual(levels(result, "quantization"), ["fail"])
        self.assertEqual(result["verdict"], "NOT READY")

    def test_awq_on_volta_family_table(self):
        m = copy.deepcopy(MODEL)
        m["config"]["quantization_config"] = {"quant_method": "awq", "bits": 4}
        self.assertEqual(levels(run(m, gpu="t4"), "quantization"), ["pass"])

    def test_blackwell_has_no_matrix_column(self):
        self.assertEqual(levels(run(gpu="b200"), "quantization"), ["warn"])

    def test_removed_architecture_fails(self):
        m = copy.deepcopy(MODEL)
        m["config"]["architectures"] = ["QWenLMHeadModel"]
        self.assertEqual(levels(run(m), "architecture"), ["fail"])

    def test_unknown_architecture_warns(self):
        m = copy.deepcopy(MODEL)
        m["config"]["architectures"] = ["NotARealModel"]
        self.assertEqual(levels(run(m), "architecture"), ["warn"])

    def test_architecture_newer_than_release_fails(self):
        old = dict(RT, version="0.8.0", registry=dict(REGISTRY, native=REGISTRY["native"] - {"Qwen3ForCausalLM"}))
        m = copy.deepcopy(MODEL)
        m["config"]["architectures"] = ["Qwen3ForCausalLM"]
        result = run(m, rt=old)
        arch = [f for f in result["findings"] if f["check"] == "architecture"][0]
        self.assertEqual(arch["level"], "fail")
        self.assertIn("starts in 0.8.4", arch["message"])

    def test_weights_larger_than_vram_fail(self):
        m = copy.deepcopy(MODEL)
        for s in m["siblings"]:
            if s["rfilename"].endswith(".safetensors"):
                s["size"] *= 3
        self.assertEqual(levels(run(m, gpu="t4"), "memory"), ["fail"])
        self.assertEqual(levels(run(m, gpu="t4", count=2), "memory"), ["pass"])

    def test_missing_tokenizer_warns(self):
        m = copy.deepcopy(MODEL)
        m["siblings"] = [s for s in m["siblings"] if "token" not in s["rfilename"]]
        self.assertIn("warn", levels(run(m), "artifact"))

    def test_sglang_release_index(self):
        rt = runtimes.load_sglang("0.5.23")
        self.assertEqual(levels(run(rt=rt), "architecture"), ["pass"])
        self.assertEqual(levels(run(rt=rt), "quantization"), ["info"])
        m = copy.deepcopy(MODEL)
        m["config"]["architectures"] = ["Qwen3ForCausalLM"]
        old = [f for f in run(m, rt=runtimes.load_sglang("0.4.5"))["findings"] if f["check"] == "architecture"][0]
        self.assertEqual(old["level"], "fail")
        self.assertIn("starts in 0.4.6", old["message"])

    def test_sglang_before_fallback_fails_unknown(self):
        m = copy.deepcopy(MODEL)
        m["config"]["architectures"] = ["NotARealModel"]
        self.assertEqual(levels(run(m, rt=runtimes.load_sglang("0.4.5")), "architecture"), ["fail"])
        self.assertEqual(levels(run(m, rt=runtimes.load_sglang("0.5.23")), "architecture"), ["warn"])

    def test_support_ranges(self):
        s = runtimes.support("QWenLMHeadModel")
        self.assertEqual(s["vLLM"]["seen"]["last"], "0.23.0")
        self.assertEqual(runtimes.support("Qwen3ForCausalLM")["SGLang"]["seen"]["first"], "0.4.6")

    def test_sglang_unknown_release(self):
        with self.assertRaises(runtimes.RuntimeSpecError):
            runtimes.load_sglang("9.9.9")

    def test_unknown_gpu(self):
        with self.assertRaises(KeyError):
            gpus.lookup("rtx-9999")


if __name__ == "__main__":
    unittest.main()
