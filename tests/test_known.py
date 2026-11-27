import copy
import json
import unittest
from pathlib import Path

from readyml import check, gpus, known, runtimes, vllm

FIX = Path(__file__).parent / "fixtures"
MODEL = json.loads((FIX / "llama-3.1-8b-fp8.json").read_text())
RT = runtimes.load_vllm("0.34.0", fetch=lambda v, p: (FIX / f"vllm-0.34.0-{'registry' if p == vllm.REGISTRY else 'quantization'}.py").read_text())
RECORD = {
    "id": "vllm-1", "runtime": "vllm", "url": "https://example.test/1", "architectures": ["LlamaForCausalLM"], "models": [],
    "quantization": ["fp8"], "gpu_families": ["ampere"], "versions": {"affected": ">=0.30.0", "fixed_in": None},
    "flags": [], "symptom": "x", "cause": "FP8 kernels are missing on this GPU", "workaround": "use an Ada or Hopper GPU",
}


def hits(record, gpu="a100-80", rt=RT, model=MODEL):
    return known.match([record], model, rt, gpus.lookup(gpu))


class VersionTest(unittest.TestCase):
    def test_ranges(self):
        self.assertTrue(known.version_matches(">=0.25.0", "0.34.0"))
        self.assertFalse(known.version_matches(">=0.25.0", "0.24.1"))
        self.assertTrue(known.version_matches("0.33.0", "0.33.0"))
        self.assertFalse(known.version_matches("0.33.0", "0.34.0"))
        self.assertTrue(known.version_matches("0.25", "0.25.3"))
        self.assertTrue(known.version_matches("0.25.0 to 0.31.0", "0.28.0"))
        self.assertFalse(known.version_matches(">=0.25.0, <0.30.0", "0.31.0"))
        self.assertIsNone(known.version_matches("main", "0.34.0"))
        self.assertIsNone(known.version_matches(None, "0.34.0"))


class MatchTest(unittest.TestCase):
    def test_match_and_filters(self):
        self.assertEqual(len(hits(RECORD)), 1)
        self.assertEqual(hits(RECORD, gpu="h100"), [])
        self.assertEqual(hits(dict(RECORD, runtime="sglang")), [])
        self.assertEqual(hits(dict(RECORD, architectures=["Other"])), [])
        self.assertEqual(hits(dict(RECORD, quantization=["awq"])), [])
        self.assertEqual(hits(dict(RECORD, versions={"affected": "0.20.0", "fixed_in": None})), [])
        self.assertEqual(hits(dict(RECORD, versions={"affected": ">=0.30.0", "fixed_in": "0.33.1"})), [])

    def test_unreadable_version_still_matches_unconfirmed(self):
        h = hits(dict(RECORD, versions={"affected": "nightly", "fixed_in": None}))
        self.assertFalse(h[0]["version_confirmed"])

    def test_record_without_model_needs_quantization(self):
        loose = dict(RECORD, architectures=[], models=[])
        self.assertEqual(len(hits(loose)), 1)
        self.assertEqual(hits(dict(loose, quantization=[])), [])

    def test_model_quantization_tags(self):
        self.assertIn("fp8", known.model_quantization(MODEL))
        m = copy.deepcopy(MODEL)
        m["config"]["quantization_config"] = {"quant_method": "awq"}
        m["safetensors"] = {"parameters": {"I32": 1}}
        m["id"] = "x/y-AWQ"
        self.assertEqual(known.model_quantization(m), {"awq"})

    def test_finding_text(self):
        result = check.run(MODEL, RT, gpus.lookup("a100-80"), 1, True, records=[RECORD])
        f = [x for x in result["findings"] if x["check"] == "known issue"][0]
        self.assertEqual(f["level"], "warn")
        self.assertIn("Workaround: use an Ada or Hopper GPU.", f["message"])
        self.assertEqual(f["source"], "https://example.test/1")

    def test_bundled_data_is_well_formed(self):
        for r in known.load():
            self.assertIn(r["runtime"], ("vllm", "sglang", "kserve"))
            self.assertTrue(r["url"].startswith("https://github.com/"))
            self.assertTrue(r["cause"])


if __name__ == "__main__":
    unittest.main()
