import unittest
from pathlib import Path

from readyml import trace

FIX = Path(__file__).parent / "fixtures"


def log(*lines):
    return [f"(X pid=1) INFO 11-14 {t} [f.py:1] {msg}" for t, msg in lines]


class RealLogTest(unittest.TestCase):
    def setUp(self):
        self.r = trace.parse((FIX / "vllm-startup-cpu-67598.log").read_text().splitlines())

    def test_total_and_ready(self):
        self.assertTrue(self.r["ready"])
        self.assertEqual(self.r["total_seconds"], 3044)

    def test_engine_init_matches_vllm_own_figure(self):
        # vLLM logged "init engine ... took 3004.62 s" in this run.
        init = [p for p in self.r["phases"] if p["phase"] == "engine init"][0]
        self.assertAlmostEqual(init["seconds"], 3004.62, delta=2)

    def test_reported_and_hint(self):
        self.assertEqual(self.r["reported"]["weight load"], 0.26)
        self.assertEqual(self.r["model_memory_gib"], 1.55)
        self.assertTrue(any("only accounts for" in h for h in self.r["hints"]))


class SyntheticTest(unittest.TestCase):
    def test_not_ready_names_last_phase(self):
        r = trace.parse(log(("10:00:00", "boot"), ("10:00:05", "Initializing a V1 LLM engine (v0.34.0) with config"), ("10:00:09", "Loading model from scratch...")))
        self.assertFalse(r["ready"])
        self.assertEqual(r["phases"][-1], {"phase": "model load", "seconds": None, "note": trace.NOTES["model load"]})
        self.assertIn("model load", r["hints"][0])

    def test_compile_without_cache_hint_and_sum(self):
        r = trace.parse(log(
            ("10:00:00", "boot"),
            ("10:00:01", "Initializing a V1 LLM engine (v0.34.0) with config"),
            ("10:00:02", "Loading model from scratch..."),
            ("10:00:12", "Model loading took 15.2 GiB memory and 10.0 seconds"),
            ("10:00:40", "Compiling a graph for compile range (1, 8192) takes 20.00 s"),
            ("10:01:00", "Compiling a graph for compile range (1, 8192) takes 15.50 s"),
            ("10:01:10", "init engine (profile, create kv cache, warmup model) took 58.00 s (compilation: 35.50 s)"),
            ("10:01:12", "Starting vLLM server on http://0.0.0.0:8000"),
        ))
        self.assertEqual(r["reported"]["compile"], 35.5)
        self.assertEqual(r["total_seconds"], 72)
        self.assertTrue(any("VLLM_CACHE_ROOT" in h for h in r["hints"]))

    def test_engine_only_log_is_engine_ready(self):
        r = trace.parse(log(
            ("10:00:00", "boot"),
            ("10:00:01", "Initializing a V1 LLM engine (v0.10.1.1) with config"),
            ("10:00:02", "Loading model from scratch..."),
            ("10:00:12", "Model loading took 24.2841 GiB and 10.0 seconds"),
            ("10:00:40", "init engine (profile, create kv cache, warmup model) took 28.00 seconds"),
        ))
        self.assertFalse(r["ready"])
        self.assertTrue(r["engine_ready"])
        self.assertEqual(r["total_seconds"], 40)
        self.assertEqual([p["phase"] for p in r["phases"]], ["process start", "engine start", "model load", "engine init"])
        self.assertFalse(any("ends before" in h for h in r["hints"]))
        self.assertIn("ENGINE READY", trace.render(r))

    def test_no_timestamps(self):
        self.assertIsNone(trace.parse(["hello", "world"]))
        self.assertIn("no timestamped", trace.render(None))


if __name__ == "__main__":
    unittest.main()
