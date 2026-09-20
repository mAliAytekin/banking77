import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from banking77_jev import load_records, write_json
from compare import deepseek_payload, estimate_cost, execute, parse_prediction, select_indices
from plot_comparison import create_report


class Dataset(list):
    features = {"label": SimpleNamespace(names=[f"intent_{i:02}" for i in range(77)])}
    _fingerprint = "synthetic-test-fixture"


def dataset_fixture():
    return Dataset({"text": f"Message {label}-{example}", "label": label}
                   for label in range(77) for example in range(4))


def report_fixture(folder):
    labels = Dataset.features["label"].names
    write_json(folder / "comparison.json", {
        "labels": labels, "indices": list(range(154)), "prices_usd_per_million": {}, "synthetic": True,
    })
    for provider in ("jev", "deepseek"):
        with (folder / f"{provider}.jsonl").open("w") as handle:
            for i in range(154):
                expected = labels[i // 2]
                predicted = expected if i % (5 if provider == "jev" else 7) else labels[(i // 2 + 1) % 77]
                handle.write(json.dumps({
                    "index": i, "text": f"SYNTHETIC FIXTURE {i}", "expected": expected,
                    "predicted": predicted, "valid": True, "usage": {"input_tokens": 100, "output_tokens": 5},
                    "estimated_cost_usd": 0.00004 if provider == "jev" else 0.00008,
                    "api_seconds": 0.15 if provider == "jev" else 0.6,
                    "retry_wait_seconds": 0, "pacing_wait_seconds": 0.3, "attempts": 1,
                    "model": "SYNTHETIC-NOT-A-BENCHMARK",
                }) + "\n")


class ComparisonTests(unittest.TestCase):
    def test_pilot_is_balanced_reproducible_and_full_default(self):
        dataset = dataset_fixture()
        indices = select_indices(dataset, True)
        self.assertEqual(len(indices), 154)
        self.assertEqual(len(set(indices)), 154)
        self.assertEqual(set(Counter(dataset[i]["label"] for i in indices).values()), {2})
        self.assertEqual(indices, select_indices(dataset, True))
        self.assertNotEqual(indices, select_indices(dataset, True, 7))
        self.assertEqual(select_indices(dataset), list(range(308)))

    def test_deepseek_prompt_and_invalid_answers(self):
        body = deepseek_payload("Where is my card?", ["card", "cash"], "deepseek-v4-pro")
        self.assertEqual(body["thinking"]["type"], "disabled")
        self.assertEqual(body["messages"][1]["content"], "Where is my card?")
        for content, finish, valid in [(' {"label":"card"}', "stop", True),
                                       ('{"label":"card"}', "length", False),
                                       ('{"label":"other"}', "stop", False),
                                       ("not json", "stop", False)]:
            data = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
            self.assertEqual(parse_prediction("deepseek", data, ["card", "cash"])["valid"], valid)

    def test_cost_includes_cached_and_output_tokens(self):
        prices = {"deepseek": {"input": 2, "cached": 0.2, "output": 4}, "jev": {"input": 0.427}}
        usage = {"input_tokens": 1000, "output_tokens": 100, "cached_tokens": 800}
        self.assertAlmostEqual(estimate_cost("deepseek", usage, prices), 0.00096)
        self.assertAlmostEqual(estimate_cost("jev", usage, prices), 0.000427)
        self.assertIsNone(estimate_cost("deepseek", {**usage, "cached_tokens": None}, prices))

    def test_two_providers_resume_and_configuration_guard(self):
        dataset = dataset_fixture()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(pilot=True, seed=42, split="test", revision="master",
                                   jev_model="jev-latest", deepseek_model="deepseek-v4-pro",
                                   output=Path(directory), dry_run=False, delay=0.3, batch_delay=2,
                                   batch_size=20, concurrency=1, timeout=10, retries=2, jev_input_price=None,
                                   deepseek_input_price=None, deepseek_cached_price=None, deepseek_output_price=None)
            calls = []

            def respond(client, body, retries, delay, endpoint):
                provider = "jev" if "state" in body else "deepseek"
                text = body.get("state") or body["messages"][1]["content"]
                calls.append((provider, text))
                answer = "intent_00"
                return {"answers": {"intent": {"choice": answer}},
                        "choices": [{"message": {"content": json.dumps({"label": answer})}, "finish_reason": "stop"}],
                        "_timing": {"api_seconds": 0.1, "retry_wait_seconds": 0, "attempts": 1}}

            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "fake", "DEEPSEEK_API_KEY": "fake"}), \
                 patch("compare.load_banking_data", return_value=dataset), \
                 patch("compare.request_json", side_effect=respond), \
                 patch("compare.time.sleep") as sleep, \
                 patch("plot_comparison.plot_report"), redirect_stdout(io.StringIO()):
                execute(args)
                self.assertEqual(len(calls), 308)
                self.assertEqual(sum(c.args[0] == 2.3 for c in sleep.call_args_list), 7)
                self.assertEqual({t for p, t in calls if p == "jev"}, {t for p, t in calls if p == "deepseek"})
                execute(args)
                self.assertEqual(len(calls), 308)
                args.seed = 99
                with self.assertRaisesRegex(RuntimeError, "settings differ"):
                    execute(args)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["paired_samples"], 154)

    def test_paired_report_and_actual_plot_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report_fixture(folder)
            report = create_report(folder)
            self.assertTrue(report["complete"])
            self.assertEqual(report["paired"]["jev"]["correct"], 123)
            self.assertEqual(report["paired"]["jev"]["evaluated"], 154)
            for name in ("comparison", "per_class_f1"):
                self.assertGreater((folder / f"{name}.png").stat().st_size, 10000)
                self.assertTrue((folder / f"{name}.svg").exists())
            overview_svg = (folder / "comparison.svg").read_text()
            class_svg = (folder / "per_class_f1.svg").read_text()
            self.assertIn("79.9%", overview_svg)
            self.assertIn("(123/154)", overview_svg)
            self.assertIn("Overall paired results", class_svg)
            self.assertIn("100%", class_svg)
            rows = load_records(folder / "deepseek.jsonl")
            with (folder / "deepseek.jsonl").open("w") as handle:
                handle.write(json.dumps(rows[0]) + "\n")
            with patch("plot_comparison.plot_report"):
                partial = create_report(folder)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["paired_samples"], 1)
            self.assertEqual(partial["paired"]["jev"]["completed"], 1)
            self.assertEqual(partial["all_completed"]["jev"]["completed"], 154)


if __name__ == "__main__":
    unittest.main()
