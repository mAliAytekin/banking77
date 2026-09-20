import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from banking77_jev import evaluate, jev_credentials, load_records, metrics, payload, retry_delay, run


class RunnerTests(unittest.TestCase):
    def test_jev_credentials_select_matching_endpoint(self):
        with patch.dict("os.environ", {"JEV_AI_KEY": "sk_gateway"}, clear=True):
            self.assertEqual(jev_credentials(), ("sk_gateway", "https://jev-ai.pro/api/v1/systemone"))
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "ts_direct", "JEV_AI_KEY": "sk_gateway"}, clear=True):
            self.assertEqual(jev_credentials(), ("ts_direct", "https://api.typesafe.ai/v1/systemone"))
    def test_batch_pacing_and_resume(self):
        class Dataset(list):
            features = {"label": SimpleNamespace(names=["cash", "card"])}
            _fingerprint = "fixture"

        dataset = Dataset([{"text": "Cash please", "label": 0}] * 3)
        response = {"answers": {"intent": {"choice": "cash", "confidence": 1}},
                    "usage": {"input_tokens": 10}}
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(dry_run=False, split="test", revision="main", limit=0,
                                   model="jev-latest", output=Path(folder), timeout=10,
                                   batch_size=2, delay=0.3, batch_delay=2, retries=1)
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "fake-test-key"}), \
                 patch("banking77_jev.load_banking_data", return_value=dataset), \
                 patch("banking77_jev.evaluate", return_value=response) as evaluate_mock, \
                 patch("banking77_jev.time.sleep") as sleep:
                run(args)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.3, 0.3, 2])
                run(args)
                self.assertEqual(evaluate_mock.call_count, 3)
            self.assertEqual(json.loads((Path(folder) / "metrics.json").read_text())["accuracy"], 1)

    def test_request_has_no_ground_truth(self):
        body = payload("Where is my card?", ["card_arrival", "cash"], "jev-latest")
        self.assertEqual(body["state"], "Where is my card?")
        self.assertNotIn("expected", json.dumps(body))
        self.assertEqual(len(body["questions"]["intent"]["criteria"]), 2)

    def test_retry_after_then_success(self):
        responses = iter([
            httpx.Response(429, headers={"Retry-After": "4"}),
            httpx.Response(200, json={"answers": {"intent": {"choice": "cash", "confidence": 0.9}}}),
        ])
        with httpx.Client(transport=httpx.MockTransport(lambda request: next(responses))) as client:
            with patch("banking77_jev.time.sleep") as sleep:
                result = evaluate(client, payload("cash", ["cash"], "jev-latest"), 2, 0.3)
        self.assertEqual(result["answers"]["intent"]["choice"], "cash")
        sleep.assert_called_once_with(4.0)

    def test_auth_failure_is_not_retried(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as client:
            with patch("banking77_jev.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "Jev HTTP 401"):
                    evaluate(client, payload("x", ["cash"], "jev-latest"), 5, 0.3)
        sleep.assert_not_called()

    def test_gateway_credit_failure_is_clear(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(402))) as client:
            with self.assertRaisesRegex(RuntimeError, "no usable credits"):
                evaluate(client, payload("x", ["cash"], "jev-latest"), 0, 0,
                         "https://jev-ai.pro/api/v1/systemone")

    def test_retry_exhaustion(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            with patch("banking77_jev.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "retry limit"):
                    evaluate(client, payload("x", ["cash"], "jev-latest"), 2, 0.3)
        self.assertEqual(sleep.call_count, 2)

    def test_unknown_label_rejected(self):
        response = {"answers": {"intent": {"choice": "unknown", "confidence": 1}}}
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))) as client:
            with self.assertRaisesRegex(RuntimeError, "Invalid Jev"):
                evaluate(client, payload("x", ["cash"], "jev-latest"), 0, 0)

    def test_truncated_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_bytes(b'{"index": 0}\n{"ind')
            self.assertEqual(load_records(path), {0: {"index": 0}})
            self.assertEqual(path.read_bytes(), b'{"index": 0}\n')

    def test_metrics(self):
        rows = [{"expected": "a", "predicted": "a", "usage": {"input_tokens": 10}},
                {"expected": "b", "predicted": "a", "usage": {"input_tokens": 20}}]
        result = metrics(rows, ["a", "b"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1_all_classes"], 1 / 3)
        self.assertEqual(result["recorded_input_tokens"], 30)

    def test_http_date_retry_after(self):
        self.assertGreater(retry_delay("Wed, 01 Jan 2098 00:00:00 GMT", 0), 60)


if __name__ == "__main__":
    unittest.main()
