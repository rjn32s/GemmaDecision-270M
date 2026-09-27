"""Protocol tests with mocked HTTP responses; no model inference or network."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_public


HARNESS = Path(os.environ.get("JEVBENCH_TEST_HARNESS", "work/jevbench-submission")).resolve()


class PublicRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules, _ = run_public._load_harness(HARNESS)
        cls.tasks, cls.tiers = run_public.load_plan(cls.modules, HARNESS)

    def fake_success(self, url, body, headers, timeout):
        self.assertTrue(url.endswith("/v1/systemone"))
        self.assertEqual(set(body), {"state", "model", "questions"})
        self.assertEqual(body["model"], run_public.MODEL_ID)
        self.assertNotIn("Authorization", headers)
        question = body["questions"]["decision"]
        kind = question["type"]
        answer = {"type": kind}
        if kind == "noul":
            answer["noul"] = 0.2
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice" else [str(index) for index in range(len(question["criteria"]))]
            )
            answer["probabilities"] = {label: 1.0 / len(labels) for label in labels}
            if kind == "choice":
                answer["choice"] = labels[0]
        return 200, {"model": run_public.MODEL_ID, "answers": {"decision": answer}}, 0.001

    def run_mocked(self, output, response):
        with patch.object(self.modules["adapters.typesafe"], "http_post_json", side_effect=response) as http:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_public.run(HARNESS, output)
        return summary, http.call_count

    def test_complete_plan_keeps_official_metrics_and_unknown_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            summary, calls = self.run_mocked(output, self.fake_success)
            self.assertEqual(calls, 231)
            self.assertEqual(summary["n_planned"], 231)
            self.assertEqual(summary["n_attempted"], 231)
            self.assertEqual(summary["n_valid"], 231)
            self.assertEqual(summary["unattempted_task_ids"], [])
            self.assertEqual(summary["brier_n"], summary["calibration_n"])
            self.assertEqual(summary["ece_n"], summary["calibration_n"])
            self.assertGreater(summary["ordinal_mae_n"], 0)
            self.assertEqual(summary["latency"]["n"], 231)
            self.assertIsNone(summary["price_per_1000_decisions_usd"])
            self.assertIsNone(summary["ledger_charged_usd"])
            self.assertAlmostEqual(summary["reservation_ledger"]["charged_reservations_usd"], 4.62)
            self.assertFalse(summary["reservation_ledger"]["is_measured_spend"])
            self.assertEqual(
                {tier: value["n_attempted"] for tier, value in summary["per_tier"].items()},
                {"easy": 48, "original": 72, "hard": 111},
            )
            records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
            self.assertEqual([record["task_id"] for record in records], [task.id for task in self.tasks])
            self.assertTrue(all(record["cost_usd"] is None for record in records))
            expected = self.modules["summarize"].summarize(self.tasks, records, ledger_charged=None)
            self.assertEqual(json.loads((output / "official_summary.json").read_text()), expected)
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 231)
            # A second invocation cannot silently rerun any task in this run.
            with self.assertRaises(FileExistsError):
                run_public.run(HARNESS, output)

    def test_three_infrastructure_errors_stop_without_retry(self):
        def unavailable(*args):
            return 500, {"error": "synthetic test outage"}, 0.001

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            summary, calls = self.run_mocked(output, unavailable)
            self.assertEqual(calls, 3)
            self.assertEqual(summary["n_attempted"], 3)
            self.assertEqual(summary["n_valid"], 0)
            self.assertEqual(summary["n_correct"], 0)
            self.assertEqual(summary["unattempted_n"], 228)
            self.assertEqual(summary["unattempted_task_ids"], [task.id for task in self.tasks[3:]])
            self.assertEqual(summary["coverage"], 3 / 231)
            self.assertEqual(summary["accuracy_all_planned_scorable"], 0.0)
            self.assertEqual(summary["brier_n"], 0)
            self.assertEqual(summary["ece_n"], 0)
            self.assertFalse(summary["complete"])

    def test_context_refusals_count_wrong_and_do_not_trigger_outage_stop(self):
        def refuse(*args):
            return 422, {"error": "synthetic context refusal"}, 0.001

        with tempfile.TemporaryDirectory() as temporary:
            summary, calls = self.run_mocked(Path(temporary) / "run", refuse)
            self.assertEqual(calls, 231)
            self.assertEqual(summary["n_valid"], 0)
            self.assertEqual(summary["n_correct"], 0)
            self.assertEqual(summary["accuracy"], 0.0)
            self.assertEqual(summary["invalid_attempts_n"], 231)
            self.assertTrue(summary["complete"])

    def test_accidental_tariff_and_remote_endpoint_fail_before_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch.dict(os.environ, {"TYPESAFE_PRICE_INPUT_PER_M": "0"}):
                with self.assertRaisesRegex(ValueError, "no per-token tariff"):
                    run_public.run(HARNESS, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "loopback"):
                run_public.run(HARNESS, output, "https://example.com")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
