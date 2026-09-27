"""Contract checks with a stub ranker; never load torch or execute model weights.

python test_server.py --model-dir /path/to/pinned/snapshot \
    --harness-dir /path/to/pinned/jevbench
Only clm_schema.py is read from the snapshot. Providing the pinned harness
also verifies the unmodified TypeSafeAdapter over a real loopback HTTP call.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

import server

SCHEMA = None
HARNESS_DIR = None


class PoisonMetadata(dict):
    """Fail the test if inference even reads answer/provenance metadata."""
    banned = {"expected", "id", "family", "provenance", "label", "answer_key"}

    def __getitem__(self, key):
        if key in self.banned:
            raise AssertionError(f"Inference read forbidden metadata: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.banned:
            raise AssertionError(f"Inference read forbidden metadata: {key}")
        return super().get(key, default)


class StubRanker:
    def __init__(self, scores=None, error=None):
        self.scores = scores or {}
        self.error = error
        self.calls = []

    def _tokens(self, text):
        return [0] + text.split()

    def rank(self, state, candidates, question=""):
        self.calls.append((state, candidates, question))
        if self.error:
            raise self.error
        rows = [{"candidate": key, "text": text, "score": self.scores.get(key, 0.0)}
                for key, text in candidates.items()]
        return sorted(rows, key=lambda row: -row["score"])


def engine_with(ranker):
    def build_joint_text(state, question, text):
        return SCHEMA.state_text(state, question) + "\n\nCandidate action:\n" + text
    return server.DecisionEngine(ranker, SCHEMA, build_joint_text, {"device": "stub"})


def request(qtype="choice", criteria=None):
    if criteria is None and qtype == "choice":
        criteria = {"b": "Beta action", "a": "Alpha action", "c": "Gamma action"}
    return {"model": server.MODEL_ID, "state": {"request": "Unmodified context"},
            "questions": {"decision": {"type": qtype, "instructions": "Choose the action.",
                                       "criteria": criteria}}}


class CoreContractTests(unittest.TestCase):
    def test_choice_order_exact_labels_and_softmax(self):
        ranker = StubRanker({"a": 2.0, "b": 0.0, "c": 1.0})
        result = engine_with(ranker).answer(request())
        answer = result["answers"]["decision"]
        self.assertEqual(answer["choice"], "a")
        self.assertEqual(list(answer["probabilities"]), ["b", "a", "c"])
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertAlmostEqual(answer["probabilities"]["a"] / answer["probabilities"]["b"],
                               math.exp(2.0 / server.TEMPERATURE))
        self.assertEqual(ranker.calls[0][1], {"b": "Beta action", "a": "Alpha action", "c": "Gamma action"})
        self.assertEqual(result["model"], server.MODEL_ID)
        self.assertEqual(result["probabilities_source"], server.PROBABILITIES_SOURCE)

    def test_ranking_ties_keep_request_order(self):
        result = engine_with(StubRanker()).answer(request())
        self.assertEqual(result["answers"]["decision"]["choice"], "b")

    def test_noul_aliases_and_default_wording(self):
        ranker = StubRanker({"no": -1.0, "yes": 1.0})
        result = engine_with(ranker).answer(request("noul"))
        self.assertEqual(ranker.calls[0][1], {
            "no": "false: No. This is false: Choose the action.",
            "yes": "true: Yes. This is true: Choose the action.",
        })
        self.assertGreater(result["answers"]["decision"]["noul"], 0.5)
        for criteria in ({"no": "Rejected", "yes": "Accepted"},
                         {"false": "Rejected", "true": "Accepted"}):
            engine_with(ranker).answer(request("noul", criteria))
            self.assertEqual(ranker.calls[-1][1], {"no": "false: Rejected", "yes": "true: Accepted"})

    def test_ambiguous_noul_rubric_refused(self):
        for criteria in ({"no": "N", "false": "F", "yes": "Y"}, {"maybe": "M"}, ["N", "Y"]):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                engine_with(StubRanker()).answer(request("noul", criteria))

    def test_score_levels_preserve_order_and_all_descriptions(self):
        ranker = StubRanker({"0": -1.0, "1": 2.0, "2": 0.0})
        result = engine_with(ranker).answer(request("score", ["low", {"level": "medium"}, "high"]))
        answer = result["answers"]["decision"]
        self.assertEqual(answer["type"], "score")
        self.assertEqual(list(answer["probabilities"]), ["0", "1", "2"])
        self.assertEqual(ranker.calls[0][1], {"0": "low", "1": "level: medium", "2": "high"})
        self.assertEqual(max(answer["probabilities"], key=answer["probabilities"].get), "1")

    def test_no_answer_or_task_metadata_is_read(self):
        body = request()
        body["expected"] = "a"
        body["id"] = "secret-id"
        body["family"] = "secret-family"
        body["questions"]["decision"] = PoisonMetadata({**body["questions"]["decision"],
                                                        "expected": "a", "provenance": {}})
        result = engine_with(StubRanker()).answer(PoisonMetadata(body))
        self.assertEqual(result["answers"]["decision"]["choice"], "b")

    def test_invalid_candidate_sets_refused_before_ranker(self):
        invalid = [{"one": "Only one"}, {"a": "same", "b": "same"}, {"a": "  ", "b": "yes"},
                   {str(i): str(i) for i in range(65)}]
        ranker = StubRanker()
        for criteria in invalid:
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                engine_with(ranker).answer(request("choice", criteria))
        self.assertEqual(ranker.calls, [])

    def test_bad_model_scores_not_repaired(self):
        for ranking in ([{"candidate": "a", "score": float("nan")}],
                        [{"candidate": "a", "score": 1.0}, {"candidate": "a", "score": 2.0}],
                        [{"candidate": "a", "score": 1.0}, {"candidate": "wrong", "score": 2.0}]):
            with self.subTest(ranking=ranking), self.assertRaises(RuntimeError):
                server.probabilities_from_ranking(["a", "b"], ranking)

    def test_usage_counts_repeated_joint_inputs_and_zero_generation(self):
        ranker = StubRanker()
        engine = engine_with(ranker)
        body = request()
        result = engine.answer(body)
        state, candidates, instructions = ranker.calls[0]
        count = sum(len(ranker._tokens(engine.build_joint_text(state, instructions, text)))
                    for text in candidates.values())
        self.assertEqual(result["usage"], {"input_tokens": count, "output_tokens": 0, "total_tokens": count})
        self.assertEqual(result["runtime"]["encoder_forward_passes"], 3)


class HTTPContractTests(unittest.TestCase):
    def setUp(self):
        self.ranker = StubRanker({"a": 2.0, "b": -1.0, "c": 0.0, "yes": 1.0, "no": -1.0})
        self.httpd = server.make_server(engine_with(self.ranker), port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def send(self, body):
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(self.endpoint + "/v1/systemone", data=encoded,
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError as response:
            return response.code, json.loads(response.read())
        with response:
            return response.status, json.loads(response.read())

    def test_health_and_loopback_only(self):
        with urllib.request.urlopen(self.endpoint + "/health", timeout=2) as response:
            health = json.loads(response.read())
        self.assertEqual(health["model"], server.MODEL_ID)
        self.assertFalse(health["limits"]["truncation"])
        with self.assertRaises(ValueError):
            server.make_server(engine_with(self.ranker), host="0.0.0.0", port=0)

    def test_original_limit_refusals_are_visible_as_422(self):
        for message in ("State has 2049 tokens; the limit is 2048",
                        "A candidate has 769 tokens; the limit is 768"):
            self.ranker.error = ValueError(message)
            status, result = self.send(request())
            self.assertEqual(status, 422)
            self.assertEqual(result["error"], message)

    def test_runtime_failure_500_is_distinct_from_refusal(self):
        self.ranker.error = RuntimeError("Secret request text must not be returned")
        status, result = self.send(request())
        self.assertEqual(status, 500)
        self.assertNotIn("Secret", result["error"])

    def test_duplicate_json_keys_and_nonfinite_numbers_refused(self):
        for raw in (b'{"state":1,"state":2}', b'{"state":NaN}', b'{}'):
            with self.subTest(raw=raw):
                self.assertEqual(self.send(raw)[0], 422)

    def test_unchanged_upstream_typesafe_adapter_all_question_types(self):
        if not HARNESS_DIR:
            self.skipTest("Pass --harness-dir for upstream integration check")
        from jevbench.adapters.typesafe import TypeSafeAdapter
        adapter = TypeSafeAdapter(endpoint=self.endpoint, model=server.MODEL_ID, key_env="")
        for qtype, criteria, labels in (
            ("choice", {"b": "Beta action", "a": "Alpha action", "c": "Gamma action"}, ["b", "a", "c"]),
            ("noul", {"no": "Rejected", "yes": "Accepted"}, ["no", "yes"]),
            ("score", ["low", "medium", "high"], ["0", "1", "2"]),
        ):
            body = request(qtype, criteria)
            task = SimpleNamespace(state=body["state"], question=body["questions"]["decision"], labels=labels)
            result = adapter.run(task)
            self.assertTrue(result.ok, result.error)
            self.assertEqual(set(result.probs), set(labels))
            self.assertAlmostEqual(sum(result.probs.values()), 1.0)
            self.assertEqual(result.model, server.MODEL_ID)
            self.assertEqual(result.usage["output_tokens"], 0)
            self.assertEqual(result.raw["probabilities_source"], server.PROBABILITIES_SOURCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--harness-dir", type=Path)
    args, remaining = parser.parse_known_args()
    schema_path = args.model_dir / "clm_schema.py"
    if hashlib.sha256(schema_path.read_bytes()).hexdigest() != server.PINNED_FILES["clm_schema.py"]:
        raise SystemExit("Tests require the pinned v0.4.0 schema")
    spec = importlib.util.spec_from_file_location("frozen_schema_contract_test", schema_path)
    SCHEMA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SCHEMA)
    HARNESS_DIR = args.harness_dir
    if HARNESS_DIR:
        sys.path.insert(0, str(HARNESS_DIR))
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
