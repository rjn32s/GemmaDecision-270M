"""Offline, single-request-at-a-time TypeSafe server for frozen GemmaDecision v4.

The stock JevBench TypeSafe adapter calls POST /v1/systemone. No JevBench
package, dataset, gold label, calibration fitter, credential or network client
is imported here. Download the pinned model separately before starting this
server. CUDA uses the release's BF16 encoder; CPU uses its FP32 encoder.

The native model returns uncalibrated scalar ranks. This server adds softmax
at the fixed temperature selected on the project's earlier, separate 300-item
calibration set. These probabilities are not generated confidence statements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_REPO = "rajan2k/GemmaDecision-270M"
MODEL_REVISION = "785d530221c990671f29976902540101bb9c7647"
MODEL_ID = MODEL_REPO + "@" + MODEL_REVISION
TEMPERATURE = 4.136820402388508
PROBABILITIES_SOURCE = "softmax_ranking_scores_external_temperature"
MAX_REQUEST_BYTES = 2 * 1024 * 1024

# Bind executable inference code, tokenizer, config, head, weights and the
# source calibration record to the published immutable v0.4.0 snapshot.
PINNED_FILES = {
    "model.safetensors": "d7a3e291bfdfa7cd85b33a8a99ef81a4a7d3192e46c77253f3daf14dfd7d6b95",
    "joint_head.safetensors": "72ec4e7d1f0908ad684eeae250f0f70128aa4342b46174bf92a3c9851c5dd965",
    "joint_config.json": "d48f2b5ad5fbea6a3d6723017a6f6a114260a331f0b917b8008de14501091533",
    "config.json": "c1c64396b2939c76f0fa091aa07815b6e6f5ac1a60bfe3878993f0f575dd3ff3",
    "joint_deployment.py": "2e5212fe7bf6bd40fc9414c8aa480184996eb2416fafc9e2aee57b183c851c64",
    "common.py": "7a41cecc445af13a1b88902920ebdc1cc0e854a4379c243302c4237c29dab307",
    "clm_schema.py": "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335",
    "clm_heads.py": "3f3b880e940a47b45879614b140fd873f7de9b13ccb8254b07989af7ea92e093",
    "tokenizer.json": "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
    "tokenizer_config.json": "94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf",
    "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "evidence/calibration.json": "3988b0b74e33ba4ccde8b2761ce6c4c9b3fd5fedf78cf2c52666565a40180c67",
}


def verify_model_files(model_dir):
    model_dir = Path(model_dir).resolve()
    for name, expected in PINNED_FILES.items():
        path = model_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Pinned v0.4.0 file hash mismatch: {name}")
    calibration = json.loads((model_dir / "evidence/calibration.json").read_text())
    if calibration["systems"]["selected"]["temperature"] != TEMPERATURE:
        raise ValueError("Frozen external calibration temperature mismatch")
    return model_dir


def question_candidates(question, schema):
    """Preserve the shipped schema's wording; translate binary label aliases.

    Choice insertion order and score level order remain unchanged. Binary
    candidates use canonical no/yes order, represented by the shipped schema's
    false/true prefixes. Both aliases supplied for one value are ambiguous and
    refused, rather than silently selecting a rubric.
    """
    if not isinstance(question, dict):
        raise ValueError("questions.decision must be an object")
    qtype = question.get("type")
    if qtype not in {"noul", "choice", "score"}:
        raise ValueError("Question type must be noul, choice or score")
    instructions = question.get("instructions")
    if not isinstance(instructions, str):
        raise ValueError("Question instructions must be a string")
    criteria = question.get("criteria")
    if qtype == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object or null")
        criteria = criteria or {}
        if set(criteria) - {"no", "yes", "false", "true"}:
            raise ValueError("noul criteria keys must be no/yes or false/true")
        normalized = {}
        for native, canonical in (("false", "no"), ("true", "yes")):
            if native in criteria and canonical in criteria:
                raise ValueError("noul criteria contain ambiguous label aliases")
            if native in criteria:
                normalized[native] = criteria[native]
            elif canonical in criteria:
                normalized[native] = criteria[canonical]
        criteria = normalized
    elif qtype == "choice" and isinstance(criteria, dict):
        if not all(isinstance(key, str) for key in criteria):
            raise ValueError("Choice labels must be strings")
    # Select only inference fields. In particular, do not forward arbitrary
    # caller metadata into the frozen schema or model.
    native_question = {"type": qtype, "instructions": instructions, "criteria": criteria}
    labels, texts = schema.candidates(native_question)
    if qtype == "noul":
        labels = ["no", "yes"]
    if not 2 <= len(labels) <= 64:
        raise ValueError("Provide 2–64 candidates")
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Candidate descriptions must be nonempty strings")
    if len(set(texts)) != len(texts):
        raise ValueError("Candidate descriptions must be distinct")
    return qtype, instructions, dict(zip(labels, texts))


def probabilities_from_ranking(labels, ranking):
    """Recover original option order before applying the frozen full softmax."""
    if not isinstance(ranking, list) or len(ranking) != len(labels):
        raise RuntimeError("Model returned an incomplete ranking")
    scores = {}
    for row in ranking:
        label, score = row["candidate"], row["score"]
        if label in scores or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError("Model returned invalid ranking scores")
        if not math.isfinite(score):
            raise RuntimeError("Model returned non-finite ranking scores")
        scores[label] = float(score)
    if set(scores) != set(labels):
        raise RuntimeError("Model returned unexpected ranking labels")
    logits = [scores[label] / TEMPERATURE for label in labels]
    highest = max(logits)
    weights = [math.exp(logit - highest) for logit in logits]
    total = math.fsum(weights)
    return dict(zip(labels, (weight / total for weight in weights)))


class DecisionEngine:
    """Injectable pure request boundary; production injects the frozen ranker."""

    def __init__(self, ranker, schema, build_joint_text, runtime=None):
        self.ranker = ranker
        self.schema = schema
        self.build_joint_text = build_joint_text
        self.runtime = dict(runtime or {})

    def health(self):
        return {
            "status": "ready", "model": MODEL_ID,
            "temperature": TEMPERATURE, "probabilities_source": PROBABILITIES_SOURCE,
            "limits": {"state_question_tokens": 2048, "candidate_tokens": 768,
                       "candidates_min": 2, "candidates_max": 64,
                       "truncation": False, "max_request_bytes": MAX_REQUEST_BYTES},
            "runtime": self.runtime,
        }

    def answer(self, body):
        if not isinstance(body, dict):
            raise ValueError("Request must be a JSON object")
        requested_model = body.get("model")
        if requested_model not in (None, MODEL_REPO, MODEL_ID, "GemmaDecision-270M"):
            raise ValueError("Requested model does not match this frozen server")
        if "state" not in body:
            raise ValueError("Request needs state")
        questions = body.get("questions")
        if not isinstance(questions, dict) or set(questions) != {"decision"}:
            raise ValueError("Provide exactly questions.decision")
        qtype, instructions, candidates = question_candidates(questions["decision"], self.schema)
        started = time.perf_counter()
        # The shipped ranker validates every limit before any encoder work,
        # uses separate full joint inputs, and never truncates.
        ranking = self.ranker.rank(body["state"], candidates, question=instructions)
        probs = probabilities_from_ranking(list(candidates), ranking)
        if qtype == "noul":
            answer = {"type": qtype, "noul": probs["yes"]}
        elif qtype == "choice":
            answer = {"type": qtype, "choice": max(probs, key=probs.get),
                      "probabilities": probs}
        else:
            answer = {"type": qtype, "score": math.fsum(int(k) * p for k, p in probs.items()),
                      "probabilities": probs}
        # Actual encoder input token count, including special tokens and the
        # full repeated request text for each individually scored candidate.
        input_tokens = sum(len(self.ranker._tokens(self.build_joint_text(
            body["state"], instructions, text))) for text in candidates.values())
        return {
            "model": MODEL_ID, "answers": {"decision": answer},
            "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                      "total_tokens": input_tokens},
            "probabilities_source": PROBABILITIES_SOURCE, "temperature": TEMPERATURE,
            "runtime": {**self.runtime, "candidate_count": len(candidates),
                        "encoder_forward_passes": len(candidates),
                        "server_processing_s": time.perf_counter() - started},
        }


def load_engine(model_dir, device="cpu", torch_threads=2):
    started = time.perf_counter()
    if device not in {"cpu", "cuda"}:
        raise ValueError("Benchmark devices are cpu or cuda")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    model_dir = verify_model_files(model_dir)
    # No token is required and all HF model loads below are local_files_only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(model_dir))
    import torch
    torch.set_num_threads(torch_threads)
    modules = {}
    for name in ("clm_schema", "common", "joint_deployment"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (model_dir / (name + ".py")).resolve():
            raise RuntimeError(f"Unexpected already-loaded inference module: {name}")
        modules[name] = module
    ranker = modules["joint_deployment"].GemmaJointRanker(model_dir, device=device)
    if ranker.max_state_tokens != 2048 or ranker.max_action_tokens != 768:
        raise RuntimeError("Unexpected model token limits")
    runtime = {
        "device": str(ranker.device), "encoder_dtype": str(next(ranker.encoder.parameters()).dtype),
        "head_dtype": str(next(ranker.head.parameters()).dtype),
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "load_and_verify_s": time.perf_counter() - started,
    }
    if device == "cuda":
        runtime["gpu_name"] = torch.cuda.get_device_name()
    return DecisionEngine(ranker, modules["clm_schema"], modules["joint_deployment"].build_joint_text, runtime)


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise ValueError("Non-finite JSON numeric constant")


def make_server(engine, host="127.0.0.1", port=8000):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The evaluator server binds only to loopback")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Avoid writing request content or request paths into sealed logs.
            pass

        def send_json(self, status, data):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "Unknown route"})
            self.send_json(200, engine.health())

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.send_json(404, {"error": "Unknown route"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Request body size is missing or exceeds the configured limit")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys,
                                  parse_constant=_reject_nonfinite_constant)
                response = engine.answer(body)
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
                return self.send_json(422, {"error": str(error), "model": MODEL_ID})
            except Exception as error:
                # Do not expose request content, tracebacks, or sealed task text.
                return self.send_json(500, {"error": f"Inference failed: {type(error).__name__}",
                                            "model": MODEL_ID})
            self.send_json(200, response)

    # Deliberately single-threaded: each request and its candidate forward
    # passes complete before the next request begins.
    return HTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    engine = load_engine(args.model_dir, args.device, args.torch_threads)
    server = make_server(engine, args.host, args.port)
    print(json.dumps({"listening": f"http://{args.host}:{server.server_port}", **engine.health()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
