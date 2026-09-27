"""Full shared-development comparison after the small screening pilot.

Uses all development rows, including explicitly labeled exposed retention rows.
The final and calibration files are never opened. These results can inform an
extension choice, so they are not an unbiased final-performance estimate.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

SYSTEMS = ("gemma_likelihood", "v2_unchanged", "v2_hard_initialization", "frozen", "lora")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _resume(output, system, provenance, population):
    from common import save
    directory = output / system; directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "provenance.json"
    predictions = directory / "logits.json"
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise RuntimeError(f"{system} development or checkpoint provenance changed; refusing stale scores")
    if predictions.exists() and not metadata.exists():
        raise RuntimeError(f"{system} has logits without provenance")
    save(metadata, provenance)
    values = json.loads(predictions.read_text()) if predictions.exists() else []
    if len(values) > population: raise RuntimeError("Stored predictions exceed development population")
    return directory, values


def _validate_predictions(rows, values):
    import math
    for row, logits in zip(rows, values):
        if logits is None:
            if row["eligible"]: raise RuntimeError("An eligible case has an unexplained null prediction")
        elif not row["eligible"] or len(logits) != len(row["candidates"]) or not all(math.isfinite(x) for x in logits):
            raise RuntimeError("Stored development predictions are invalid")


def _score_heads(model, rows, values, directory, deadline, vectors=None):
    import torch
    from common import save
    model.eval(); latest = time.monotonic()
    for offset in range(len(values), len(rows), 4):
        if time.monotonic() > deadline - 25: break
        block = rows[offset:offset + 4]
        eligible = [row for row in block if row["eligible"]]
        with torch.no_grad():
            scored = model.logits_batch(eligible, vectors=vectors) if eligible else []
        iterator = iter(scored)
        values.extend(next(iterator).float().cpu().tolist() if row["eligible"] else None for row in block)
        if time.monotonic() - latest >= 15:
            save(directory / "logits.json", values)
            print(f"Full development {directory.name}: {len(values)}/{len(rows)}", flush=True)
            latest = time.monotonic()
    save(directory / "logits.json", values)
    return values


def run(seconds=600, output_name='full_development'):
    """Bounded GPU entry point. A partial run cannot emit comparison scores."""
    if not 90 <= seconds <= 1200: raise ValueError("Full development allocation must be 90–1200 seconds")
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from common import read_jsonl, save, schema_state, MODEL_REVISION
    from model import DecisionModel, LORA_CONFIG
    from train import score_predictions, vector_cache, likelihood
    if not torch.cuda.is_available(): raise RuntimeError("Full-development inference runs on Modal GPU only")
    started = time.monotonic(); deadline = started + seconds
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    data_root = Path(os.environ.get("V3_DATA_ROOT", str(root / "data")))
    base = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
    v2 = Path(os.environ.get("V2_ROOT", "/experiment/work/decision-v2"))
    if output_name not in {'full_development','extended_development'}:
        raise ValueError('Unknown development report name')
    output = root / output_name; output.mkdir(parents=True, exist_ok=True)
    checkpoint_files = {"v2_unchanged": v2 / "train/agent.safetensors",
                        "v2_hard_initialization": v2 / "train/hard.safetensors",
                        "frozen": root / "frozen/best/heads.safetensors",
                        "lora": root / "lora/best/heads.safetensors"}
    adapter = root / "lora/best/adapter.safetensors"
    # Verify all selected artifacts exist before spending inference time.
    hashes = {name: _sha(path) for name, path in checkpoint_files.items()}
    adapter_hash = _sha(adapter)
    dataset = data_root / "development.jsonl"
    rows = list(read_jsonl(dataset))
    if not rows or len({row["id"] for row in rows}) != len(rows): raise RuntimeError("Empty or duplicate-ID development population")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    for row in rows:
        row["rendered_state"] = schema_state(row["state"], row.get("question", ""))
        lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in [row["rendered_state"], *row["candidates"]]]
        row["eligible"] = lengths[0] <= 2048 and max(lengths[1:]) <= 768
    protocol = {"development_data_sha256": _sha(dataset), "ids_sha256": _digest([row["id"] for row in rows]),
        "population": len(rows), "ids": [row["id"] for row in rows], "base_revision": MODEL_REVISION,
        "head_sha256": hashes, "adapter_sha256": adapter_hash,
        "state_limit": 2048, "candidate_limit": 768, "truncation": False,
        "fresh_cases": sum(row.get("evaluation_scope") != "retention_exposed" for row in rows),
        "retention_cases": sum(row.get("evaluation_scope") == "retention_exposed" for row in rows),
        "no_calibration_or_final_access": True,
        "purpose": "Full development selection check after the small pilot; not final performance evidence.",
        "lora_config": LORA_CONFIG,
        "scoring_code_sha256": _sha(Path(__file__).parent / "train.py"),
        "model_code_sha256": _sha(Path(__file__).parent / "model.py"),
        "schema_code_sha256": _sha(Path(__file__).parent / "clm_schema.py")}
    existing = output / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise RuntimeError("Full-development protocol or checkpoint changed; use a separate result directory")
    save(existing, protocol)
    predictions = {}; directories = {}
    shared = {key: protocol[key] for key in ["development_data_sha256", "ids_sha256", "base_revision", "state_limit", "candidate_limit", "scoring_code_sha256", "model_code_sha256", "schema_code_sha256"]}
    for system in SYSTEMS:
        provenance = {**shared, "system": system, "head_sha256": hashes.get(system),
                      "adapter_sha256": adapter_hash if system == "lora" else None,
                      "likelihood_format": "Mean candidate token log-likelihood; state plus newline-newline Candidate response colon newline" if system == "gemma_likelihood" else None}
        directories[system], predictions[system] = _resume(output, system, provenance, len(rows))
        _validate_predictions(rows, predictions[system])
    base_systems = [name for name in SYSTEMS if name != "lora" and len(predictions[name]) < len(rows)]
    if base_systems and time.monotonic() < deadline - 60:
        model = DecisionModel(base, checkpoint_files["v2_hard_initialization"]).eval()
        head_systems = [name for name in base_systems if name != "gemma_likelihood"]
        vectors = None
        if head_systems:
            vectors, partial = vector_cache(model, rows, deadline)
            if partial:
                # Persisted cache resumes on the next invocation; no model outcomes invented.
                base_systems = []
            else:
                for system in head_systems:
                    model.heads.load_state_dict(load_file(str(checkpoint_files[system]), device="cuda"))
                    predictions[system] = _score_heads(model, rows, predictions[system], directories[system], deadline, vectors)
        if "gemma_likelihood" in base_systems and time.monotonic() < deadline - 60:
            predictions["gemma_likelihood"] = likelihood(model, rows, deadline,
                                                         directories["gemma_likelihood"] / "logits.json")
        del model, vectors
        gc.collect(); torch.cuda.empty_cache()
    if len(predictions["lora"]) < len(rows) and time.monotonic() < deadline - 45:
        model = DecisionModel(base, checkpoint_files["lora"], "lora", adapter).eval()
        predictions["lora"] = _score_heads(model, rows, predictions["lora"], directories["lora"], deadline)
        del model
        gc.collect(); torch.cuda.empty_cache()
    pending = {name: {"completed": len(values), "population": len(rows)} for name, values in predictions.items() if len(values) != len(rows)}
    report = {"status": "pending" if pending else "complete", "pending_systems": pending,
              "development_only": True, "final_and_calibration_opened": False,
              "development_data_sha256": protocol["development_data_sha256"], "population": len(rows),
              "provenance": protocol, "seconds": time.monotonic() - started,
              "publication_on_hold": True}
    if not pending:
        systems = {}; outcomes = {}
        for name, values in predictions.items():
            _validate_predictions(rows, values)
            systems[name], outcomes[name] = score_predictions(rows, values)
            save(directories[name] / "metrics.json", systems[name])
        report["systems"] = systems
        report["development_deltas"] = {}
        for arm in ["frozen", "lora"]:
            comparisons = {}
            for baseline in ["gemma_likelihood", "v2_unchanged", "v2_hard_initialization"]:
                comparisons[baseline] = {"fresh_macro": systems[arm]["fresh_family_macro"] - systems[baseline]["fresh_family_macro"],
                    "by_family": {family: result["accuracy"] - systems[baseline]["by_family"][family]["accuracy"]
                                  for family, result in systems[arm]["by_family"].items()
                                  if result["accuracy"] is not None and systems[baseline]["by_family"][family]["accuracy"] is not None}}
            report["development_deltas"][arm] = comparisons
        report["provisional_arm_by_fresh_accuracy"] = max(["frozen", "lora"], key=lambda name: (
            systems[name]["fresh_family_macro"], -systems[name]["fresh_family_nll"]))
        report["selection_caution"] = "Full development shares cases with the screening subset and remains selection data. Inspect retention and per-family tradeoffs before allocating an extension; this does not pass the untouched-final gate."
    save(output / "metrics.json", report)
    return report
