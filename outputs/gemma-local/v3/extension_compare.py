"""CPU-only comparison of existing full-development logits and extended heads.

No model construction, inference, tokenizer, calibration file or final file is
used. Historical logits retain their original producer provenance; every system's
metrics are recomputed with the same current score_predictions function.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

BASELINE_NAMES = {
    "gemma_likelihood": "gemma_likelihood",
    "v2_unchanged": "v2_unchanged",
    "v2_hard_initialization": "v2_hard_initialization",
    "frozen": "frozen_pilot",
    "lora": "lora_pilot",
}


def _read(path):
    with Path(path).open() as handle: return json.load(handle)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _assert(condition, message):
    if not condition: raise RuntimeError(message)


def _validate_logits(name, rows, values):
    _assert(isinstance(values, list) and len(values) == len(rows), f"{name}: incomplete or mismatched development prediction count")
    for row, logits in zip(rows, values):
        eligible = row.get("within_pilot_limits")
        _assert(isinstance(eligible, bool), "Development row lacks verified length eligibility")
        if logits is None:
            _assert(not eligible, f"{name}: unexplained null prediction for an eligible development row")
        else:
            _assert(eligible, f"{name}: an ineligible row has an unaccounted prediction")
            _assert(isinstance(logits, list) and len(logits) == len(row["candidates"]), f"{name}: candidate-logit shape mismatch")
            _assert(all(isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in logits), f"{name}: non-finite or non-numeric development logits")


def _difference(current, baseline):
    changes = {}
    for family, value in current["by_family"].items():
        other = baseline["by_family"].get(family)
        if other and value["accuracy"] is not None and other["accuracy"] is not None:
            changes[family] = 100 * (value["accuracy"] - other["accuracy"])
    ordinal = {family: error - baseline["ordinal_mae_by_family"][family]
               for family, error in current.get("ordinal_mae_by_family", {}).items()
               if family in baseline.get("ordinal_mae_by_family", {})}
    return {
        "fresh_family_macro_percentage_points": 100 * (current["fresh_family_macro"] - baseline["fresh_family_macro"]),
        "family_accuracy_percentage_points": changes,
        "coverage_percentage_points": 100 * (current["coverage"] - baseline["coverage"]),
        "ordinal_mae_delta_lower_is_better": ordinal,
        "improved_families": [family for family, change in changes.items() if change > 1e-9],
        "regressed_families": [family for family, change in changes.items() if change < -1e-9],
    }


def run():
    """Validate immutable provenance, recompute metrics, and save one report."""
    from common import save, MODEL_REVISION
    from train import score_predictions, target_distribution

    started = time.monotonic()
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    data_root = Path(os.environ.get("V3_DATA_ROOT", str(root / "data")))
    v2 = Path(os.environ.get("V2_ROOT", "/experiment/work/decision-v2"))
    directory = root / "full_development"
    extension = root / "frozen_extension"
    dataset = data_root / "development.jsonl"
    with dataset.open() as handle: rows = [json.loads(line) for line in handle if line.strip()]
    _assert(len(rows) == 1000, "Comparison requires all 1,000 prepared development rows")
    ids = [row["id"] for row in rows]
    _assert(len(ids) == len(set(ids)), "Development IDs are not unique")
    _assert(all(row.get("split") == "development" for row in rows), "Unexpected partition in development file")
    for row in rows: target_distribution(row)
    data_hash = _sha(dataset); ids_hash = _digest(ids)
    protocol = _read(directory / "protocol.json")
    _assert(protocol["development_data_sha256"] == data_hash, "Full-development data hash changed")
    _assert(protocol["ids"] == ids and protocol["ids_sha256"] == ids_hash, "Full-development ID order changed")
    _assert(protocol["population"] == len(rows), "Full-development population changed")
    _assert(protocol["base_revision"] == MODEL_REVISION, "Base-model revision differs from the full-development protocol")
    _assert(protocol["state_limit"] == 2048 and protocol["candidate_limit"] == 768 and protocol["truncation"] is False, "Full-development input contract differs")
    _assert(protocol.get("no_calibration_or_final_access") is True, "Full-development data-access provenance is missing")
    checkpoint_files = {
        "v2_unchanged": v2 / "train/agent.safetensors",
        "v2_hard_initialization": v2 / "train/hard.safetensors",
        "frozen": root / "frozen/best/heads.safetensors",
        "lora": root / "lora/best/heads.safetensors",
    }
    current_heads = {name:_sha(path) for name,path in checkpoint_files.items()}
    current_adapter = _sha(root / "lora/best/adapter.safetensors")
    _assert(protocol["head_sha256"] == current_heads, "A baseline or pilot selected head changed after full-development inference")
    _assert(protocol["adapter_sha256"] == current_adapter, "The pilot LoRA adapter changed after full-development inference")
    code_root = Path(__file__).parent
    # Rendering/encoder implementation must match the recorded inference. A whole
    # train.py hash may legitimately change for extension orchestration; its old
    # hash remains verified against historical per-system provenance, not rewritten.
    _assert(protocol["model_code_sha256"] == _sha(code_root / "model.py"), "Encoder implementation changed since baseline inference")
    _assert(protocol["schema_code_sha256"] == _sha(code_root / "clm_schema.py"), "Input schema changed since baseline inference")
    shared_keys = ("development_data_sha256", "ids_sha256", "base_revision", "state_limit", "candidate_limit", "scoring_code_sha256", "model_code_sha256", "schema_code_sha256")
    predictions = {}; provenance_report = {}
    for stored_name, output_name in BASELINE_NAMES.items():
        path = directory / stored_name
        provenance = _read(path / "provenance.json")
        _assert(provenance.get("system") == stored_name, "Baseline provenance system name mismatch")
        _assert(all(provenance.get(key) == protocol[key] for key in shared_keys), f"{stored_name}: per-system provenance does not match frozen full-development protocol")
        _assert(provenance.get("head_sha256") == current_heads.get(stored_name), f"{stored_name}: head identity mismatch")
        _assert(provenance.get("adapter_sha256") == (current_adapter if stored_name == "lora" else None), f"{stored_name}: adapter identity mismatch")
        expected_likelihood = "Mean candidate token log-likelihood; state plus newline-newline Candidate response colon newline" if stored_name == "gemma_likelihood" else None
        _assert(provenance.get("likelihood_format") == expected_likelihood, f"{stored_name}: likelihood scoring contract differs")
        values = _read(path / "logits.json")
        _validate_logits(stored_name, rows, values)
        predictions[output_name] = values
        provenance_report[output_name] = {
            "producer_provenance": provenance,
            "producer_provenance_sha256": _sha(path / "provenance.json"),
            "logits_sha256": _sha(path / "logits.json"),
        }
    extension_report = _read(extension / "metrics.json")
    manifest = extension_report["manifest"]
    _assert(extension_report.get("status") == "complete" and extension_report.get("arm") == "frozen", "Frozen extension has not completed successfully")
    _assert(extension_report["steps"] == extension_report["requested_steps"] == manifest["requested_steps"], "Frozen extension update count is incomplete")
    _assert(manifest.get("development_population") == "full", "Extension selected against a subset rather than full development")
    _assert(manifest["development_data_sha256"] == data_hash, "Extension development data hash differs")
    _assert(manifest["development_ids"] == ids and manifest["development_ids_sha256"] == ids_hash, "Extension predictions do not match all 1,000 development IDs in order")
    _assert(manifest["initial_head_sha256"] == current_heads["v2_hard_initialization"], "Extension initialization head differs")
    _assert(manifest["training_data_sha256"] == _sha(data_root / "train.jsonl"), "Extension training data changed after its recorded run")
    checks = extension_report["checks"]
    _assert(checks.get("base_parameter_sample_unchanged") is True, "Extension backbone immutability check failed")
    _assert(checks.get("heads_changed_this_invocation") is True, "Extension reports no completed head update")
    _assert(checks.get("trainable_encoder_parameters") == 0, "Frozen extension unexpectedly trained encoder parameters")
    _assert(checks.get("reload_verified") is True, "Extension selected-checkpoint reload has not been verified")
    for name in ("selected_reload_max_logit_error", "candidate_permutation_max_logit_error"):
        value = checks.get(name)
        _assert(isinstance(value,(float,int)) and math.isfinite(value) and 0 <= value <= 1e-4, f"Extension {name} check failed")
    gradient = checks.get("head_gradient_norm")
    _assert(isinstance(gradient,(float,int)) and math.isfinite(gradient) and gradient > 0, "Extension head-gradient check failed")
    best = extension_report["best"]
    _assert(isinstance(best.get("step"),int) and 0 <= best["step"] <= extension_report["steps"], "Extension selected step is invalid")
    selected_file = extension / "best/heads.safetensors"
    selected_hash = _sha(selected_file)
    selected_source = checkpoint_files["v2_hard_initialization"] if best["step"] == 0 else extension / f"step-{best['step']:05d}/heads.safetensors"
    _assert(selected_hash == _sha(selected_source), "Extension selected heads differ from the recorded selected-step checkpoint")
    extension_predictions = _read(extension / "best-predictions.json")
    _validate_logits("frozen_extension", rows, extension_predictions)
    predictions["frozen_extension"] = extension_predictions
    provenance_report["frozen_extension"] = {
        "metrics_sha256": _sha(extension / "metrics.json"), "manifest_sha256": _digest(manifest),
        "logits_sha256": _sha(extension / "best-predictions.json"),
        "head_sha256": selected_hash, "adapter_sha256": None,
        "best_step": best["step"], "completed_steps": extension_report["steps"], "checks": checks,
        "development_data_sha256": data_hash, "ids_sha256": ids_hash,
    }
    systems = {}
    for name, values in predictions.items(): systems[name], _ = score_predictions(rows, values)
    # This checks that the newly recomputed metric is the same metric used to
    # select the extension checkpoint; no old per-system summary is reused.
    extension_score = systems["frozen_extension"]
    _assert(abs(extension_score["fresh_family_macro"] - best["score"]) <= 1e-10, "Recomputed extension score disagrees with its selection record")
    _assert(abs(extension_score["fresh_family_nll"] - best["nll"]) <= 1e-10, "Recomputed extension NLL disagrees with its selection record")
    comparisons = {name:_difference(extension_score, scores) for name,scores in systems.items() if name != "frozen_extension"}
    rankers = [name for name in systems if name != "gemma_likelihood"]
    fresh_counts = _population_counts(rows)
    practical = {
        "ranker_order_by_fresh_development_accuracy": sorted(rankers,key=lambda name:(-systems[name]["fresh_family_macro"],systems[name]["fresh_family_nll"])),
        "extension_tradeoffs": comparisons,
        "selection_policy": "Use fresh categorical families plus task-specific retention and coverage to choose a practical 270M ranker; ordinal MAE and preference ties remain separate.",
        "scope": "Choosing among these checkpoints is development selection, not evidence of JEV parity or universal task expertise.",
        "retention_note": "Factual/tool/recorded-action retrieval cases were exposed during v2; their metrics describe regression behavior rather than independent generalization.",
        "latency_note": "No timing is inferred from stored logits. Frozen heads reuse the original backbone and need no LoRA adapter; existing package/context timing reports remain separate.",
    }
    report = {
        "status": "complete", "development_only": True, "population": len(rows), **fresh_counts,
        "development_data_sha256": data_hash, "ids_sha256": ids_hash,
        "systems": systems, "practical_tradeoffs": practical, "provenance": provenance_report,
        "historical_full_development_protocol_sha256": _sha(directory / "protocol.json"),
        "historical_scoring_code_sha256": protocol["scoring_code_sha256"],
        "current_train_code_sha256": _sha(code_root / "train.py"),
        "current_metric_function_sha256": hashlib.sha256(inspect.getsource(score_predictions).encode()).hexdigest(),
        "comparison_code_sha256": _sha(__file__),
        "all_metrics_recomputed_from_logits": True, "historical_provenance_rewritten": False,
        "models_loaded": False, "calibration_opened": False, "final_opened": False,
        "publication_on_hold": True, "seconds": time.monotonic() - started,
    }
    output = root / "extension_compare"; output.mkdir(parents=True,exist_ok=True)
    save(output / "metrics.json", report)
    return report


def _population_counts(rows):
    """Counts only; labels and predictions are absent from this summary."""
    return {"fresh_cases":sum(row.get("evaluation_scope") != "retention_exposed" for row in rows),
            "retention_cases":sum(row.get("evaluation_scope") == "retention_exposed" for row in rows)}
