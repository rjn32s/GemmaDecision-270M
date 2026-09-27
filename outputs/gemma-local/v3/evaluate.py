"""Frozen, resumable private final evaluation; no training or publication.

Nothing is read or loaded at import. run() requires frozen-recipe.json before it
reads calibration or final labels. Calibration is completed and saved before the
final file is decoded. This is our private held-out evaluation, not sealed
maintainer JevBench data. All model execution requires cloud CUDA.

Required frozen-recipe.json schema (paths below are relative to V3_ROOT):
{
  "schema_version": 1,
  "selected": {"arm": "frozen" or "lora" or "joint", "checkpoint_dir": "frozen/best",
               "heads_sha256": "...", "adapter_sha256": "... (LoRA only)"},
  "data_sha256": {"calibration.jsonl": "...", "final.jsonl": "..."},
  "base_revision": "the pinned common.MODEL_REVISION",
  "base_files": {"config.json": "...", "model.safetensors": "...",
                 "tokenizer.json": "...", "tokenizer_config.json": "..."},
  "v2_agent_heads_sha256": "...",
  "limits": {"state": 2048, "candidate": 768},
  "development_repeatability": {
      "passed": false, "seeds": [2704203],
      "evidence_path": "repeatability/metrics.json", "evidence_sha256": "..."},
  "selection_rationale": "Chosen from development results before final access."
}

For limits above 2048 state / 768 candidate, additionally require:
  "context_validation": {"passed": true, "state": 4096, "candidate": 768,
                         "evidence_path": "...", "evidence_sha256": "..."}
The caller must freeze this manifest from already reviewed development evidence.
The evaluator does not infer successful replication from different random seeds.
A failed repeatability report is allowed for an honest final diagnostic, but the
substantial-improvement gate cannot pass without successful two-seed evidence.
That earlier broad gate is historical and superseded by the user's practical
local-ranker scope; its checks remain as descriptive evidence, not a release
requirement. A single seed must be recorded honestly as passed=false.

For the joint arm, use checkpoint_dir="joint/best", heads_sha256 for
head.safetensors (singular filename), and selected.head_config={"hidden": 640,
"width": 512}. It runs the shared joint_deployment scorer: normalized last-token
vectors from the BF16 frozen encoder, an FP32 scalar head, and the exact text
schema_state(state, question) + "\\n\\nCandidate action:\\n" + candidate.
"""
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time


SYSTEMS = ("selected", "v2_unchanged", "gemma_likelihood")
BOOTSTRAP_REPLICATES = 1500
BOOTSTRAP_SEED = 2704903
CALIBRATION_GRID_POINTS = 121


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): value.update(block)
    return value.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inside(root, relative):
    path = Path(relative)
    if path.is_absolute(): raise ValueError("Frozen artifact paths must be relative")
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Frozen artifact path escapes the experiment directory")
    return result


def _check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(x not in "0123456789abcdef" for x in expected):
        raise ValueError("Frozen SHA256 must be a lowercase 64-character hexadecimal string")
    if _file_sha(path) != expected: raise RuntimeError(f"Frozen artifact has changed: {Path(path).name}")


def _evidence(root, item, label, require_passed=True):
    if type(item.get("passed")) is not bool: raise ValueError(f"{label} must record an explicit pass/fail outcome")
    if require_passed and item["passed"] is not True: raise RuntimeError(f"{label} has not passed before final evaluation")
    path = _inside(root, item["evidence_path"])
    if path.suffix != ".json" or path.name in {"frozen-recipe.json", "final.jsonl", "calibration.jsonl"}:
        raise ValueError("Development evidence must be an independently saved JSON report")
    _check_hash(path, item["evidence_sha256"])
    # The evidence's interpretation is frozen by the caller; hash and retain it.
    # Never reinterpret final performance as proof of successful development.
    return {"path": item["evidence_path"], "sha256": item["evidence_sha256"]}


def validate_frozen_recipe(root, base, v2_root):
    """Validate artifacts and bind an immutable evaluation attempt before labels."""
    from common import MODEL_REVISION
    root = Path(root); base = Path(base); v2_root = Path(v2_root)
    path = root / "frozen-recipe.json"
    if not path.is_file(): raise RuntimeError("Create and review frozen-recipe.json before final evaluation")
    recipe = json.loads(path.read_text())
    if recipe.get("schema_version") != 1: raise ValueError("Unsupported frozen recipe schema")
    if recipe.get("base_revision") != MODEL_REVISION: raise ValueError("This evaluation requires the pinned Gemma-270M backbone")
    if not str(recipe.get("selection_rationale", "")).strip(): raise ValueError("Record the development-only selection rationale")
    selected = recipe["selected"]
    if selected["arm"] not in {"frozen", "lora", "joint"}: raise ValueError("Unknown selected arm")
    checkpoint = _inside(root, selected["checkpoint_dir"])
    head_file = "head.safetensors" if selected["arm"] == "joint" else "heads.safetensors"
    _check_hash(checkpoint / head_file, selected["heads_sha256"])
    if selected["arm"] == "joint" and selected.get("head_config") != {"hidden": 640, "width": 512}:
        raise ValueError("Freeze the trained joint scalar-head configuration (hidden=640, width=512)")
    if selected["arm"] == "lora": _check_hash(checkpoint / "adapter.safetensors", selected["adapter_sha256"])
    _check_hash(v2_root / "train/agent.safetensors", recipe["v2_agent_heads_sha256"])
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(recipe["base_files"]): raise ValueError("Freeze all required backbone and tokenizer files")
    for filename, value in recipe["base_files"].items():
        if Path(filename).name != filename: raise ValueError("Base artifact paths must be flat filenames")
        _check_hash(base / filename, value)
    repetition = recipe["development_repeatability"]
    if not isinstance(repetition.get("seeds"), list) or not repetition["seeds"]:
        raise ValueError("Record the actual development seeds, including unsuccessful attempts")
    if repetition["passed"] is True and len(set(repetition["seeds"])) < 2:
        raise ValueError("Successful repeatability requires at least two recorded seeds")
    _evidence(root, repetition, "Development repeatability", require_passed=False)
    limits = recipe["limits"]
    for key, maximum in (("state", 8192), ("candidate", 2048)):
        if type(limits[key]) is not int or not 1 <= limits[key] <= maximum:
            raise ValueError(f"Unsupported {key} token limit")
    if limits["state"] > 2048 or limits["candidate"] > 768:
        evidence = recipe["context_validation"]
        _evidence(root, evidence, "Long-context development validation")
        if evidence["state"] < limits["state"] or evidence["candidate"] < limits["candidate"]:
            raise ValueError("Declared limits exceed the validated development limits")
    if set(recipe["data_sha256"]) != {"calibration.jsonl", "final.jsonl"}:
        raise ValueError("Freeze exactly the calibration and final data files")
    # Binary hashing does not inspect or print labels. All freezing checks above
    # precede opening either label-bearing data file as JSON.
    for filename, value in recipe["data_sha256"].items(): _check_hash(root / "data" / filename, value)
    code = Path(__file__).resolve().parent
    source_names = ["evaluate.py", "train.py", "model.py", "common.py", "clm_heads.py", "clm_schema.py"]
    if selected["arm"] == "joint": source_names += ["joint.py", "joint_deployment.py"]
    code_hashes = {name: _file_sha(code / name) for name in source_names}
    identity = {"recipe_sha256": _file_sha(path), "source_sha256": code_hashes,
                "data_sha256": recipe["data_sha256"], "protocol": "v3-frozen-evaluation-1"}
    output = root / "final-evaluation"; output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "attempt-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("This final evaluation attempt is immutable; recipe, data, or evaluation code changed")
    else:
        _save(identity_path, identity)
        _save(output / "frozen-recipe.json", recipe)
    return recipe, identity


def fit_temperature(rows, logits):
    """Equal-family soft-label calibration; only caller-supplied calibration rows."""
    import numpy as np
    from train import target_distribution
    if len(rows) != len(logits): raise ValueError("Calibration prediction count mismatch")
    groups = defaultdict(list)
    all_families = {row["family"] for row in rows}
    for row, values in zip(rows, logits):
        if values is None: continue
        z = np.asarray(values, dtype=np.float64)
        target = target_distribution(row)
        if z.shape != target.shape or not np.isfinite(z).all(): raise ValueError("Invalid calibration logits")
        groups[row["family"]].append((z, target))
    if set(groups) != all_families or not groups:
        raise RuntimeError("Every declared calibration family needs covered examples")
    temperatures = np.logspace(-1, 1, CALIBRATION_GRID_POINTS)
    objectives = []
    for temperature in temperatures:
        family_losses = []
        for examples in groups.values():
            losses = []
            for z, target in examples:
                scaled = z / temperature; scaled -= scaled.max()
                logp = scaled - np.log(np.exp(scaled).sum())
                losses.append(-float((target * logp).sum()))
            family_losses.append(float(np.mean(losses)))
        objectives.append(float(np.mean(family_losses)))
    chosen = int(np.argmin(objectives))
    return {"temperature": float(temperatures[chosen]), "equal_family_nll": objectives[chosen],
            "covered": sum(map(len, groups.values())), "total": len(rows),
            "covered_by_family": {name: len(examples) for name, examples in groups.items()},
            "grid": {"minimum": .1, "maximum": 10., "points": CALIBRATION_GRID_POINTS},
            "fit_partition": "calibration", "uses_soft_targets": True,
            "objective": "Mean covered soft-label cross entropy within family, then equal family mean"}


def _categorical(row):
    return row.get("evaluation_scope", "fresh_prompt") != "retention_exposed" and row.get("metric", "categorical") not in {"ordinal", "preference_tie"}


def paired_cluster_bootstrap(rows, outcomes_by_system, replicates=BOOTSTRAP_REPLICATES):
    """Paired, family-stratified cluster bootstrap; failures retain zero credit."""
    import numpy as np
    if set(outcomes_by_system) != set(SYSTEMS): raise ValueError("All three paired systems are required")
    for system, outcomes in outcomes_by_system.items():
        if [x["id"] for x in outcomes] != [x["id"] for x in rows]: raise ValueError(f"Population mismatch: {system}")
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if not _categorical(row): continue
        cluster = row["workflow_group"] if row["source"] == "synthetic_rules" else row["group"]
        grouped[row["family"]][cluster].append(index)
    if not grouped: raise ValueError("No categorical families for paired uncertainty")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {name: [] for name in SYSTEMS}; per_family = {}; cluster_counts = {}
    point = {name: [] for name in SYSTEMS}
    for family, clusters in sorted(grouped.items()):
        indices = list(clusters.values()); counts = np.asarray(list(map(len, indices)), dtype=np.float64)
        draws = rng.integers(0, len(indices), size=(replicates, len(indices)))
        family_samples = {}
        for name in SYSTEMS:
            credits = np.asarray([item["credit"] for item in outcomes_by_system[name]])
            sums = np.asarray([credits[index].sum() for index in indices])
            scores = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            family_samples[name] = scores; samples[name].append(scores)
            point[name].append(float(sums.sum() / counts.sum()))
        cluster_counts[family] = len(indices)
        per_family[family] = {}
        for baseline in SYSTEMS[1:]:
            difference = family_samples["selected"] - family_samples[baseline]
            per_family[family][baseline] = {
                "delta": point["selected"][-1] - point[baseline][-1],
                "ci95": [float(x) for x in np.quantile(difference, [.025, .975])],
                "clusters": len(indices),
                "uncertainty_between_workflows_estimable": len(indices) > 1,
            }
    means = {name: np.mean(values, axis=0) for name, values in samples.items()}
    point_means = {name: float(np.mean(values)) for name, values in point.items()}
    stronger = max(SYSTEMS[1:], key=lambda name: point_means[name])
    differences = {}
    for baseline in SYSTEMS[1:]:
        d = means["selected"] - means[baseline]
        differences[baseline] = {"delta": point_means["selected"] - point_means[baseline],
                                 "ci95": [float(x) for x in np.quantile(d, [.025, .975])]}
    adaptive_difference = means["selected"] - np.maximum(means["v2_unchanged"], means["gemma_likelihood"])
    return {"replicates": replicates, "seed": BOOTSTRAP_SEED,
            "method": "Within-family paired cluster resampling; whole prompt clusters, synthetic whole workflow clusters; equal-family accuracy",
            "cluster_counts": cluster_counts, "point_macro": point_means,
            "stronger_baseline_at_point_estimate": stronger,
            "versus_each_baseline": differences, "by_family": per_family,
            "versus_stronger_baseline": {
                "delta": point_means["selected"] - point_means[stronger],
                "ci95": [float(x) for x in np.quantile(adaptive_difference, [.025, .975])],
                "baseline_selection": "Maximum of the two baseline macros recomputed inside every replicate"},
            "single_cluster_families": [name for name, count in cluster_counts.items() if count == 1],
            "interpretation": "Conditional on the declared source/workflow mixture; one-cluster families cannot estimate uncertainty across unseen workflows."}


def finalize_results(rows, logits_by_system, temperatures, recipe, metadata=None):
    """CPU-only final summaries; caller must supply the complete frozen population."""
    import numpy as np
    from train import score_predictions
    def score(partition, values):
        metrics, outcomes = score_predictions(partition, values)
        # Empty covered populations have no NLL. Do not serialize a NaN if a
        # shared scorer version returns it for an entirely rejected length bin.
        if metrics.get("fresh_family_nll") is not None and not math.isfinite(metrics["fresh_family_nll"]):
            metrics["fresh_family_nll"] = None
        return metrics, outcomes
    if set(logits_by_system) != set(SYSTEMS): raise ValueError("Missing a required final system")
    if len({row["id"] for row in rows}) != len(rows): raise ValueError("Duplicate final row IDs")
    real_groups = {row["group"] for row in rows if row["source"] != "synthetic_rules"}
    if len(real_groups) < 1000: raise RuntimeError("Final evaluation requires at least 1,000 real prompt groups")
    categorical_families = {row["family"] for row in rows if _categorical(row)}
    if len(categorical_families) < 5: raise RuntimeError("Final evaluation requires at least five categorical decision families")
    raw, calibrated, outcomes = {}, {}, {}
    length_results = {}
    for name in SYSTEMS:
        logits = logits_by_system[name]
        if len(logits) != len(rows): raise ValueError(f"Missing final cases for {name}")
        temperature = float(temperatures[name]["temperature"])
        if not math.isfinite(temperature) or temperature <= 0: raise ValueError("Invalid frozen temperature")
        raw[name], outcomes[name] = score(rows, logits)
        scaled = [None if x is None else (np.asarray(x) / temperature).tolist() for x in logits]
        calibrated[name], _ = score(rows, scaled)
        buckets = defaultdict(list)
        for index, row in enumerate(rows):
            state_length, *action_lengths = row["lengths"]
            label = "short_state_and_candidates" if state_length <= 2048 and max(action_lengths) <= 768 else "long_state_or_candidate"
            buckets[label].append(index)
            if state_length > 2048: buckets["state_over_2048"].append(index)
            if max(action_lengths) > 768: buckets["candidate_over_768"].append(index)
        length_results[name] = {}
        for bucket, indices in buckets.items():
            length_results[name][bucket], _ = score([rows[i] for i in indices], [logits[i] for i in indices])
    uncertainty = paired_cluster_bootstrap(rows, outcomes)
    stronger = uncertainty["stronger_baseline_at_point_estimate"]
    family_deltas = {}
    for family in sorted(categorical_families):
        selected = raw["selected"]["by_family"][family]["accuracy"]
        family_deltas[family] = {name: selected - raw[name]["by_family"][family]["accuracy"] for name in SYSTEMS[1:]}
    improving = [family for family, values in family_deltas.items() if values[stronger] > 0]
    regressions = {family: values["v2_unchanged"] for family, values in family_deltas.items() if values["v2_unchanged"] < -.03 - 1e-12}
    paired = uncertainty["versus_stronger_baseline"]
    checks = {
        "at_least_10_percentage_point_gain": paired["delta"] >= .10 - 1e-12,
        "paired_cluster_ci_lower_above_zero": paired["ci95"][0] > 0,
        "improvement_in_at_least_three_families": len(improving) >= 3,
        "no_family_regression_over_three_points_vs_v2": not regressions,
        "at_least_95_percent_request_coverage": raw["selected"]["coverage"] >= .95,
        "development_gain_reproduced": recipe["development_repeatability"].get("passed") is True and len(set(recipe["development_repeatability"].get("seeds", []))) >= 2,
        "at_least_1000_real_prompt_groups": len(real_groups) >= 1000,
        "at_least_five_categorical_families": len(categorical_families) >= 5,
    }
    return {"status": "complete", "evaluation": "Private frozen final evaluation; not an official JevBench result",
            "cases": len(rows), "real_prompt_groups": len(real_groups),
            "raw_metrics": raw, "calibrated_metrics": calibrated, "length_results": length_results,
            "calibration": temperatures, "paired_uncertainty": uncertainty,
            "family_deltas": family_deltas,
            "substantial_improvement_gate": {"passed": all(checks.values()), "checks": checks,
                "historical_superseded_by_practical_scope": True,
                "scope_note": "Earlier broad improvement gate retained for transparency. The current task is a usable local CLM-style ranker; passing this historical gate is not required.",
                "families_improving_vs_stronger_baseline": improving, "regressions_vs_v2": regressions,
                "policy": "Five primary categorical families; ordinal error and preference ties are reported separately, not converted into accuracy."},
            "metadata": metadata or {}, "clm_final_reference_run": False, "jev_parity_measured": False,
            "publication_on_hold": True,
            "next_action": "Review evidence with the user. Preserve a failed final result; do not tune against this final set."}


class _Deadline(Exception):
    pass


class _JointEvaluationModel:
    """Use the actual portable joint scorer without copying the base weights.

    The temporary local layout contains links only to already hash-verified
    backbone/head files. The scorer's initialization and ranking implementation
    are unchanged, and the layout disappears after this inference phase.
    """
    arm = "joint"

    def __init__(self, base, checkpoint, selected, base_files, limits):
        import tempfile
        from joint_deployment import GemmaJointRanker
        self._directory = tempfile.TemporaryDirectory(prefix="frozen-joint-ranker-")
        directory = Path(self._directory.name)
        try:
            for name in base_files:
                (directory / name).symlink_to((Path(base) / name).resolve())
            (directory / "joint_head.safetensors").symlink_to((Path(checkpoint) / "head.safetensors").resolve())
            _save(directory / "joint_config.json", {
                "head_config": selected["head_config"], "head_file": "joint_head.safetensors",
                "max_state_tokens": limits["state"], "max_action_tokens": limits["candidate"],
            })
            self.ranker = GemmaJointRanker(directory, device="cuda")
        except BaseException:
            self._directory.cleanup()
            raise

    def ids(self, text):
        return self.ranker._tokens(text)

    def logits(self, row):
        # Stable IDs recover the original option ordering from the public rank
        # API, whose return list is intentionally sorted by score.
        options = {str(index): text for index, text in enumerate(row["candidates"])}
        ranking = self.ranker.rank(row["state"], options, row.get("question", ""))
        scores = {item["candidate"]: item["score"] for item in ranking}
        if len(ranking) != len(options) or set(scores) != set(options):
            raise RuntimeError("Joint deployment returned an incomplete candidate population")
        return [scores[str(index)] for index in range(len(options))]

    def close(self):
        self._directory.cleanup()


def _prepare_rows(rows, model, limits):
    from common import schema_state
    from train import target_distribution
    if len({row["id"] for row in rows}) != len(rows): raise ValueError("Duplicate evaluation IDs")
    for row in rows:
        target_distribution(row)
        candidates = row["candidates"]
        if not 2 <= len(candidates) <= 64 or any(not isinstance(x, str) or not x.strip() for x in candidates) or len(set(candidates)) != len(candidates):
            raise ValueError("Invalid evaluation candidates")
        row["rendered_state"] = schema_state(row["state"], row.get("question", ""))
        row["lengths"] = [len(model.ids(text)) for text in [row["rendered_state"], *candidates]]
        row["eligible"] = row["lengths"][0] <= limits["state"] and max(row["lengths"][1:]) <= limits["candidate"]


def _progress(output, phase, system, rows, identity):
    path = output / f"{phase}-{system}-predictions.json"
    header = {"phase": phase, "system": system, "attempt_identity_sha256": _json_sha(identity),
              "row_ids_sha256": _json_sha([row["id"] for row in rows])}
    if path.exists():
        result = json.loads(path.read_text())
        if result["header"] != header or [x["id"] for x in result["rows"]] != [x["id"] for x in rows]:
            raise RuntimeError("Prediction resume population or frozen recipe changed")
    else:
        result = {"header": header, "rows": [{"id": row["id"], "status": "pending", "logits": None} for row in rows]}
        _save(path, result)
    return path, result


def _likelihood_row(model, row, deadline):
    import torch
    from torch.nn import functional as F
    # Complete state/question prefix and complete candidates. No truncation.
    prefix = model.tokenizer(row["rendered_state"] + "\n\nCandidate response:\n", add_special_tokens=True)["input_ids"]
    values = []
    for candidate in row["candidates"]:
        if time.monotonic() >= deadline - 35: raise _Deadline()
        action = model.tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if not action: raise ValueError("Empty candidate after likelihood tokenization")
        ids = torch.tensor([prefix + action], device="cuda")
        hidden = model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
        hidden = hidden[len(prefix) - 1:len(prefix) + len(action) - 1]
        total = 0.
        for offset in range(0, len(action), 16):
            z = F.linear(hidden[offset:offset + 16], model.encoder.get_input_embeddings().weight)
            cap = getattr(model.encoder.config, "final_logit_softcapping", None)
            if cap is not None: z = (z / cap).tanh() * cap
            z = z.float(); labels = torch.tensor(action[offset:offset + 16], device="cuda")
            if not torch.isfinite(z).all(): raise RuntimeError("Non-finite original Gemma token logits")
            total += float((z[torch.arange(len(labels), device="cuda"), labels] - torch.logsumexp(z, dim=-1)).sum())
        values.append(total / len(action))
    return values


def _predict(model, rows, system, phase, output, identity, deadline, cache):
    import torch
    path, progress = _progress(output, phase, system, rows, identity)
    last_save = time.monotonic(); completed_now = 0
    with torch.inference_mode():
        for row, record in zip(rows, progress["rows"]):
            if record["status"] != "pending": continue
            if time.monotonic() >= deadline - 40: break
            if not row["eligible"]:
                record.update(status="length_rejection", failure_reason="declared_token_limits", logits=None)
            else:
                try:
                    if system == "gemma_likelihood":
                        values = _likelihood_row(model, row, deadline)
                    elif getattr(model, "arm", None) == "joint":
                        values = model.logits(row)
                    else:
                        missing = [text for text in [row["rendered_state"], *row["candidates"]] if text not in cache]
                        if missing:
                            cache.update(zip(missing, model.encode(missing)))
                        values = model.logits(row, vectors=cache).float().cpu().tolist()
                    if len(values) != len(row["candidates"]) or not all(math.isfinite(x) for x in values):
                        raise RuntimeError("Non-finite or incomplete decision logits")
                    record.update(status="ok", logits=values)
                except _Deadline:
                    break
                except torch.cuda.OutOfMemoryError:
                    # This is a failed request, not an omitted example. Do not
                    # retry with a shorter input or alter the frozen policy.
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                    record.update(status="runtime_failure", failure_reason="cuda_out_of_memory", logits=None)
            completed_now += 1
            if completed_now % 24 == 0 or time.monotonic() - last_save >= 20:
                _save(path, progress); last_save = time.monotonic()
                done = sum(r["status"] != "pending" for r in progress["rows"])
                print(f"Frozen {phase} {system}: {done}/{len(rows)} cases completed", flush=True)
    _save(path, progress)
    return progress


def _all_complete(progress):
    return all(record["status"] != "pending" for record in progress["rows"])


def _phase(rows, phase, recipe, identity, root, base, v2_root, output, deadline):
    import torch
    from model import DecisionModel
    result = {}
    for name in SYSTEMS:
        _, existing = _progress(output, phase, name, rows, identity)
        if _all_complete(existing):
            result[name] = existing
            continue
        if time.monotonic() >= deadline - 90: break
        if name == "selected":
            selected = recipe["selected"]; checkpoint = _inside(root, selected["checkpoint_dir"])
            if selected["arm"] == "joint":
                model = _JointEvaluationModel(base, checkpoint, selected, recipe["base_files"], recipe["limits"])
            else:
                model = DecisionModel(base, checkpoint / "heads.safetensors", selected["arm"],
                                      checkpoint / "adapter.safetensors" if selected["arm"] == "lora" else None).eval()
        else:
            model = DecisionModel(base, v2_root / "train/agent.safetensors", "frozen").eval()
        cache = {}
        try:
            _prepare_rows(rows, model, recipe["limits"])
            result[name] = _predict(model, rows, name, phase, output, identity, deadline, cache)
        finally:
            if getattr(model, "arm", None) == "joint": model.close()
        del model, cache
        gc.collect(); torch.cuda.empty_cache()
        if not _all_complete(result[name]): break
    # Always return every declared case for every system, including pending ones.
    for name in SYSTEMS:
        if name not in result: _, result[name] = _progress(output, phase, name, rows, identity)
    return result


def _partial(output, phase, progress, identity, started):
    report = {"status": "partial", "phase": phase, "attempt_identity": identity,
              "systems": {name: dict(Counter(row["status"] for row in state["rows"])) for name, state in progress.items()},
              "accuracy_reported": False, "reason": "Deadline reached with explicit pending cases; resume the same frozen attempt",
              "function_seconds": time.monotonic() - started, "publication_on_hold": True}
    _save(output / "metrics.json", report)
    return report


def run(seconds=1680):
    """Bounded cloud entry point; resumable without altering data or checkpoints."""
    import torch
    from common import read_jsonl
    from transformers import AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError("Final model evaluation runs only on the cloud CUDA GPU")
    if seconds < 180: raise ValueError("Final evaluation requires a bounded allocation of at least 180 seconds")
    started = time.monotonic(); deadline = started + seconds
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    base = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
    v2_root = Path(os.environ.get("V2_ROOT", "/experiment/work/decision-v2"))
    output = root / "final-evaluation"
    recipe, identity = validate_frozen_recipe(root, base, v2_root)
    calibration_path = output / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration["attempt_identity_sha256"] != _json_sha(identity): raise RuntimeError("Calibration belongs to another attempt")
    else:
        rows = list(read_jsonl(root / "data/calibration.jsonl"))
        if not rows: raise ValueError("Empty calibration partition")
        progress = _phase(rows, "calibration", recipe, identity, root, base, v2_root, output, deadline)
        if not all(_all_complete(value) for value in progress.values()):
            return _partial(output, "calibration", progress, identity, started)
        temperatures = {name: fit_temperature(rows, [x["logits"] for x in progress[name]["rows"]]) for name in SYSTEMS}
        calibration = {"attempt_identity_sha256": _json_sha(identity), "systems": temperatures,
                       "calibration_file_sha256": recipe["data_sha256"]["calibration.jsonl"],
                       "fitted_before_final_labels_opened": True}
        _save(calibration_path, calibration)
    if time.monotonic() >= deadline - 90:
        report = {"status": "partial", "phase": "calibration_complete", "final_labels_opened": (output / "final-labels-opened.json").exists(),
                  "reason": "Calibration is frozen; final phase needs another bounded invocation", "publication_on_hold": True}
        _save(output / "metrics.json", report); return report
    marker = output / "final-labels-opened.json"
    if not marker.exists():
        _save(marker, {"attempt_identity_sha256": _json_sha(identity), "calibration_sha256": _file_sha(calibration_path),
                       "policy": "Final labels may now be read only for this frozen evaluation. Do not resume model selection using this set."})
    else:
        previous = json.loads(marker.read_text())
        if previous["attempt_identity_sha256"] != _json_sha(identity) or previous["calibration_sha256"] != _file_sha(calibration_path):
            raise RuntimeError("Final labels were opened under a different frozen calibration or recipe")
    rows = list(read_jsonl(root / "data/final.jsonl"))
    if not rows: raise ValueError("Empty final partition")
    # Length metadata must be populated even when all prediction files resume as
    # complete and no model needs loading. This tokenizer does not perform inference.
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    class TokenLengths:
        @staticmethod
        def ids(text): return tokenizer(text, add_special_tokens=True)["input_ids"]
    _prepare_rows(rows, TokenLengths(), recipe["limits"])
    progress = _phase(rows, "final", recipe, identity, root, base, v2_root, output, deadline)
    if not all(_all_complete(value) for value in progress.values()):
        return _partial(output, "final", progress, identity, started)
    logits = {name: [record["logits"] for record in value["rows"]] for name, value in progress.items()}
    report = finalize_results(rows, logits, calibration["systems"], recipe, metadata={
        "attempt_identity": identity, "failures": {name: dict(Counter(record["status"] for record in value["rows"])) for name, value in progress.items()},
        "limits": recipe["limits"], "development_repeatability": recipe["development_repeatability"],
        "calibration_sha256": _file_sha(calibration_path), "final_labels_opened": True})
    report["function_seconds"] = time.monotonic() - started
    _save(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=1680)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.seconds), indent=2, allow_nan=False))
