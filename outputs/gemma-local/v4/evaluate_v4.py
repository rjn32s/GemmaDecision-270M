"""Resumable v4 evaluation on fresh prompt groups; no training/publication.

run_development() scores the two fixed baselines on development only. freeze()
hashes a selected complete model and all split files without decoding final or
calibration labels. run() validates that frozen identity, calibrates on calibration
only, then evaluates every declared final case. Failed cases remain in denominators.
All model execution is cloud CUDA only; metric helpers are CPU-testable.
"""
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

PRIMARY = ("intent_routing", "evidence_relation")
SYSTEMS = ("selected", "gemma_likelihood", "v3_frozen_joint")
BASELINES = SYSTEMS[1:]
HEAD_CONFIG = {"hidden": 640, "width": 512}
DEFAULT_LIMITS = {"state": 2048, "candidate": 768}
REQUIRED_BASE = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute(): raise ValueError("Artifact paths must be relative")
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()): raise ValueError("Artifact path escapes experiment root")
    return result


def paths():
    return (Path(os.environ.get("V4_ROOT", "/experiment/work/decision-v4")),
            Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model")),
            Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3")))


def _check(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise RuntimeError(f"Frozen artifact changed: {path}")


def model_files(path):
    result = {p.name: sha(p) for p in Path(path).iterdir()
              if p.is_file() and p.suffix in {".json", ".safetensors", ".model"}
              and p.name not in {"metrics.json", "manifest.json", "optimizer.pt"}}
    if not REQUIRED_BASE.issubset(result): raise RuntimeError("Complete flat model/tokenizer directory required")
    return result


def source_files():
    directory = Path(__file__).resolve().parent
    required = ["evaluate_v4.py", "joint_deployment.py", "common.py", "clm_schema.py"]
    result = {name: sha(directory / name) for name in required}
    for name in ("train_v4.py", "train.py", "full_joint.py", "clm_heads.py"):
        if (directory / name).exists(): result[name] = sha(directory / name)
    return result


def freeze(checkpoint_dir, training_report="train/metrics.json", selection_rationale="",
           seeds=None, limits=None):
    """CPU-only explicit selection freeze; no calibration/final labels decoded."""
    from common import MODEL_REVISION
    root, base, v3 = paths(); output = root / "final-evaluation"
    if (output / "final-labels-opened.json").exists(): raise RuntimeError("Final already opened; no new selection")
    if not selection_rationale.strip(): raise ValueError("A development-only selection rationale is required")
    checkpoint = inside(root, checkpoint_dir); report_path = inside(root, training_report)
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete": raise RuntimeError("Selected training run must be complete")
    if report.get("selected_backbone_updated") is not True or report.get("best", {}).get("step", 0) <= 0:
        raise RuntimeError("The selected checkpoint is still the untuned step-zero initialization")
    audit_path = root / "data/audit.json"; audit = json.loads(audit_path.read_text())
    if not (audit.get("passed") is True and audit.get("passed_structural") is True
            and audit.get("all_prior_groups_excluded_from_new_holdouts") is True
            and audit.get("new_holdout_prior_exposure_conflicts") == 0
            and audit.get("prior_holdout_training_conflicts") == 0):
        raise RuntimeError("Fresh split audit has not passed")
    head_name = "head.safetensors" if (checkpoint / "head.safetensors").exists() else "joint_head.safetensors"
    checkpoint_files = model_files(checkpoint)
    if head_name not in checkpoint_files: raise RuntimeError("Selected scalar head is missing")
    if checkpoint_files["model.safetensors"] != report.get("selected_encoder_sha256") or checkpoint_files[head_name] != report.get("selected_head_sha256"):
        raise RuntimeError("Selected checkpoint differs from the development-selected training report")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if checkpoint_files[name] != sha(base / name): raise RuntimeError("Selected and baseline tokenizers differ")
    if checkpoint_files["model.safetensors"] == sha(base / "model.safetensors"):
        raise RuntimeError("The full-tuning checkpoint still contains unchanged official base weights")
    limits = dict(limits or DEFAULT_LIMITS)
    if limits != DEFAULT_LIMITS: raise ValueError("This serving/evaluation protocol is fixed to 2048/768 tokens")
    recipe = {"schema_version": 1, "protocol": "gemmadecision-v4-fresh-prompts-1",
        "selected": {"checkpoint_dir": str(checkpoint_dir), "files": checkpoint_files,
                     "head_file": head_name, "head_config": HEAD_CONFIG, "base_weights_modified": True},
        "base_model": "google/gemma-3-270m", "base_revision": MODEL_REVISION,
        "initial_base_files": model_files(base), "v3_head_sha256": sha(v3 / "joint/best/head.safetensors"),
        "data_sha256": {name: sha(root / "data" / name) for name in
                        ("train.jsonl", "development.jsonl", "calibration.jsonl", "final.jsonl")},
        "sources_sha256": source_files(), "training_report": str(training_report),
        "training_report_sha256": sha(report_path), "selection_rationale": selection_rationale,
        "seeds": list(seeds or [report["manifest"]["seed"]]), "repeatability_established": False,
        "limits": limits, "primary_families": list(PRIMARY),
        "release_targets": {"intent_routing": .85, "evidence_relation": .70,
                            "minimum_primary_coverage": .99, "require_positive_paired_ci": True},
        "scope": "Fresh known-domain prompts: four-way banking-intent selection and three-way NLI; no arbitrary routing, unseen-intent or JEV-parity claim",
        "publication_authorized": True, "benchmark_issue_submission_authorized": False}
    for name in ("manifest.json", "audit.json", "metrics.json"):
        path = root / "data" / name
        if path.exists(): recipe.setdefault("data_evidence_sha256", {})[name] = sha(path)
    for name, digest in recipe["data_sha256"].items():
        if audit.get("output_hashes", {}).get(name) != digest: raise RuntimeError("Audited split file changed")
    manifest = report["manifest"]
    if manifest.get("base_revision") != MODEL_REVISION:
        raise RuntimeError("Training initial model revision differs from the pinned model")
    if manifest.get("base_weights_sha256") != recipe["initial_base_files"]["model.safetensors"]:
        raise RuntimeError("Training initialized from another base checkpoint")
    if manifest.get("initial_head_sha256") != recipe["v3_head_sha256"]:
        raise RuntimeError("Training initialized from another v3 head")
    for split in ("train", "development"):
        if manifest.get("data_sha256", {}).get(split) != recipe["data_sha256"][split + ".jsonl"]:
            raise RuntimeError("Training report belongs to another data split")
    destination = root / "frozen-recipe.json"
    if destination.exists() and json.loads(destination.read_text()) != recipe:
        raise RuntimeError("A different recipe is already frozen; cannot overwrite it")
    save(destination, recipe)
    return {"status": "frozen", "recipe_sha256": sha(destination), "selected": recipe["selected"],
            "calibration_or_final_labels_opened": False}


def validate_recipe():
    from common import MODEL_REVISION
    root, base, v3 = paths(); path = root / "frozen-recipe.json"
    recipe = json.loads(path.read_text())
    if recipe.get("schema_version") != 1 or recipe.get("base_revision") != MODEL_REVISION:
        raise RuntimeError("Unexpected frozen recipe or initial model revision")
    if recipe.get("primary_families") != list(PRIMARY) or recipe.get("limits") != DEFAULT_LIMITS:
        raise RuntimeError("Frozen task/length policy changed")
    selected = recipe["selected"]; checkpoint = inside(root, selected["checkpoint_dir"])
    if selected.get("head_config") != HEAD_CONFIG or selected.get("base_weights_modified") is not True:
        raise RuntimeError("Full joint architecture/provenance mismatch")
    if not REQUIRED_BASE.issubset(selected["files"]): raise RuntimeError("Missing frozen checkpoint files")
    if selected["head_file"] not in selected["files"]: raise RuntimeError("Head is not bound to the recipe")
    for name, digest in selected["files"].items(): _check(inside(checkpoint, name), digest)
    for name, digest in recipe["initial_base_files"].items(): _check(inside(base, name), digest)
    _check(v3 / "joint/best/head.safetensors", recipe["v3_head_sha256"])
    _check(inside(root, recipe["training_report"]), recipe["training_report_sha256"])
    if set(recipe["data_sha256"]) != {"train.jsonl", "development.jsonl", "calibration.jsonl", "final.jsonl"}:
        raise RuntimeError("All four new split identities must be frozen")
    for name, digest in recipe["data_sha256"].items(): _check(root / "data" / name, digest)
    for name, digest in recipe.get("data_evidence_sha256", {}).items(): _check(inside(root / "data", name), digest)
    if source_files() != recipe["sources_sha256"]: raise RuntimeError("Frozen scoring/serving sources changed")
    identity = {"protocol": recipe["protocol"], "recipe_sha256": sha(path)}
    output = root / "final-evaluation"; output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "attempt-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("Cannot change a resumed final evaluation")
    save(identity_path, identity)
    return recipe, identity


def target_distribution(row):
    import numpy as np
    count = len(row["candidates"])
    if "target_probs" in row: result = np.asarray(row["target_probs"], dtype=np.float64)
    else:
        target = row["target"]
        if not isinstance(target, int) or not 0 <= target < count: raise ValueError("Invalid categorical target")
        result = np.zeros(count, dtype=np.float64); result[target] = 1.
    if result.shape != (count,) or not np.isfinite(result).all() or (result < 0).any() or abs(result.sum() - 1) > 1e-5:
        raise ValueError("Invalid target distribution")
    return result


def score_predictions(rows, logits, temperature=1.):
    """Coverage-aware exact-max-tie scoring; ordinal/preference ties kept separate."""
    import numpy as np
    if len(rows) != len(logits) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Prediction count or temperature invalid")
    outcomes = []; groups = defaultdict(list)
    for row, values in zip(rows, logits):
        target = target_distribution(row)
        result = {"id": row["id"], "group": row["group"], "family": row["family"],
                  "covered": values is not None, "credit": 0., "nll": None, "brier": None,
                  "ordinal_mae": None, "predicted_index": None,
                  "categorical": row.get("metric", "categorical") not in {"ordinal", "preference_tie"}}
        if values is not None:
            z = np.asarray(values, dtype=np.float64)
            if z.shape != target.shape or not np.isfinite(z).all(): raise ValueError("Invalid ranking logits")
            maxima = np.flatnonzero(z == z.max()); result["credit"] = float(target[maxima].mean())
            result["predicted_index"] = int(maxima[0]); scaled = (z - z.max()) / temperature
            logp = scaled - np.log(np.exp(scaled).sum()); p = np.exp(logp)
            result["nll"] = -float((target * logp).sum()); result["brier"] = float(((p - target) ** 2).sum())
            if row.get("metric") == "ordinal":
                values = np.asarray(row.get("candidate_values", list(range(len(target)))), dtype=np.float64)
                result["ordinal_mae"] = abs(float((p * values).sum()) - float(row["expected_rating"]))
        groups[row["family"]].append(result); outcomes.append(result)
    def average(items, key):
        values = [x[key] for x in items if x[key] is not None]
        return float(np.mean(values)) if values else None
    families = {}
    for family, values in groups.items():
        categorical = [x for x in values if x["categorical"]]
        families[family] = {"count": len(values), "prompt_groups": len({x["group"] for x in values}),
            "classification_count": len(categorical), "accuracy": average(categorical, "credit"),
            "coverage": sum(x["covered"] for x in values) / len(values),
            "nll_covered": average(values, "nll"), "brier_covered": average(values, "brier"),
            "ordinal_mae_covered": average(values, "ordinal_mae")}
    primary = [families[name]["accuracy"] for name in PRIMARY if name in families and families[name]["accuracy"] is not None]
    return {"by_family": families, "primary_macro": float(np.mean(primary)) if len(primary) == len(PRIMARY) else None,
            "primary_families": list(PRIMARY), "coverage": sum(x["covered"] for x in outcomes) / len(outcomes) if outcomes else 0.,
            "count": len(rows), "accuracy_policy": "Equal primary-family mean; failed cases receive zero; exact predicted ties average gold credit"}, outcomes


def fit_temperature(rows, logits):
    """One positive temperature per system, fitted on primary calibration only."""
    import numpy as np
    if len(rows) != len(logits): raise ValueError("Calibration count mismatch")
    examples = defaultdict(list)
    for row, values in zip(rows, logits):
        if row["family"] not in PRIMARY or values is None: continue
        z = np.asarray(values, dtype=np.float64); target = target_distribution(row)
        if z.shape != target.shape or not np.isfinite(z).all(): raise ValueError("Invalid calibration logits")
        examples[row["family"]].append((z - z.max(), target))
    if set(examples) != set(PRIMARY): raise RuntimeError("Each primary family needs covered calibration examples")
    grid = np.logspace(-1, 1, 121); losses = []
    for temp in grid:
        per_family = []
        for family in PRIMARY:
            values = []
            for z, target in examples[family]:
                scaled = z / temp; logp = scaled - np.log(np.exp(scaled).sum())
                values.append(-float((target * logp).sum()))
            per_family.append(float(np.mean(values)))
        losses.append(float(np.mean(per_family)))
    best = int(np.argmin(losses))
    return {"temperature": float(grid[best]), "equal_primary_family_nll": losses[best],
            "fit_partition": "calibration", "covered_by_family": {k: len(v) for k, v in examples.items()},
            "grid": {"minimum": .1, "maximum": 10., "points": 121}, "changes_ranking": False}


def paired_bootstrap(rows, outcomes, replicates=1500):
    import numpy as np
    if set(outcomes) != set(SYSTEMS): raise ValueError("All three paired systems are required")
    for system, values in outcomes.items():
        if [x["id"] for x in values] != [r["id"] for r in rows]: raise ValueError("Paired population mismatch: " + system)
    clusters = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row["family"] in PRIMARY: clusters[row["family"]][row["group"]].append(index)
    if set(clusters) != set(PRIMARY): raise ValueError("Both primary families are required")
    rng = np.random.default_rng(2704905); samples = {name: [] for name in SYSTEMS}; point = {name: [] for name in SYSTEMS}
    by_family = {}
    for family in PRIMARY:
        indices = list(clusters[family].values()); counts = np.asarray([len(x) for x in indices])
        draws = rng.integers(0, len(indices), size=(replicates, len(indices))); scores = {}; means = {}
        for name in SYSTEMS:
            credits = np.asarray([item["credit"] for item in outcomes[name]])
            sums = np.asarray([credits[index].sum() for index in indices]); means[name] = float(sums.sum() / counts.sum())
            scores[name] = sums[draws].sum(1) / counts[draws].sum(1)
            samples[name].append(scores[name]); point[name].append(means[name])
        differences = {name: {"delta": means["selected"] - means[name],
                        "ci95": np.quantile(scores["selected"] - scores[name], [.025, .975]).tolist()} for name in BASELINES}
        adaptive = scores["selected"] - np.maximum(scores[BASELINES[0]], scores[BASELINES[1]])
        by_family[family] = {"clusters": len(indices), "point_accuracy": means,
            "selected_accuracy_ci95": np.quantile(scores["selected"], [.025, .975]).tolist(),
            "versus_each_baseline": differences,
            "versus_stronger_baseline": {"delta": means["selected"] - max(means[x] for x in BASELINES),
                "ci95": np.quantile(adaptive, [.025, .975]).tolist()}}
    means = {name: np.mean(value, axis=0) for name, value in samples.items()}
    point_macro = {name: float(np.mean(value)) for name, value in point.items()}
    adaptive = means["selected"] - np.maximum(means[BASELINES[0]], means[BASELINES[1]])
    return {"replicates": replicates, "seed": 2704905, "by_family": by_family, "point_macro": point_macro,
        "versus_each_baseline": {name: {"delta": point_macro["selected"] - point_macro[name],
            "ci95": np.quantile(means["selected"] - means[name], [.025, .975]).tolist()} for name in BASELINES},
        "versus_stronger_baseline": {"delta": point_macro["selected"] - max(point_macro[x] for x in BASELINES),
            "ci95": np.quantile(adaptive, [.025, .975]).tolist()},
        "method": "Paired prompt-group bootstrap, stratified by primary family; stronger baseline recomputed per replicate",
        "scope": "Primary real tasks only; synthetic challenge excluded from macro and bootstrap"}


def finalize_results(rows, predictions, calibration, recipe):
    if set(predictions) != set(SYSTEMS): raise ValueError("Final comparison needs all three systems")
    if len({row["id"] for row in rows}) != len(rows): raise ValueError("Duplicate final IDs")
    primary_groups = {row["group"] for row in rows if row["family"] in PRIMARY}
    if len(primary_groups) < 1000: raise RuntimeError("Fresh final needs at least 1000 primary prompt groups")
    raw = {}; calibrated = {}; outcomes = {}
    for name in SYSTEMS:
        raw[name], outcomes[name] = score_predictions(rows, predictions[name])
        calibrated[name], _ = score_predictions(rows, predictions[name], calibration[name]["temperature"])
    uncertainty = paired_bootstrap(rows, outcomes); qualified = {}; targets = recipe["release_targets"]
    for family in PRIMARY:
        m = raw["selected"]["by_family"][family]; interval = uncertainty["by_family"][family]["versus_stronger_baseline"]["ci95"]
        checks = {"accuracy_target_met": m["accuracy"] >= targets[family],
                  "coverage_target_met": m["coverage"] >= targets["minimum_primary_coverage"],
                  "positive_paired_improvement": interval[0] > 0}
        qualified[family] = {"qualified": all(checks.values()), "checks": checks,
                             "target": targets[family], "accuracy": m["accuracy"]}
    return {"status": "complete", "scope": recipe["scope"], "count": len(rows),
        "primary_prompt_groups": len(primary_groups), "raw": raw, "calibrated": calibrated,
        "paired_uncertainty": uncertainty, "release_criteria": qualified,
        "qualified_capabilities": [k for k, v in qualified.items() if v["qualified"]],
        "release_status": "targets_met" if all(v["qualified"] for v in qualified.values()) else "experimental",
        "synthetic_challenge": {name: {family: values for family, values in raw[name]["by_family"].items()
                                      if family in {"rule_compliance", "priority_selection"}} for name in SYSTEMS},
        "repeatability_established": recipe.get("repeatability_established", False),
        "seeds": recipe["seeds"], "publication_authorized": True, "benchmark_issue_submission_authorized": False,
        "limitations": ["Fresh prompts in previously seen source domains, not unseen intents or unseen domains",
                        "BANKING77 uses four supplied choices with lexical hard alternatives; not 77-way classification or arbitrary tool routing",
                        "Evidence relation is a class-balanced filtered MultiNLI task, not general evidence verification",
                        "Public pretraining overlap is unknown", "Synthetic challenge is reported separately",
                        "Targets are descriptive release criteria, not guarantees on arbitrary user requests"]}


def _read_rows(path):
    with Path(path).open() as stream: rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows): raise ValueError("Empty or duplicate evaluation population")
    for row in rows:
        for key in ("id", "group", "family", "source", "state", "candidates"): row[key]
        candidates = row["candidates"]
        if not 2 <= len(candidates) <= 64 or len(set(candidates)) != len(candidates) or any(not isinstance(x, str) or not x.strip() for x in candidates):
            raise ValueError("Invalid candidate population")
        target_distribution(row)
    return rows


def _prepare(rows, tokenizer, limits):
    from common import schema_state
    for row in rows:
        row["rendered_state"] = schema_state(row["state"], row.get("question", ""))
        lengths = [len(tokenizer(x, add_special_tokens=True)["input_ids"]) for x in [row["rendered_state"], *row["candidates"]]]
        row["eligible"] = lengths[0] <= limits["state"] and max(lengths[1:]) <= limits["candidate"]
        row["token_lengths"] = lengths


class JointScorer:
    def __init__(self, base, head, limits):
        from joint_deployment import GemmaJointRanker
        self.directory = tempfile.TemporaryDirectory(prefix="v4-frozen-serving-"); folder = Path(self.directory.name)
        for path in Path(base).iterdir():
            if path.is_file() and path.suffix in {".json", ".safetensors", ".model"} and path.name != "joint_config.json":
                (folder / path.name).symlink_to(path.resolve())
        head_target = folder / "evaluation_head.safetensors"; head_target.symlink_to(Path(head).resolve())
        save(folder / "joint_config.json", {"head_config": HEAD_CONFIG, "head_file": head_target.name,
             "max_state_tokens": limits["state"], "max_action_tokens": limits["candidate"]})
        self.ranker = GemmaJointRanker(folder, device="cuda")
        self.tokenizer = self.ranker.tokenizer

    def logits(self, row, deadline):
        result = self.ranker.rank(row["state"], row["candidates"], row.get("question", ""))
        values = {int(item["candidate"]): item["score"] for item in result}
        return [values[index] for index in range(len(row["candidates"]))]

    def close(self): self.directory.cleanup()


class Deadline(Exception): pass


class LikelihoodScorer:
    def __init__(self, base):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)

    def logits(self, row, deadline):
        import torch
        from torch.nn import functional as F
        prefix = self.tokenizer(row["rendered_state"] + "\n\nCandidate response:\n", add_special_tokens=True)["input_ids"]
        result = []
        for candidate in row["candidates"]:
            if time.monotonic() > deadline - 25: raise Deadline()
            action = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if not action: raise ValueError("Empty likelihood candidate")
            ids = torch.tensor([prefix + action], device="cuda")
            hidden = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
            hidden = hidden[len(prefix)-1:len(prefix)+len(action)-1]; total = 0.
            for offset in range(0, len(action), 16):
                z = F.linear(hidden[offset:offset+16], self.encoder.get_input_embeddings().weight)
                cap = getattr(self.encoder.config, "final_logit_softcapping", None)
                if cap is not None: z = (z / cap).tanh() * cap
                z = z.float(); labels = torch.tensor(action[offset:offset+16], device="cuda")
                if not torch.isfinite(z).all(): raise RuntimeError("Non-finite likelihood logits")
                total += float((z[torch.arange(len(labels), device="cuda"), labels] - torch.logsumexp(z, -1)).sum())
            result.append(total / len(action))
        return result

    def close(self): pass


def _progress(output, phase, system, rows, identity):
    path = output / f"{phase}-{system}-predictions.json"
    header = {"phase": phase, "system": system, "identity_sha256": json_sha(identity),
              "rows_sha256": json_sha([{k: r[k] for k in r if k not in {"eligible", "rendered_state", "token_lengths"}} for r in rows])}
    if path.exists():
        progress = json.loads(path.read_text())
        if progress["header"] != header or [x["id"] for x in progress["rows"]] != [r["id"] for r in rows]:
            raise RuntimeError("Resume identity/population mismatch")
    else:
        progress = {"header": header, "rows": [{"id": row["id"], "status": "pending", "logits": None} for row in rows]}
        save(path, progress)
    return path, progress


def _complete(progress): return all(x["status"] != "pending" for x in progress["rows"])


def _phase(rows, phase, systems, identity, recipe, output, deadline):
    import torch
    root, base, v3 = paths(); result = {}
    for system in systems:
        path, progress = _progress(output, phase, system, rows, identity); result[system] = progress
        if _complete(progress): continue
        if time.monotonic() > deadline - 60: break
        if system == "gemma_likelihood": model = LikelihoodScorer(base)
        elif system == "v3_frozen_joint": model = JointScorer(base, v3 / "joint/best/head.safetensors", recipe["limits"])
        else:
            checkpoint = inside(root, recipe["selected"]["checkpoint_dir"])
            model = JointScorer(checkpoint, checkpoint / recipe["selected"]["head_file"], recipe["limits"])
        try:
            _prepare(rows, model.tokenizer, recipe["limits"]); last_save = time.monotonic()
            with torch.inference_mode():
                for index, (row, record) in enumerate(zip(rows, progress["rows"])):
                    if record["status"] != "pending": continue
                    if time.monotonic() > deadline - 30: break
                    if not row["eligible"]: record.update(status="length_rejection", logits=None)
                    else:
                        try:
                            values = model.logits(row, deadline)
                            if len(values) != len(row["candidates"]) or not all(math.isfinite(x) for x in values):
                                raise RuntimeError("Non-finite or incomplete candidate scores")
                            record.update(status="ok", logits=values)
                        except Deadline: break
                        except torch.cuda.OutOfMemoryError:
                            gc.collect(); torch.cuda.empty_cache(); record.update(status="runtime_failure", logits=None, reason="cuda_out_of_memory")
                    if time.monotonic() - last_save > 20 or index % 32 == 0:
                        save(path, progress); last_save = time.monotonic()
                        print(f"v4 {phase} {system}: {sum(x['status']!='pending' for x in progress['rows'])}/{len(rows)}", flush=True)
        finally:
            save(path, progress); model.close(); del model; gc.collect(); torch.cuda.empty_cache()
        if not _complete(progress): break
    for name in systems:
        if name not in result: _, result[name] = _progress(output, phase, name, rows, identity)
    return result


def _partial(output, phase, progress):
    report = {"status": "partial", "phase": phase, "accuracy_reported": False,
              "systems": {name: dict(Counter(x["status"] for x in p["rows"])) for name, p in progress.items()},
              "reason": "Pending cases retained; resume the unchanged attempt"}
    save(output / "metrics.json", report); return report


def _cloud():
    import modal
    import torch
    if modal.is_local() or not torch.cuda.is_available(): raise RuntimeError("Model evaluation is Modal CUDA only")


def run_development(seconds=900):
    _cloud(); root, base, v3 = paths(); started = time.monotonic(); deadline = started + seconds
    if (root / "final-evaluation/final-labels-opened.json").exists(): raise RuntimeError("No development model selection after final access")
    output = root / "development-baselines"; output.mkdir(parents=True, exist_ok=True)
    identity = {"development_sha256": sha(root / "data/development.jsonl"), "initial_base_files": model_files(base),
                "v3_head_sha256": sha(v3 / "joint/best/head.safetensors"), "source_sha256": source_files()}
    rows = _read_rows(root / "data/development.jsonl")
    progress = _phase(rows, "development", BASELINES, identity, {"limits": DEFAULT_LIMITS}, output, deadline)
    if not all(_complete(x) for x in progress.values()): return _partial(output, "development", progress)
    report = {"status": "complete", "development_only": True, "identity": identity,
        "systems": {name: score_predictions(rows, [x["logits"] for x in p["rows"]])[0] for name, p in progress.items()},
        "final_or_calibration_opened": False, "seconds": time.monotonic() - started}
    save(output / "metrics.json", report); return report


def run(seconds=1680):
    _cloud(); started = time.monotonic(); deadline = started + seconds; root, _, _ = paths()
    recipe, identity = validate_recipe(); output = root / "final-evaluation"
    calibration_path = output / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration["identity_sha256"] != json_sha(identity): raise RuntimeError("Calibration identity changed")
    else:
        rows = _read_rows(root / "data/calibration.jsonl")
        progress = _phase(rows, "calibration", SYSTEMS, identity, recipe, output, deadline)
        if not all(_complete(x) for x in progress.values()): return _partial(output, "calibration", progress)
        calibration = {"identity_sha256": json_sha(identity), "fit_partition": "calibration",
            "systems": {name: fit_temperature(rows, [x["logits"] for x in p["rows"]]) for name, p in progress.items()}}
        save(calibration_path, calibration)
    if time.monotonic() > deadline - 60:
        report = {"status": "partial", "phase": "calibration_complete", "accuracy_reported": False}
        save(output / "metrics.json", report); return report
    marker = output / "final-labels-opened.json"
    identity_final = {"identity_sha256": json_sha(identity), "calibration_sha256": sha(calibration_path),
                      "policy": "No further checkpoint or recipe selection using this final set"}
    if marker.exists() and json.loads(marker.read_text()) != identity_final: raise RuntimeError("Final identity changed")
    save(marker, identity_final)
    rows = _read_rows(root / "data/final.jsonl")
    progress = _phase(rows, "final", SYSTEMS, identity, recipe, output, deadline)
    if not all(_complete(x) for x in progress.values()): return _partial(output, "final", progress)
    predictions = {name: [x["logits"] for x in p["rows"]] for name, p in progress.items()}
    report = finalize_results(rows, predictions, calibration["systems"], recipe)
    report.update(identity=identity, calibration_sha256=sha(calibration_path), seconds=time.monotonic() - started,
                  outcomes_by_status={name: dict(Counter(x["status"] for x in p["rows"])) for name, p in progress.items()})
    save(output / "metrics.json", report); return report
