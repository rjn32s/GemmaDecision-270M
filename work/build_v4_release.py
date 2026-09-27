"""Build the authorized v4 model/source release from explicit verified artifacts.

No network, publishing, model execution, or credential reads. Place the downloaded
modified encoder at outputs/GemmaDecision-270M-v4/model.safetensors first, or pass
--weights. Existing GemmaDecision-270M and GemmaDecision-270M-local stay untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "GemmaDecision-270M-v4"
SOURCE_NAME = "GemmaDecision-270M-source"
INFERENCE_REQUIREMENTS = "torch==2.14.0\ntransformers==5.17.0\nsafetensors==0.8.0\nnumpy==2.4.6\nhuggingface_hub==1.33.0\n"
TRAINING_REQUIREMENTS = INFERENCE_REQUIREMENTS + "pyarrow==25.0.1\nijson==3.4.0\nmodal==1.5.5\n"
PRIMARY = ("intent_routing", "evidence_relation")
LABELS = {"intent_routing": "Four-way banking topic selection", "evidence_relation": "Three-way NLI evidence relation"}
SYSTEM_NAMES = {"selected": "GemmaDecision v4", "gemma_likelihood": "Original Gemma likelihood", "v3_frozen_joint": "Frozen v3 joint ranker"}
SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})")
DOCS = ("DATA_PROTOCOL.md", "GPU_PROFILE_RESULTS.md", "RELEASE_ATTRIBUTION_AUDIT.md")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while value := stream.read(8 * 1024 * 1024): digest.update(value)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def safe_relative(root, name):
    name = Path(name)
    if name.is_absolute(): raise RuntimeError("Manifest contains an absolute path")
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise RuntimeError("Manifest path escapes the release root")
    return path


def copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if source.is_symlink(): raise RuntimeError("Release inputs must be regular files: " + str(source))
    digest = sha(source)
    if expected is not None and digest != expected: raise RuntimeError("Artifact checksum mismatch: " + source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_symlink(): raise RuntimeError("Refusing to overwrite a release symlink")
    if source.resolve() != destination.resolve() and (not destination.exists() or sha(destination) != digest):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if sha(temporary) != digest: raise RuntimeError("Release copy checksum mismatch")
        temporary.replace(destination)
    return digest


def pct(value): return "n/a" if value is None else f"{100 * value:.2f}%"


def points(value): return f"{100 * value:+.2f} pp"


def interval(value): return f"[{points(value[0])}, {points(value[1])}]"


def validate_split_identities(document, expected_source_sha256):
    if set(document) != {"source_jsonl_sha256", "contains_raw_text_or_labels", "rows"}:
        raise RuntimeError("Unexpected split-identity document fields")
    if document["source_jsonl_sha256"] != expected_source_sha256 or document["contains_raw_text_or_labels"] is not False:
        raise RuntimeError("Split identities do not match the audited source manifest")
    if not isinstance(document["rows"], list) or not document["rows"]:
        raise RuntimeError("Empty or invalid split identities")
    allowed = {"id", "group", "workflow_group", "source", "family", "split"}
    for row in document["rows"]:
        if not isinstance(row, dict) or set(row) != allowed or any(type(value) is not str for value in row.values()):
            raise RuntimeError("Split identities must contain exactly six allowed string fields")
        if row["split"] not in {"train", "development", "calibration", "final"}:
            raise RuntimeError("Unknown split identity partition")
    return document


def public_predictions(document, split_identities):
    if set(document) != {"header", "rows"} or not isinstance(document["header"], dict):
        raise RuntimeError("Unexpected prediction document fields")
    header = document["header"]
    allowed_header = {"phase", "system", "identity_sha256", "rows_sha256"}
    if set(header) != allowed_header or any(type(value) is not str for value in header.values()):
        raise RuntimeError("Unexpected prediction provenance fields")
    if header["phase"] not in {"calibration", "final"} or header["system"] not in SYSTEM_NAMES:
        raise RuntimeError("Prediction file is outside the frozen three-system evaluation")
    if not isinstance(document["rows"], list): raise RuntimeError("Invalid prediction rows")
    allowed_ids = {row["id"] for row in split_identities["rows"] if row["split"] == header["phase"]}
    result = []; seen = set()
    for row in document["rows"]:
        if not isinstance(row, dict) or not {"id", "status", "logits"}.issubset(row) or not set(row).issubset({"id", "status", "logits", "reason", "failure_reason"}):
            raise RuntimeError("Prediction row contains unexpected or label-bearing fields")
        if type(row["id"]) is not str or row["id"] not in allowed_ids or row["id"] in seen:
            raise RuntimeError("Prediction ID is not a unique member of the published partition")
        seen.add(row["id"]); status = row["status"]; values = row["logits"]
        if status == "ok":
            if not isinstance(values, list) or not 2 <= len(values) <= 64 or any(type(x) not in {int, float} or not math.isfinite(x) for x in values):
                raise RuntimeError("Invalid published candidate scores")
        elif status not in {"length_rejection", "runtime_failure"} or values is not None:
            raise RuntimeError("Incomplete or invalid prediction status")
        # Failure details are not needed for public auditing; retain only the
        # requested ID, scores and status, never arbitrary exception strings.
        result.append({key: row[key] for key in ("id", "logits", "status")})
    if seen != allowed_ids: raise RuntimeError("Prediction file omits part of the audited partition")
    return {"header": header, "rows": result}


def check_tree(folder, model=False):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.is_symlink(): raise RuntimeError("Release symlink: " + str(path))
        if not path.is_file(): continue
        relative = path.relative_to(folder)
        if any(part in {".git", ".env", "__pycache__", "downloads", "cache"} for part in relative.parts):
            raise RuntimeError("Private/cache directory in release: " + str(relative))
        if path.suffix in {".jsonl", ".parquet", ".arrow", ".pt", ".pyc"} or any(
                token in path.name.lower() for token in ("hf_token", "credential", "optimizer")):
            raise RuntimeError("Disallowed private artifact: " + str(relative))
        if path.stat().st_size > 90_000_000 and not (model and relative == Path("model.safetensors")):
            raise RuntimeError("Unexpected large release file: " + str(relative))
        if not model and path.suffix in {".safetensors", ".bin", ".model"}:
            raise RuntimeError("Weights/tokenizer binary in the source-only release")
        if path.suffix in {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}:
            if SECRET_PATTERN.search(path.read_text()): raise RuntimeError("Credential-like value in " + str(relative))
        if relative != Path("SHA256SUMS.json"):
            result[str(relative)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def data_provenance(audit, training):
    lines = ["# Data provenance", "",
        "The fully tuned model starts from official Gemma and the previously trained v3 joint scalar head. Attribution covers direct v4 data and that head's earlier training lineage. Raw datasets and private split files are not redistributed.", "",
        "Actual prepared counts are below; optimization uses the eligible rows recorded in the training manifest. These are measured counts, not requested quotas.", "",
        "| Split | Cases | Prompt groups | Real primary prompt groups |", "|---|---:|---:|---:|"]
    for name, item in audit["splits"].items():
        lines.append(f"| {name} | {item['cases']:,} | {item['prompt_groups']:,} | {item.get('real_primary_prompt_groups', 0):,} |")
    lines += ["", "| Prepared training family | Cases |", "|---|---:|"]
    for name, count in sorted(audit["splits"]["train"]["families"].items()): lines.append(f"| {name} | {count:,} |")
    lines += ["", f"The selected training run records {training['manifest']['training_rows']:,} eligible training rows.", "",
        "BANKING77 examples become four-way topic choices with lexical hard alternatives and shuffled candidate order. MultiNLI examples become support/unresolved/contradiction choices, preserving premise groups and class balancing. Counterfactual policy pairs have mechanically checked labels and remain together by group; held-out templates are correlated diagnostics. Replay uses previous training partitions only, including soft preference labels.", "",
        "New holdouts exclude prior v2/v3 prompt groups and normalized inputs. Approximate lexical screens provide additional protection, without proving semantic independence. All banking intents and the selected NLI genres were previously encountered during development; these are fresh prompts within known domains. Public pretraining overlap is unknown.", "",
        "Source licenses, authors, revisions, transformations and inherited sources are listed in [RELEASE_ATTRIBUTION_AUDIT.md](RELEASE_ATTRIBUTION_AUDIT.md). The precise split protocol is in [DATA_PROTOCOL.md](DATA_PROTOCOL.md), with measured audit evidence in [evidence/data-audit.json](evidence/data-audit.json). Source terms remain separate from the model's Gemma terms and the code's Apache-2.0 license.", ""]
    return "\n".join(lines)


def results_text(evaluation, verification, training):
    raw = evaluation["raw"]; selected = raw["selected"]; paired = evaluation["paired_uncertainty"]
    lines = ["# Measured results", "", "This is a private fresh-prompt project evaluation. It is not an official BANKING77, MultiNLI or JevBench score.", "",
        "| Task | Original Gemma likelihood | Frozen v3 joint ranker | GemmaDecision v4 | Cases |", "|---|---:|---:|---:|---:|"]
    for family in PRIMARY:
        values = [raw[name]["by_family"][family] for name in ("gemma_likelihood", "v3_frozen_joint", "selected")]
        lines.append(f"| {LABELS[family]} | {pct(values[0]['accuracy'])} | {pct(values[1]['accuracy'])} | {pct(values[2]['accuracy'])} | {values[2]['count']} |")
    lines += [f"| Equal-family primary mean | {pct(raw['gemma_likelihood']['primary_macro'])} | {pct(raw['v3_frozen_joint']['primary_macro'])} | {pct(selected['primary_macro'])} | — |", "",
        f"The final set contains {evaluation['primary_prompt_groups']:,} primary prompt groups. Rejections stay in accuracy denominators. All-population coverage for v4 is {pct(selected['coverage'])}; per-family coverage and calibrated NLL/Brier are in `evaluation.json`.", "",
        "| Paired comparison against stronger baseline | Accuracy difference | 95% interval |", "|---|---:|---:|"]
    for family in PRIMARY:
        comparison = paired["by_family"][family]["versus_stronger_baseline"]
        lines.append(f"| {LABELS[family]} | {points(comparison['delta'])} | {interval(comparison['ci95'])} |")
    comparison = paired["versus_stronger_baseline"]
    lines += [f"| Equal-family mean | {points(comparison['delta'])} | {interval(comparison['ci95'])} |", "",
        f"Intervals use {paired['replicates']:,} paired prompt-group bootstrap resamples within each primary family. The stronger of the original-Gemma and frozen-v3 baselines is recomputed within every resample. They quantify uncertainty for this constructed population, not arbitrary real-world decisions.", "",
        "| Frozen release target | Accuracy target | Observed | All checks |", "|---|---:|---:|---|"]
    for family in PRIMARY:
        item = evaluation["release_criteria"][family]
        lines.append(f"| {LABELS[family]} | {pct(item['target'])} | {pct(item['accuracy'])} | {'Met' if item['qualified'] else 'Not met'} |")
    qualified = [LABELS[name] for name in evaluation.get("qualified_capabilities", [])]
    lines += ["", "All checks combine the accuracy target, coverage requirement and positive lower paired interval against the stronger baseline. These thresholds were fixed before this final evaluation; they are not promises of deployment quality.", "",
        "Capabilities satisfying those checks: " + (", ".join(qualified) if qualified else "none") + ".",
        f"Release status from this protocol: **{evaluation['release_status']}**. The repository remains a research model with the documented limits.", "",
        "Synthetic challenge performance is separate from the primary mean:", "", "| System | Challenge | Accuracy | Cases |", "|---|---|---:|---:|"]
    for name, families in evaluation.get("synthetic_challenge", {}).items():
        for family, item in families.items(): lines.append(f"| {SYSTEM_NAMES[name]} | {family} | {pct(item['accuracy'])} | {item['count']} |")
    lines += ["", "Synthetic cases come from a small number of policy templates and include correlated pairs. Their case count is not a count of independent workflows.", "",
        f"The selected run's best checkpoint is update {training['best']['step']:,}; its completed run reached {training['steps']:,} updates. Selection used development accuracy with NLL tie-breaking. Calibration used a separate partition and does not change candidate ordering. Final outcomes did not select the checkpoint.", "",
        f"Development seeds recorded for final evaluation: {evaluation.get('seeds', [])}. Repeatability established: {evaluation.get('repeatability_established', False)}. Different seed runs, when included in evidence, are development diagnostics rather than proof of replicated final gains.", "",
        "Offline CPU verification passed software checks for loading with no network, independent reload, candidate reversal/addition invariance and explicit overlength rejection. These checks do not assert correct decisions. The following examples were already exposed during development and are not fresh benchmarks:", "",
        "| Illustrative example | Actual top choice | Expected choice | Match | CPU request seconds |", "|---|---|---|---|---:|"]
    for item in verification["examples"]:
        lines.append(f"| {item['name']} | {item['ranking'][0]['candidate']} | {item['illustrative_expected_top']} | {'Yes' if item['illustrative_top_matches_expectation'] else 'No'} | {item['cpu_seconds']:.3f} |")
    lines += ["", f"Cold model load: {verification.get('cold_model_load_seconds', 0):.3f} seconds on the recorded Modal CPU worker. Latencies describe two CPU threads and these short requests; they are not measurements from the user's Mac. MPS and CUDA-to-CPU ranking equivalence are not certified by this CPU test.", ""]
    return "\n".join(lines)


def model_card(evaluation, verification, training, config, hf_repo, revision, github_repo):
    raw = evaluation["raw"]; all_met = evaluation["release_status"] == "targets_met"
    status = "Both frozen task targets were met on this private test." if all_met else "One or more frozen task targets were not met; this is an experimental release."
    return f"""---
license: gemma
base_model: google/gemma-3-270m
pipeline_tag: text-ranking
library_name: pytorch
language:
- en
tags:
- gemma3
- candidate-ranking
- text-classification
- experimental
---

# GemmaDecision-270M v4

A fully tuned Gemma-270M candidate ranker: it scores each supplied choice together with the request, using a scalar decision head. Intended research tasks are banking-support **topic selection** and three-way evidence relations. It does not generate answers. The complete modified encoder and head are included.

On the private fresh-prompt test, accuracy is **{pct(raw['selected']['by_family']['intent_routing']['accuracy'])}** for four-way banking topics and **{pct(raw['selected']['by_family']['evidence_relation']['accuracy'])}** for three-way NLI, with an equal-family mean of **{pct(raw['selected']['primary_macro'])}**. {status} See [RESULTS.md](RESULTS.md) for both baselines, paired uncertainty, coverage, synthetic challenge results and failures.

## Run on CPU

From a downloaded complete model directory:

```sh
python -m pip install -r requirements.txt
python joint_deployment.py --model . --input example_request.json --device cpu
```

To download the versioned release and use the Python interface:

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id={hf_repo!r}, revision={revision!r})
sys.path.insert(0, path)
from joint_deployment import GemmaDecisionRanker

model = GemmaDecisionRanker(path, device="cpu")
ranking = model.rank(
    "The same card payment appears twice on my statement.",
    {{"payments": "Investigate charges, transfers and refunds.",
     "technical": "Investigate app crashes and login errors."}},
    question="Which support team should handle this request?",
)
print(ranking)
```

After download, inference uses local files only. `AutoModelForCausalLM` alone does not apply the decision head. The helper supports CPU, CUDA and MPS selection; only the devices described in the evidence have been tested. Candidates may be a list of texts or a label-to-text mapping, with 2–64 distinct choices. The limits are {config['max_state_tokens']} state/question tokens and {config['max_action_tokens']} tokens per choice. Long inputs raise an error; they are not silently shortened.

Scores are raw, uncalibrated rankings. They are not probabilities, cannot be compared across requests, and do not establish that the winning choice is correct. The calibration reported in `evaluation.json` is an evaluation diagnostic; the public `rank()` interface does not apply it.

## What this evaluation means

The banking task supplies the correct topic and three lexical hard alternatives. It is not official 77-way BANKING77 classification or arbitrary tool routing. The evidence task is a class-balanced filtered MultiNLI construction. New prompt groups exclude previously used v2/v3 groups and inputs, but source domains and all banking intents were already encountered. Public foundation-model pretraining overlap is unknown. Synthetic policy challenges are reported separately.

This release makes **no JEV/CLM parity, general decision-reliability, or official JevBench claim**. Use the error reports and documented task scope when assessing a use case.

## Training and provenance

The encoder starts from `google/gemma-3-270m` revision `{config['base_revision']}` and the scalar head starts from the project's v3 joint checkpoint. Both become trainable. Training uses FP32 master weights and AdamW state with BF16 autocast; the chosen encoder is exported in BF16 and evaluated with an FP32 scalar head, normalized last-token pooling and the exact serving input format. CPU serving loads the encoder in FP32. No quantized or unchanged-base substitute is implied.

Checkpoint selection used development data; temperature fitting used separate calibration data; final cases were opened only after freezing the recipe. See `base-provenance.json`, `evidence/frozen-recipe.json`, [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for hashes and limits. Training code is at [github.com/{github_repo}](https://github.com/{github_repo}). Exact training reconstruction also needs the documented historical split inputs and initialization head; it is not claimed to work from code alone with one command.

## Terms and attribution

The modified model weights are governed by the included [Gemma terms](LICENSE) and [Prohibited Use Policy](GEMMA_PROHIBITED_USE_POLICY.txt). Original and vendored Apache code uses [LICENSE_CODE](LICENSE_CODE); that does not relicense weights or datasets. [NOTICE](NOTICE) identifies modifications and upstream CLM helpers. Training-source attribution includes direct v4 data and inherited v3 head lineage in [RELEASE_ATTRIBUTION_AUDIT.md](RELEASE_ATTRIBUTION_AUDIT.md). Google, CLM and dataset authors do not endorse this project.
"""


def reproducibility_text(training, provenance, hf_repo, revision):
    controls = training["manifest"]["controls"]
    return f"""# Reproducibility boundaries

The release contains working inference artifacts and the source used for the experiment. Exact retraining is conditional on the historical inputs below; no one-command reconstruction from public downloads alone is claimed.

- Initial model: official `google/gemma-3-270m` at `{provenance['initial_revision']}`; exact original file hashes are in `base-provenance.json`. Accept its terms through the normal upstream flow.
- Initial scalar head: `reproduction/initial-v3-joint-head.safetensors` in HF `{hf_repo}` revision `{revision}`, SHA256 `{training['manifest']['initial_head_sha256']}`. It is included for lineage/reconstruction and is not the final inference head.
- Data inputs: pinned public source caches plus the earlier v2/v3 split inputs required by `data_v4.prior_inputs`. The latter determine exclusions and training-only replay. They are not redistributed here. Hashes of those inputs and of the new splits are in `evidence/data-audit.json` and `evidence/frozen-recipe.json`.
- Compute: the recorded Modal training environment, dependency pins and source hashes. A fresh user's Modal workspace/volume and their own accepted model access must replace project-specific volume names; no credentials are included.

The GitHub tree preserves `outputs/gemma-local/v2`, `v3` and `v4`, because the launchers resolve sibling modules by those paths. V4 mounts v2 `common.py`, `clm_schema.py`, `clm_heads.py` and v3 `data.py` as `data_v3.py`, plus `joint_deployment.py`. Changing this topology without adjusting mounts will break imports.

Provision `/experiment/work/model`, `/experiment/work/decision-v2`, and `/experiment/work/decision-v3` to match the recorded hashes, including `/experiment/work/decision-v3/joint/best/head.safetensors`. Only then use the documented v4 launcher stages: `prepare`, `train`, `development`, `freeze`, `evaluate`, `package`. A second development seed uses its separate `train-repeat` namespace. Inspect the launcher's cost guards and configure a fresh budget ledger; the original account's ledger is deliberately not distributed. Freezing binds the chosen checkpoint before calibration/final access, and a final-open marker blocks more selection on that test.

Selected-run controls: `{json.dumps(controls, sort_keys=True)}`. Actual step count, early stopping, seed, source/data hashes and development history are in `evidence/training.json`. Learning rates, candidate cross-entropy and complete request formatting must be preserved. The trainer verifies export precision during selection. GPU timing profiles are throughput measurements, not accuracy experiments.

Published `evidence/split-identities.json` contains only row, prompt-group, workflow, source, family and split identities, bound to the audited source-manifest hash. `evidence/predictions/` contains per-case candidate logits and coverage/failure statuses for calibration and final evaluation. These records aid independent checks of partition membership, paired populations and reported scoring. They contain no raw prompt text or explicit gold-label fields. IDs and scores do not replace the historical inputs needed for exact exclusion/replay reconstruction and retraining.

The earlier v3 final is exposed historical evidence. It cannot be reused as an independent test for a new tuned version. Further changes require newly reserved evaluation prompts. Raw datasets, split JSONL, caches, optimizer state and credentials are absent from both release folders.
"""


def build(root=ROOT, weights=None, hf_repo="rajan2k/GemmaDecision-270M", revision="v0.4.0", github_repo="rjn32s/GemmaDecision-270M"):
    root = Path(root).resolve(); v4 = root / "outputs/gemma-local/v4"; package = v4 / "results/package"
    model = root / "outputs" / MODEL_NAME; source = root / "outputs" / SOURCE_NAME
    weights = Path(weights).resolve() if weights else model / "model.safetensors"
    evaluation_path = v4 / "results/evaluate/metrics.json"; evaluation = read(evaluation_path)
    verification = read(package / "offline-check.json"); provenance = read(package / "base-provenance.json")
    config = read(package / "joint_config.json"); payload = read(package / "SHA256SUMS.json")
    required_payload = {"model.safetensors", "joint_head.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "joint_config.json", "joint_deployment.py", "common.py", "clm_schema.py",
        "clm_heads.py", "base-provenance.json", "offline-check.json", "example_request.json"}
    if not required_payload.issubset(payload["files"]): raise RuntimeError("Cloud model payload is incomplete")
    audit_path = v4 / "results/prepare/metrics.json"; audit = read(audit_path)
    frozen_path = v4 / "results/freeze/frozen-recipe.json"; frozen = read(frozen_path)
    identities_path = v4 / "results/freeze/split-identities.json"
    identities = validate_split_identities(read(identities_path), audit["output_hashes"]["split-manifest.jsonl"])
    if evaluation.get("status") != "complete" or verification.get("status") != "passed":
        raise RuntimeError("Completed final evaluation and passing offline software checks are required")
    if config.get("base_weights_modified") is not True or audit.get("passed") is not True:
        raise RuntimeError("Full-tuning provenance or data audit is missing")
    if evaluation.get("identity", {}).get("recipe_sha256") != sha(frozen_path) or provenance["frozen_recipe_sha256"] != sha(frozen_path):
        raise RuntimeError("Package and final evaluation do not bind the same frozen recipe")
    training_name = Path(frozen["training_report"]).parts[0]
    if training_name not in {"train", "train-repeat"}: raise RuntimeError("Unexpected selected training report path")
    training_path = v4 / "results" / training_name / "metrics.json"; training = read(training_path)
    if sha(training_path) != frozen["training_report_sha256"]: raise RuntimeError("Selected training report changed")
    if training.get("status") != "complete" or not training.get("selected_backbone_updated"):
        raise RuntimeError("Selected checkpoint must be a completed trained export")
    if training["selected_encoder_sha256"] != payload["files"]["model.safetensors"]["sha256"]:
        raise RuntimeError("Packaged encoder differs from the selected training export")
    if not weights.is_file(): raise RuntimeError("Place the downloaded modified model.safetensors at " + str(weights))
    if sha(weights) != payload["files"]["model.safetensors"]["sha256"]: raise RuntimeError("Downloaded encoder checksum mismatch")
    # Validate every cloud payload input before creating other release files.
    for name, item in payload["files"].items():
        path = weights if name == "model.safetensors" else safe_relative(package, name)
        if sha(path) != item["sha256"]: raise RuntimeError("Small package artifact checksum mismatch: " + name)
    model.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True)
    for name, item in payload["files"].items():
        copy(weights if name == "model.safetensors" else safe_relative(package, name), safe_relative(model, name), item["sha256"])
    # The launcher may append function-cost metadata after the package report was
    # generated. The frozen identity binds the scientific report in both copies.
    copy(evaluation_path, model / "evaluation.json")
    old = root / "outputs/GemmaDecision-270M-local"
    for name in ("LICENSE", "LICENSE_CODE", "GEMMA_PROHIBITED_USE_POLICY.txt", "BASE_MODEL_CARD.md"):
        copy(old / name, model / name)
    notice = ("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n\n"
        "This derivative model is governed by the included Gemma Terms of Use and incorporated Prohibited Use Policy.\n"
        "Project modifications: v4 fully fine-tunes the Gemma encoder in model.safetensors and the scalar joint_head.safetensors, and supplies modified configuration and inference/training helpers. The Gemma backbone is not unchanged.\n"
        "The optional reproduction/initial-v3-joint-head.safetensors is an earlier project initialization artifact, not the v4 inference head. See provenance hashes for both.\n"
        "clm_schema.py and clm_heads.py derive from Contrastive-LM/CLM revision bb42c6c5bf914fd449bed2f6ca65be80602cb1f7, under Apache-2.0; see LICENSE_CODE.\n"
        "Original project code is provided under Apache-2.0. This code license does not replace the model's Gemma terms or any dataset's terms. No upstream endorsement is implied.\n")
    (model / "NOTICE").write_text(notice); (model / "requirements.txt").write_text(INFERENCE_REQUIREMENTS)
    evidence = model / "evidence"; evidence.mkdir(exist_ok=True)
    for path, name in ((audit_path, "data-audit.json"), (training_path, "training.json"),
                       (frozen_path, "frozen-recipe.json"), (package / "SHA256SUMS.json", "cloud-payload-hashes.json")):
        copy(path, evidence / name)
    copy(frozen_path, model / "frozen-recipe.json")
    copy(identities_path, evidence / "split-identities.json")
    prediction_paths = sorted((v4 / "results/evaluate").glob("*-predictions.json"))
    expected_predictions = {f"{phase}-{system}-predictions.json" for phase in ("calibration", "final") for system in SYSTEM_NAMES}
    if {path.name for path in prediction_paths} != expected_predictions:
        raise RuntimeError("Both complete phases for all three paired systems must be published")
    prediction_identity = hashlib.sha256(json.dumps(evaluation["identity"], sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
    for path in prediction_paths:
        predictions = public_predictions(read(path), identities)
        if predictions["header"]["identity_sha256"] != prediction_identity:
            raise RuntimeError("Prediction scores belong to another frozen evaluation")
        if path.name != f"{predictions['header']['phase']}-{predictions['header']['system']}-predictions.json":
            raise RuntimeError("Prediction filename and provenance disagree")
        write_json(evidence / "predictions" / path.name, predictions)
    for stage, name in (("development", "development-baselines.json"), ("train-repeat", "training-repeat.json")):
        path = v4 / "results" / stage / "metrics.json"
        if path.exists(): copy(path, evidence / name)
    calibration_path = v4 / "results/evaluate/calibration.json"
    if calibration_path.exists(): copy(calibration_path, evidence / "calibration.json")
    for name in DOCS: copy(v4 / name, model / name)
    copy(v4 / "GPU_CHOICE.json", evidence / "gpu-choice.json")
    initial_head = root / "outputs/gemma-local/v3/results/joint/best/head.safetensors"
    if not initial_head.exists(): initial_head = root / "outputs/gemma-local/v3/results/package_joint/joint_head.safetensors"
    copy(initial_head, model / "reproduction/initial-v3-joint-head.safetensors", training["manifest"]["initial_head_sha256"])
    (model / "DATA_PROVENANCE.md").write_text(data_provenance(audit, training))
    (model / "RESULTS.md").write_text(results_text(evaluation, verification, training))
    (model / "REPRODUCIBILITY.md").write_text(reproducibility_text(training, provenance, hf_repo, revision))
    card = model_card(evaluation, verification, training, config, hf_repo, revision, github_repo)
    training_curve = v4 / "TRAINING_CURVE.svg"
    if training_curve.exists():
        copy(training_curve, model / "TRAINING_CURVE.svg")
        copy(training_curve, source / "TRAINING_CURVE.svg")
        card = card.replace("## Training and provenance\n\n",
                            "## Training and provenance\n\n[Training curve](TRAINING_CURVE.svg)\n\n")
    (model / "README.md").write_text(card)
    write_json(model / "RELEASE.json", {"version": "0.4.0", "hf_repository": hf_repo, "hf_revision": revision,
        "github_repository": github_repo, "release_built_locally": True, "publication_performed_by_builder": False,
        "benchmark_issue_submission_authorized": False, "recipe_sha256": sha(frozen_path),
        "builder_sha256": sha(Path(__file__)), "scientific_release_status": evaluation["release_status"]})
    # Preserve the relative module topology, never recursively copy run folders.
    for version in ("v2", "v3", "v4"):
        directory = root / "outputs/gemma-local" / version
        for path in sorted(directory.glob("*.py")):
            copy(path, source / "outputs/gemma-local" / version / path.name)
    copy(root / "outputs/gemma-local/decision_data.py", source / "outputs/gemma-local/decision_data.py")
    copy(Path(__file__), source / "work/build_v4_release.py")
    for name in DOCS: copy(v4 / name, source / "outputs/gemma-local/v4" / name)
    copy(v4 / "GPU_CHOICE.json", source / "outputs/gemma-local/v4/GPU_CHOICE.json")
    # Source artifact contains no weights. Model-governing terms remain explicit.
    copy(model / "LICENSE_CODE", source / "LICENSE")
    copy(model / "LICENSE", source / "GEMMA_TERMS.txt")
    copy(model / "GEMMA_PROHIBITED_USE_POLICY.txt", source / "GEMMA_PROHIBITED_USE_POLICY.txt")
    (source / "NOTICE").write_text("Original project and vendored CLM code are Apache-2.0.\n"
        "CLM helper source: Contrastive-LM/CLM revision bb42c6c5bf914fd449bed2f6ca65be80602cb1f7.\n"
        "No model weights or raw datasets are included in this source repository. The separately distributed Gemma derivative is governed by its Gemma terms and notices, not this code license.\n")
    (source / "requirements.txt").write_text(TRAINING_REQUIREMENTS)
    for name in ("DATA_PROVENANCE.md", "RESULTS.md", "REPRODUCIBILITY.md", "RELEASE_ATTRIBUTION_AUDIT.md", "DATA_PROTOCOL.md"):
        copy(model / name, source / name)
    for path in evidence.rglob("*.json"): copy(path, source / "evidence" / path.relative_to(evidence))
    copy(model / "evaluation.json", source / "evidence/evaluation.json")
    copy(model / "offline-check.json", source / "evidence/offline-check.json")
    (source / "README.md").write_text(f"""# GemmaDecision-270M v4 source

Code and aggregate evidence for a fully tuned Gemma-270M candidate ranker. The modified model and inference helpers are distributed at [HF {hf_repo}](https://huggingface.co/{hf_repo}/tree/{revision}) under Gemma terms. This GitHub source repository uses Apache-2.0 for code and contains no model weights, datasets, credentials, caches or optimizer state.

See [RESULTS.md](RESULTS.md) for the actual fresh-prompt test, both baselines, uncertainty, failed targets and exposed illustrative examples. No official JevBench result or JEV equivalence is claimed.

The current implementation is in `outputs/gemma-local/v4`; v2/v3 sibling modules are preserved as historical dependencies. V4 learns a context-conditioned scalar score with the entire encoder trainable. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes exact initialization, required historical split inputs, Modal setup and source/data hashes. The repository alone does not recreate those historical inputs, so there is no claimed one-command exact retraining workflow.

Use the model repository's CPU quickstart for inference. Install `requirements.txt` here only when preparing the recorded training/evaluation environment. Review and adapt project-specific Modal volume names and budgets before any cloud execution. Historical benchmark scripts are source history, not an instruction to submit or a v4 benchmark claim.

Attribution and transformations are in [DATA_PROVENANCE.md](DATA_PROVENANCE.md), [DATA_PROTOCOL.md](DATA_PROTOCOL.md) and [RELEASE_ATTRIBUTION_AUDIT.md](RELEASE_ATTRIBUTION_AUDIT.md). The old v3 final is exposed history and must not be reused as an independent test for another tuned version.
""")
    # Bind copied serving/training modules to the frozen scientific recipe.
    sources = {}
    for name, expected in frozen["sources_sha256"].items():
        version = "v2" if name in {"common.py", "clm_schema.py", "clm_heads.py"} else "v3" if name == "joint_deployment.py" else "v4"
        path = source / "outputs/gemma-local" / version / name
        if not path.exists() or sha(path) != expected: raise RuntimeError("Source release differs from frozen code: " + name)
        sources[str(path.relative_to(source))] = expected
    write_json(source / "FROZEN_SOURCE_HASHES.json", {"recipe_sha256": sha(frozen_path), "files": sources})
    write_json(model / "SOURCE_HASHES.json", {"recipe_sha256": sha(frozen_path), "files": sources,
        "github_repository": github_repo, "version": "v0.4.0"})
    source_manifest = check_tree(source)
    write_json(source / "SHA256SUMS.json", {"format": "gemmadecision-v4-source-sha256", "files": source_manifest})
    model_manifest = check_tree(model, model=True)
    write_json(model / "SHA256SUMS.json", {"format": "gemmadecision-v4-release-sha256", "files": model_manifest})
    return {"status": "built", "model_directory": str(model), "source_directory": str(source),
        "model_files": len(model_manifest), "source_files": len(source_manifest),
        "model_sha256": model_manifest["model.safetensors"]["sha256"],
        "scientific_release_status": evaluation["release_status"], "publication_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Verified downloaded modified encoder; defaults to the new model output folder")
    parser.add_argument("--hf-repo", default="rajan2k/GemmaDecision-270M")
    parser.add_argument("--hf-revision", default="v0.4.0")
    parser.add_argument("--github-repo", default="rjn32s/GemmaDecision-270M")
    args = parser.parse_args()
    print(json.dumps(build(weights=args.weights, hf_repo=args.hf_repo, revision=args.hf_revision, github_repo=args.github_repo), indent=2))
