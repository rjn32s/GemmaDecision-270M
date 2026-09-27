"""Run the frozen GemmaDecision v4 endpoint through unmodified public JevBench.

Only the three public task files are loaded. This script does not load a model,
fit parameters, retry requests, compute the official composite, or assign rank.
The endpoint should already be ready; its model loading time is not included
in the official per-decision HTTP latency. All outputs are written exclusively
to a new directory outside the JevBench checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlsplit


HARNESS_REVISION = "d06ee95988da1350eb8ae5511daa0e06bfffe911"
HARNESS_CONTENT_SHA256 = "f33c24f0c0231ba3df9c4fe24ad284d60b014fac5cb42bfe642b13d1176e998d"
MODEL_REVISION = "785d530221c990671f29976902540101bb9c7647"
MODEL_ID = f"rajan2k/GemmaDecision-270M@{MODEL_REVISION}"
TEMPERATURE = 4.136820402388508
TIERS = (("easy", 48), ("original", 72), ("hard", 111))
RESERVATION_USD = 0.02
RESERVATION_LEDGER_CAP_USD = 6.0


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_harness(harness_dir: str | Path) -> dict:
    """Verify source and public data bytes, including a source-only git clone."""
    root = Path(harness_dir).resolve()
    files = sorted(
        list((root / "jevbench").rglob("*.py"))
        + list((root / "datasets" / "public").glob("*.jsonl")),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    file_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(
        "".join(f"{path}\0{sha}\n" for path, sha in file_hashes.items()).encode()
    ).hexdigest()
    if digest != HARNESS_CONTENT_SHA256:
        raise ValueError(
            f"JevBench source/data mismatch: expected {HARNESS_CONTENT_SHA256}, got {digest}; "
            f"use a clean checkout of {HARNESS_REVISION}"
        )
    return {
        "repository": "https://github.com/fstandhartinger/jevbench",
        "revision": HARNESS_REVISION,
        "content_sha256": digest,
        "content_hash_scheme": "SHA256 of sorted relative-path + NUL + file-SHA256 + LF for all jevbench/**/*.py and datasets/public/*.jsonl",
        "files": file_hashes,
    }


def _load_harness(harness_dir: str | Path) -> tuple[dict, dict]:
    root = Path(harness_dir).resolve()
    evidence = verify_harness(root)
    if "jevbench" in sys.modules:
        imported = Path(sys.modules["jevbench"].__file__).resolve()
        if not imported.is_relative_to(root):
            raise ValueError("A different JevBench installation is already imported")
    sys.path.insert(0, str(root))
    modules = {
        name: importlib.import_module(f"jevbench.{name}")
        for name in ("tasks", "runner", "budget", "summarize", "adapters.typesafe")
    }
    return modules, evidence


def load_plan(modules: dict, harness_dir: str | Path) -> tuple[list, dict]:
    root = Path(harness_dir).resolve()
    tasks = []
    tiers = {}
    for tier, required_count in TIERS:
        subset = modules["tasks"].load_jsonl(str(root / "datasets" / "public" / f"{tier}.jsonl"))
        if len(subset) != required_count or any(task.split != "public" for task in subset):
            raise ValueError(f"Unexpected public task plan for {tier}")
        tiers[tier] = subset
        tasks.extend(subset)
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique across all public tiers")
    return tasks, tiers


def _denominators(tasks: list, records: list, official: dict) -> dict:
    by_id = {task.id: task for task in tasks}
    attempted = [record for record in records if record["task_id"] in by_id]
    planned_scorable = [
        task for task in tasks
        if task.expected is not None and not task.provenance.get("exclude_reason")
    ]
    calibrated = [
        record for record in attempted
        if record.get("probs")
        and by_id[record["task_id"]].expected is not None
        and not by_id[record["task_id"]].provenance.get("exclude_reason")
    ]
    return {
        "planned_scorable_n": len(planned_scorable),
        "accuracy_denominator_attempted_scorable_n": official["n_scorable"],
        "accuracy_all_planned_scorable": (
            official["n_correct"] / len(planned_scorable) if planned_scorable else None
        ),
        "accuracy_all_planned_note": "Unattempted tasks are included in this denominator and contribute no correct answers; official accuracy separately uses attempted scorable tasks.",
        "brier_n": len(calibrated),
        "ece_n": official["ece"]["n"] if official["ece"] else 0,
        "ordinal_mae_n": sum(
            by_id[record["task_id"]].question["type"] == "score"
            for record in calibrated
        ),
        "invalid_attempts_n": len(attempted) - official["n_valid"],
        "unattempted_n": len(tasks) - len(attempted),
    }


def make_summary(modules: dict, tasks: list, tiers: dict, records: list,
                 ledger_charged: float, elapsed_s: float) -> tuple[dict, dict]:
    """Keep official aggregates intact and separately make denominators explicit."""
    summarize = modules["summarize"]
    official = summarize.summarize(tasks, records, ledger_charged=None)
    per_tier = {}
    for tier, subset in tiers.items():
        metrics = summarize.metric(subset, records)
        per_tier[tier] = {**metrics, **_denominators(subset, records, metrics)}
    report = {
        **official,
        **_denominators(tasks, records, official),
        "scope": "public-only author-run diagnostic; not an official JevBench composite or leaderboard rank",
        "harness_revision": HARNESS_REVISION,
        "model": MODEL_ID,
        "planned_task_order": [tier for tier, _ in TIERS],
        "per_tier": per_tier,
        "run_elapsed_s": elapsed_s,
        "latency_definition": "Unchanged official Runner wall time around each TypeSafeAdapter HTTP call, including loopback serialization/transport and model execution; excludes model startup. Serial, one attempt per task, no retries.",
        "ordinal_scoring": "At this pinned revision, ordinal accuracy uses probability argmax; ordinal MAE uses continuous expected value. The pinned scoring.py implementation is authoritative over older docstrings.",
        "probability_derivation": {
            "kind": "softmax_ranking_scores_external_temperature",
            "temperature": TEMPERATURE,
            "fit_source": "Frozen project calibration split before this JevBench run; no JevBench temperature fitting",
            "upstream_record_source_note": "Unchanged TypeSafeAdapter reports probs_source=native for all native-protocol responses. This describes transport, not calibrated native model probabilities: GemmaDecision emits ranking scores and this endpoint applies the fixed softmax transform.",
        },
        "reservation_ledger": {
            "cap_usd": RESERVATION_LEDGER_CAP_USD,
            "per_attempt_reserved_usd": RESERVATION_USD,
            "charged_reservations_usd": ledger_charged,
            "is_measured_spend": False,
            "note": "The official Runner retains a $0.02 accounting reservation when the local endpoint has no per-token tariff. These reservations are not Modal billing, decision price, or measured spend. Actual infrastructure accounting is reported separately.",
        },
        "tariff_note": "No per-token API tariff is claimed. Per-item cost_usd and price_per_1000_decisions_usd remain null; unmetered is not free.",
    }
    return official, report


def run(harness_dir: str | Path, output_dir: str | Path,
        endpoint: str = "http://127.0.0.1:8000") -> dict:
    """Run exactly the pinned 231-task public plan once on a ready local server."""
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.scheme != "http" or endpoint_parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This protocol requires a loopback HTTP endpoint in the benchmark container")
    if endpoint_parts.username or endpoint_parts.password or endpoint_parts.query or endpoint_parts.fragment:
        raise ValueError("Endpoint must not contain credentials, a query, or a fragment")
    if endpoint_parts.path not in {"", "/"}:
        raise ValueError("Endpoint must be the server root")
    # TypeSafeAdapter consults these variables even when constructor prices are
    # None. Refuse accidental environment pricing instead of silently changing it.
    if any(os.environ.get(name) for name in ("TYPESAFE_PRICE_INPUT_PER_M", "TYPESAFE_PRICE_OUTPUT_PER_M")):
        raise ValueError("Unset TYPESAFE_PRICE_*: this frozen local protocol has no per-token tariff")
    harness_root = Path(harness_dir).resolve()
    output_root = Path(output_dir).resolve()
    if output_root.is_relative_to(harness_root):
        raise ValueError("Output directory must be outside the JevBench checkout")
    modules, source_evidence = _load_harness(harness_root)
    tasks, tiers = load_plan(modules, harness_root)
    # An existing directory is never resumed or overwritten: an interrupted run
    # retains its artifacts, and the caller must inspect it before any new run.
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "harness_manifest.json", source_evidence)
    _write_json(output_root / "task_plan.json", {
        "harness_revision": HARNESS_REVISION,
        "model": MODEL_ID,
        "n_planned": len(tasks),
        "dataset_hash": modules["tasks"].dataset_hash(tasks),
        "tiers": {
            tier: {"n": len(subset), "task_ids": [task.id for task in subset]}
            for tier, subset in tiers.items()
        },
        "task_ids_in_order": [task.id for task in tasks],
        "temperature": TEMPERATURE,
        "serial": True,
        "retries": 0,
        "per_token_tariff": None,
        "started_unix_s": time.time(),
    })
    adapter = modules["adapters.typesafe"].TypeSafeAdapter(
        endpoint=endpoint, model=MODEL_ID, key_env="", timeout_s=120.0,
        price_input_per_m=None, price_output_per_m=None,
    )
    ledger = modules["budget"].Ledger(
        output_root / "reservation_ledger.jsonl", cap_usd=RESERVATION_LEDGER_CAP_USD
    )
    runner = modules["runner"].Runner(
        adapter, ledger, output_root / "raw", default_reserve_usd=RESERVATION_USD
    )
    started = time.perf_counter()
    exception = None
    try:
        records = runner.run_all(
            tasks, progress_every=10, results_path=output_root / "records.jsonl", delay_s=0.0
        )
    except Exception as exc:
        # Preserve and summarize completed, fsynced records if the official
        # runner raises; do not retry or replace a partially attempted decision.
        exception = exc
        with (output_root / "records.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    elapsed_s = time.perf_counter() - started
    official, summary = make_summary(modules, tasks, tiers, records, ledger.charged, elapsed_s)
    attempted = {record["task_id"] for record in records}
    unattempted = [task.id for task in tasks if task.id not in attempted]
    summary["unattempted_task_ids"] = unattempted
    summary["runner_exception"] = (
        {"type": type(exception).__name__, "message": str(exception)} if exception else None
    )
    _write_json(output_root / "unattempted.json", {
        "task_ids": unattempted,
        "note": "Tasks without completed durable records; an interrupted in-flight task may have a reservation or raw artifact. No tasks are retried by this script.",
    })
    _write_json(output_root / "official_summary.json", official)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({
        key: summary[key]
        for key in ("n_planned", "n_attempted", "n_valid", "n_correct", "accuracy", "coverage", "complete")
    }), flush=True)
    if exception:
        raise exception
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run(args.harness_dir, args.output_dir, args.endpoint)


if __name__ == "__main__":
    main()
