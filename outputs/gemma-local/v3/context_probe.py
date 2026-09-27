"""Bounded development-only context diagnostics for a supplied v3 checkpoint.

Inputs are never truncated. Increasing a cap changes request eligibility, not
model inputs. Each eligible case is therefore encoded once and its unchanged
logits are reused for all applicable cap policies. This is a diagnostic, not a
fresh test or a checkpoint-selection routine.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

STATE_CAPS = (2048, 4096, 8192)
CANDIDATE_CAPS = (768, 1536)
SEED = 2704204


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(row):
    return hashlib.sha256((str(SEED) + "|" + row["id"]).encode()).hexdigest()


def _select(rows, max_rows):
    """Length/family/identity-only sampling; no labels influence selection."""
    if not 4 <= max_rows <= 48:
        raise ValueError("The bounded context probe supports 4–48 rows")
    buckets = {"short_control": [], "state_2049_4096": [],
               "state_4097_8192": [], "candidate_769_1536": []}
    beyond = Counter()
    for row in rows:
        state, candidate = row["state_tokens"], row["max_candidate_tokens"]
        if state > 8192: beyond["state_above_8192"] += 1
        if candidate > 1536: beyond["candidate_above_1536"] += 1
        if state <= 2048 and candidate <= 768:
            buckets["short_control"].append(row)
        if 2048 < state <= 4096 and candidate <= 1536:
            buckets["state_2049_4096"].append(row)
        if 4096 < state <= 8192 and candidate <= 1536:
            buckets["state_4097_8192"].append(row)
        if 768 < candidate <= 1536 and state <= 8192:
            buckets["candidate_769_1536"].append(row)
    quota = max(1, max_rows // 4)
    selected = {}; group_counts = Counter(); membership = defaultdict(list)
    # Sample long cases first, then source/family-matched short controls where possible.
    long_families = set()
    for name in ["state_2049_4096", "state_4097_8192", "candidate_769_1536", "short_control"]:
        family_pools = defaultdict(list)
        for row in buckets[name]: family_pools[row["family"]].append(row)
        for pool in family_pools.values(): pool.sort(key=_key)
        families = sorted(family_pools, key=lambda family: (name == "short_control" and family not in long_families, family))
        at = Counter(); accepted = 0
        while accepted < quota:
            changed = False
            for family in families:
                pool = family_pools[family]
                while at[family] < len(pool):
                    row = pool[at[family]]; at[family] += 1
                    if row["id"] in selected:
                        membership[name].append(row["id"]); accepted += 1; changed = True
                        break
                    if group_counts[row["group"]] >= 2: continue
                    if len(selected) >= max_rows: break
                    selected[row["id"]] = row; group_counts[row["group"]] += 1
                    membership[name].append(row["id"]); accepted += 1; changed = True
                    if name != "short_control": long_families.add(family)
                    break
                if accepted >= quota or len(selected) >= max_rows: break
            if not changed or len(selected) >= max_rows: break
    # Interleave length strata so a deadline does not systematically omit long cases.
    order = []
    for index in range(quota):
        for name in ["short_control", "state_2049_4096", "state_4097_8192", "candidate_769_1536"]:
            ids = membership[name]
            if index < len(ids) and ids[index] not in order: order.append(ids[index])
    return [selected[identifier] for identifier in order], {
        "available_by_stratum": {name: len(values) for name, values in buckets.items()},
        "selected_by_stratum": dict(membership), "outside_probe_limits": dict(beyond),
        "sampling": "Label-independent deterministic SHA ordering, round-robin families, at most two rows per prompt group; short controls prefer families represented among long cases.",
        "overlap": "A row can belong to both a long-state and long-candidate stratum; it is encoded once.",
    }


def _latency(records):
    import numpy as np
    values = [row["latency_seconds"] for row in records if row.get("latency_seconds") is not None]
    memory = [row["peak_allocated_gpu_gib"] for row in records if row.get("peak_allocated_gpu_gib") is not None]
    return {"measured_cases": len(values),
            "p50_seconds": float(np.median(values)) if values else None,
            "p95_seconds": float(np.quantile(values, .95)) if values else None,
            "maximum_peak_allocated_gpu_gib": max(memory) if memory else None}


def run(arm, checkpoint, seconds=120, max_rows=48):
    """Run a supplied frozen/lora checkpoint; no final/calibration files are read."""
    if arm not in {"frozen", "lora"}: raise ValueError("Choose frozen or lora explicitly")
    if not 30 <= seconds <= 180: raise ValueError("Context diagnostic allocation is 30–180 seconds")
    import torch
    from common import read_jsonl, save, schema_state
    from model import DecisionModel
    from train import score_predictions
    if not torch.cuda.is_available(): raise RuntimeError("Context model execution is authorized on Modal GPU only")
    start = time.monotonic(); deadline = start + seconds
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    data_root = Path(os.environ.get("V3_DATA_ROOT", str(root / "data")))
    base = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
    checkpoint = Path(checkpoint)
    head = checkpoint / "heads.safetensors"
    adapter = checkpoint / "adapter.safetensors" if arm == "lora" else None
    output = root / "context_probe"; output.mkdir(parents=True, exist_ok=True)
    provenance = {"arm": arm, "checkpoint": str(checkpoint), "head_sha256": _sha(head),
                  "adapter_sha256": _sha(adapter) if adapter else None,
                  "development_sha256": _sha(data_root / "development.jsonl"),
                  "base_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
                  "development_only": True, "final_and_calibration_opened": False}
    torch.set_num_threads(2)
    model = DecisionModel(base, head, arm, adapter).eval()
    rows = list(read_jsonl(data_root / "development.jsonl"))
    for row in rows:
        row["rendered_state"] = schema_state(row["state"], row.get("question", ""))
        row["state_tokens"] = len(model.ids(row["rendered_state"]))
        row["candidate_token_lengths"] = [len(model.ids(value)) for value in row["candidates"]]
        row["max_candidate_tokens"] = max(row["candidate_token_lengths"])
    selected, sampling = _select(rows, max_rows)
    protocol = {**provenance, **sampling, "state_caps": list(STATE_CAPS), "candidate_caps": list(CANDIDATE_CAPS),
                "selected_ids": [row["id"] for row in selected], "truncation": False,
                "max_rows": max_rows, "allocated_seconds": seconds,
                "latency_scope": "One loaded-model untruncated encoding per case, including head scoring and pretokenized tensor construction; excludes initial tokenization, checkpoint loading and network.",
                "logit_reuse": "Identical full-input scores are reused across all cap policies permitting the request; cap changes never alter input text.",
                "selection_use": "Diagnostic only; does not change the selected checkpoint."}
    save(output / "protocol.json", protocol)
    records = []
    for index, row in enumerate(selected):
        record = {key: row.get(key) for key in ["id", "source", "family", "group", "workflow_group", "metric", "evaluation_scope"]}
        record.update(state_tokens=row["state_tokens"], candidate_token_lengths=row["candidate_token_lengths"],
                      logits=None, status="unmeasured_deadline", latency_seconds=None, peak_allocated_gpu_gib=None)
        if time.monotonic() <= deadline - 12:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = time.monotonic()
            try:
                with torch.no_grad(): values = model.logits(row).float().cpu().tolist()
                torch.cuda.synchronize()
                record.update(logits=values, status="success", latency_seconds=time.monotonic()-tick,
                              peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                record.update(status="cuda_out_of_memory", latency_seconds=time.monotonic()-tick,
                              peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(record)
        print(f"Context probe {index+1}/{len(selected)}: state={row['state_tokens']} candidate_max={row['max_candidate_tokens']} {record['status']}", flush=True)
    save(output / "predictions.json", records)
    policies = {}
    for state_cap in STATE_CAPS:
        for candidate_cap in CANDIDATE_CAPS:
            measured_rows = []; logits = []; allowed_records = []; rejections = 0; unknown = 0
            for row, record in zip(selected, records):
                allowed = row["state_tokens"] <= state_cap and row["max_candidate_tokens"] <= candidate_cap
                if not allowed:
                    measured_rows.append(row); logits.append(None); rejections += 1
                elif record["status"] == "unmeasured_deadline": unknown += 1
                else:
                    measured_rows.append(row); logits.append(record["logits"]); allowed_records.append(record)
            scores = score_predictions(measured_rows, logits)[0] if measured_rows else None
            policies[f"state_{state_cap}_candidate_{candidate_cap}"] = {
                "state_cap": state_cap, "candidate_cap": candidate_cap,
                "population": len(selected), "scored_or_length_rejected": len(measured_rows),
                "length_rejections": rejections, "unmeasured_deadline": unknown,
                "population_complete": unknown == 0, "metrics_on_measured_population": scores,
                "latency_and_memory": _latency(allowed_records),
                "comparison_caution": "Use full-population comparisons only when population_complete is true; deadline cases are unmeasured, while length/OOM rejections count as failures."}
    strata = {}
    lookup = {record["id"]: record for record in records}
    selected_lookup = {row["id"]: row for row in selected}
    for name, identifiers in sampling["selected_by_stratum"].items():
        values = [lookup[identifier] for identifier in identifiers]
        successful = sum(row["status"] == "success" for row in values)
        attempted = [value for value in values if value["status"] != "unmeasured_deadline"]
        quality = score_predictions([selected_lookup[value["id"]] for value in attempted],
                                    [value["logits"] for value in attempted])[0] if attempted else None
        strata[name] = {"natural_available": sampling["available_by_stratum"][name],
                        "selected": len(values), "successfully_scored": successful,
                        "unmeasured_deadline": len(values) - len(attempted),
                        "status": "measured" if successful else "unmeasured",
                        "metrics_on_measured_population": quality,
                        **_latency(values)}
    for name in sampling["available_by_stratum"]:
        if name not in strata: strata[name] = {"natural_available": sampling["available_by_stratum"][name], "selected": 0,
                                             "successfully_scored": 0, "status": "unmeasured"}
    report = {"status": "partial" if any(row["status"] == "unmeasured_deadline" for row in records) else "complete",
              "provenance": provenance, "selected_cases": len(selected), "policies": policies, "strata": strata,
              "actual_seconds": time.monotonic() - start,
              "limitations": ["Development diagnostic on a deliberately stratified subset; not a representative benchmark or unseen final test.",
                  "No artificial padding or generated long cases were added. A stratum without natural examples remains unmeasured.",
                  "Quality at 8K is unmeasured unless natural inputs above 4K were successfully scored; a larger declared cap alone provides no such evidence.",
                  "Cap policies reuse identical untruncated predictions; latency reflects measured input lengths, not six separate model runs.",
                  "Small strata and correlated workflows limit statistical conclusions; no JEV or benchmark-rank claim."]}
    save(output / "metrics.json", report)
    return report
