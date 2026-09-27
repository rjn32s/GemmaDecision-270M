"""Read-only, bounded structural and lexical leakage audit for prepared v3 data.

This module never loads a model, downloads a source, prints a dataset example,
returns gold labels, or edits the prepared splits. Final labels are inspected only
for type/shape validity. Call audit('/experiment/work/decision-v3/data').
"""
from collections import Counter, defaultdict
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import re
import time
import unicodedata

SPLITS = ("train", "development", "calibration", "final")
INPUT_FIELDS = ("state", "question", "candidates")
ALLOWED_FIELDS = {
    "id", "source", "family", "group", "workflow_group", "split", "state", "question",
    "candidates", "category", "metric", "evaluation_scope", "target", "target_probs",
    "within_pilot_limits", "source_partition", "independent_real_prompt", "expected_rating",
    "rating_attribute", "candidate_values",
}
MAX_ROWS = 20_000
MAX_STATE_CHARS = 32_000
MAX_SHINGLES = 1_024
MAX_BUCKET_ROWS = 200
MAX_CANDIDATE_PAIRS = 50_000
MAX_REPORTED_MATCHES = 100
SIMILARITY_THRESHOLD = .85


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024): h.update(block)
    return h.hexdigest()


def input_projection(row):
    """Explicitly matches training's input allowlist; never serialize the row."""
    return {key: row[key] for key in INPUT_FIELDS}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_row(row, expected_split):
    """Return error codes, never offending labels or example text."""
    errors = []
    for key in ("id", "source", "family", "group", "workflow_group", "split", "state", "question", "category", "metric"):
        if not isinstance(row.get(key), str): errors.append(f"invalid_{key}_type")
    for key in ("id", "source", "family", "group", "workflow_group", "state"):
        if isinstance(row.get(key), str) and not row[key].strip(): errors.append(f"empty_{key}")
    if row.get("split") != expected_split: errors.append("split_field_mismatch")
    if row.get("category") not in {"decision", "qa", "agent"}: errors.append("invalid_category")
    if row.get("metric") not in {"categorical", "ordinal", "preference_tie"}: errors.append("invalid_metric")
    if set(row) - ALLOWED_FIELDS: errors.append("unreviewed_schema_fields")
    candidates = row.get("candidates")
    valid_candidates = isinstance(candidates, list) and 2 <= len(candidates) <= 64 and all(isinstance(x, str) and x.strip() for x in candidates)
    if not valid_candidates:
        errors.append("invalid_candidates")
        return errors
    normalized = [norm(x) for x in candidates]
    if len(set(normalized)) != len(normalized): errors.append("duplicate_candidates")
    count = len(candidates)
    if "target" not in row and "target_probs" not in row: errors.append("missing_target")
    if "target" in row and (not isinstance(row["target"], int) or isinstance(row["target"], bool) or not 0 <= row["target"] < count): errors.append("invalid_target_index")
    probs = row.get("target_probs")
    if "target_probs" in row:
        if not isinstance(probs, list) or len(probs) != count or not all(_number(x) and x >= 0 for x in probs) or abs(sum(probs) - 1) > 1e-6:
            errors.append("invalid_target_distribution")
        elif row.get("metric") == "categorical" and "target" in row and isinstance(row["target"], int) and 0 <= row["target"] < count:
            if probs[row["target"]] < max(probs) - 1e-9: errors.append("target_distribution_conflict")
    if row.get("metric") == "preference_tie":
        if "target" in row or count != 2 or not isinstance(probs, list) or len(probs) != 2 or any(not _number(p) or abs(p - .5) > 1e-6 for p in probs): errors.append("invalid_preference_tie")
    if row.get("metric") == "ordinal":
        rating = row.get("expected_rating")
        if not _number(rating) or not 0 <= rating <= count - 1: errors.append("invalid_ordinal_rating")
        if row.get("rating_attribute") not in {"helpfulness", "correctness"}: errors.append("invalid_ordinal_attribute")
        if isinstance(probs, list) and len(probs) == count and all(_number(p) for p in probs) and _number(rating):
            if abs(sum(i * p for i, p in enumerate(probs)) - rating) > 1e-6: errors.append("ordinal_distribution_conflict")
    # This validates the explicit renderer contract, not whether arbitrary natural
    # source prose happens to mention words such as "answer" or "confidence".
    if all(key in row for key in INPUT_FIELDS) and any(key in input_projection(row) for key in ("target", "target_probs", "expected_rating", "rating_attribute", "confidence", "preference_strength")):
        errors.append("label_metadata_in_input_projection")
    return errors


def _shingles(text):
    value = norm(text)
    windowed = len(value) > MAX_STATE_CHARS
    if windowed: value = value[:MAX_STATE_CHARS // 2] + " " + value[-MAX_STATE_CHARS // 2:]
    words = re.findall(r"\w+|[^\w\s]", value)
    if len(words) < 5: return set(), windowed, False
    hashes = {
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + 5]).encode(), digest_size=8).digest(), "little")
        for i in range(len(words) - 4)
    }
    sampled = len(hashes) > MAX_SHINGLES
    if sampled: hashes = set(heapq.nsmallest(MAX_SHINGLES, hashes))
    return hashes, windowed, sampled


def audit(datasetroot):
    root = Path(datasetroot); started = time.monotonic()
    counts = Counter(); issues = []; records = []; hashes = {}; rows_seen = 0
    issue_counts = Counter()
    def issue(code, identity, split):
        issue_counts[code] += 1
        if len(issues) < 200: issues.append({"code": code, "id": identity, "split": split})
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if not path.exists():
            issue("missing_split_file", "<file>", split); continue
        hashes[path.name] = file_sha(path)
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if rows_seen >= MAX_ROWS:
                    issue("audit_row_limit_exceeded", "<file>", split); break
                rows_seen += 1
                try: row = json.loads(line)
                except (ValueError, TypeError):
                    issue("invalid_json", f"line:{line_number}", split); continue
                if not isinstance(row, dict):
                    issue("row_not_object", f"line:{line_number}", split); continue
                identity = row.get("id") if isinstance(row.get("id"), str) else f"line:{line_number}"
                errors = validate_row(row, split)
                for error in errors: issue(error, identity, split)
                counts[f"rows/{split}"] += 1
                # Gold is no longer needed after structural checks. Retain only
                # IDs and allowed input text for the leakage analysis.
                if errors: continue
                records.append({"id": identity, "split": split, "group": row["group"], "workflow_group": row["workflow_group"], "source": row["source"], "family": row["family"], "state": row["state"], "question": row["question"], "synthetic": row["source"] == "synthetic_rules"})
    groups = defaultdict(list); ids = defaultdict(list); exact_states = defaultdict(list); exact_inputs = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["group"]].append(i); ids[row["id"]].append(i)
        if row["synthetic"]: counts["synthetic_rows_excluded_from_lexical_audit"] += 1; continue
        exact_states[sha(norm(row["state"]))].append(i)
        exact_inputs[sha(norm(row["state"]) + "\n" + norm(row["question"]))].append(i)
    exact_matches = []
    for key, index in [("cross_split_prompt_group", groups), ("duplicate_row_id", ids), ("cross_split_exact_state", exact_states), ("cross_split_exact_input", exact_inputs)]:
        for indexes in index.values():
            conflict = len(indexes) > 1 if key == "duplicate_row_id" else len({records[i]["split"] for i in indexes}) > 1
            if not conflict: continue
            issue_counts[key] += 1
            if len(exact_matches) < 100:
                exact_matches.append({"type": key, "rows": [{k: records[i][k] for k in ("id", "split", "source")} for i in indexes[:8]]})
    # Bottom-k lexical sketches with 16 two-hash bands. This is a bounded lexical
    # screen, not embedding-based semantic deduplication or a recall guarantee.
    buckets = defaultdict(list); sketches = {}; state_keys = set(); represented = []; report = time.monotonic()
    for i, row in enumerate(records):
        if row["synthetic"]: continue
        key = (row["split"], sha(norm(row["state"])))
        if key in state_keys:
            counts["same_split_exact_states_collapsed_for_lsh"] += 1; continue
        state_keys.add(key)
        shingles, windowed, sampled = _shingles(row["state"])
        counts["states_character_windowed"] += int(windowed); counts["shingle_sets_bottom_k_sampled"] += int(sampled)
        if len(shingles) < 4:
            counts["states_too_short_for_lsh"] += 1; continue
        sketches[i] = shingles; represented.append(i)
        minima = heapq.nsmallest(32, shingles)
        for band in range(len(minima) // 2):
            bucket_key = (band, minima[2 * band], minima[2 * band + 1])
            bucket = buckets[bucket_key]
            if len(bucket) < MAX_BUCKET_ROWS + 1: bucket.append(i)
        if time.monotonic() - report > 20:
            print(f"V3 structural/lexical audit processed {len(represented)} state records; no examples or labels logged", flush=True); report=time.monotonic()
    candidate_pairs = set(); oversized = 0; pair_limit = False
    for bucket in buckets.values():
        if len(bucket) > MAX_BUCKET_ROWS:
            oversized += 1; continue
        splits = defaultdict(list)
        for i in bucket: splits[records[i]["split"]].append(i)
        for a, b in itertools.combinations(sorted(splits), 2):
            for i, j in itertools.product(splits[a], splits[b]):
                candidate_pairs.add((min(i, j), max(i, j)))
                if len(candidate_pairs) >= MAX_CANDIDATE_PAIRS:
                    pair_limit = True; break
            if pair_limit: break
        if pair_limit: break
    matches = []; pair_counts = Counter(); match_count = 0
    for i, j in sorted(candidate_pairs):
        a, b = sketches[i], sketches[j]
        similarity = len(a & b) / len(a | b)
        if similarity < SIMILARITY_THRESHOLD: continue
        match_count += 1
        left, right = records[i], records[j]
        pair_counts["/".join(sorted((left["split"], right["split"])))] += 1
        match = {"left": {k: left[k] for k in ("id", "split", "source", "family")}, "right": {k: right[k] for k in ("id", "split", "source", "family")}, "shingle_jaccard_estimate": round(similarity, 6), "same_workflow_group": left["workflow_group"] == right["workflow_group"]}
        matches.append(match)
    matches.sort(key=lambda x: (-x["shingle_jaccard_estimate"], x["left"]["id"], x["right"]["id"]))
    return {
        "audit_version": "v3-structural-lexical-1", "passed_structural": not issue_counts,
        "issue_counts": dict(issue_counts), "issues": issues, "exact_conflicts": exact_matches,
        "counts": dict(counts), "split_file_sha256": hashes, "audit_code_sha256": file_sha(Path(__file__)),
        "input_allowlist": list(INPUT_FIELDS), "gold_metadata_excluded_from_retained_audit_records": True,
        "final_gold_inspection": "Type, range, probability normalization and schema consistency only; no examples, label values or predictions returned.",
        "near_duplicate_review_required": match_count > 0,
        "near_duplicate_screen": {
            "method": "Normalized state only; 5-token shingles, bottom-k sketches, 16 two-hash LSH bands; candidates and gold omitted",
            "similarity_threshold": SIMILARITY_THRESHOLD, "eligible_distinct_states": len(represented),
            "candidate_pairs_checked": len(candidate_pairs), "matching_pairs": match_count,
            "matches_by_split_pair": dict(pair_counts), "matches": matches[:MAX_REPORTED_MATCHES],
            "matches_truncated": len(matches) > MAX_REPORTED_MATCHES,
            "oversized_buckets_skipped": oversized, "candidate_pair_limit_reached": pair_limit,
            "max_state_characters": MAX_STATE_CHARS, "max_shingles_per_state": MAX_SHINGLES,
        },
        "limitations": [
            "A lexical match is a review signal, not proof of label leakage or semantic equivalence.",
            "No match does not prove semantic independence: approximate candidate retrieval has no recall guarantee.",
            "Long states use first/last character windows and bottom-k shingle sampling; similarity is approximate in those cases.",
            "Synthetic template correlations are intentional and excluded from the real-state lexical screen; uncertainty must cluster by workflow_group.",
            "The input allowlist excludes metadata fields; this does not prove that arbitrary source prose never mentions an answer.",
            "This audit validates private final-data structure and lexical partition boundaries, not model performance or model selection.",
        ],
        "seconds": time.monotonic() - started,
    }
