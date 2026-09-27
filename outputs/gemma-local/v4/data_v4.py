"""CPU-only, deterministic fresh-prompt data for the final bounded v4 run.

Import has no download/model side effects. Call prepare on the Modal volume.
Gold labels are used for fixed class balancing and structural validation only;
no model is loaded, and no example or gold value is printed.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import heapq
import json
from pathlib import Path
import random
import re
import time

import data_v3 as old
from common import schema_state

SEED = 2704404
PROTOCOL = "gemmadecision-v4-fresh-prompts-1"
SPLITS = ("train", "development", "calibration", "final")
REAL_QUOTAS = {
    "train": {"intent_routing": 6000, "evidence_relation": 12000},
    "development": {"intent_routing": 300, "evidence_relation": 300},
    "calibration": {"intent_routing": 150, "evidence_relation": 150},
    "final": {"intent_routing": 600, "evidence_relation": 600},
}
sha = old.sha
norm = old.norm
digest_file = old.digest_file
read_jsonl = old.read_jsonl
write_json = old.write_json
write_jsonl = old.write_jsonl


def state_key(row):
    return sha(norm(row["state"]))


def render(row):
    return schema_state(row["state"], row.get("question", ""))


def legacy_render(row):
    return (row["state"].strip() + "\n\n" + row["question"].strip()).strip()


def make_row(source, family, identity, group, split, state, question, candidates, target, **extra):
    row = dict(id=f"v4:{source}:{identity}", source=source, family=family,
               group=group, workflow_group=f"{source}:{family}", split=split,
               state=state, question=question, candidates=list(candidates),
               target=int(target), category="decision", metric="categorical",
               evaluation_scope="fresh_prompt_known_workflow")
    row.update(extra)
    order = list(range(len(candidates)))
    random.Random(int(sha(f"{SEED}:order:{row['id']}")[:16], 16)).shuffle(order)
    row["candidates"] = [row["candidates"][i] for i in order]
    row["target"] = order.index(target)
    return row


def prior_inputs(v3_root, legacy_root):
    """Only input/group fields of old holdouts are retained; their labels are not reused."""
    all_groups, protected_groups, all_states, protected_states = set(), set(), set(), set()
    v3_rows, replay, prior_hashes = [], [], {}
    for split in SPLITS:
        path = v3_root / f"{split}.jsonl"
        prior_hashes[str(path)] = digest_file(path)
        for row in read_jsonl(path):
            all_groups.add(row["group"]); all_states.add(state_key(row))
            all_states.add(sha(norm(render(row))))
            all_states.add(sha(norm(legacy_render(row))))
            v3_rows.append({"id": row["id"], "state": row["state"], "group": row["group"], "split": split})
            if split == "train":
                replay.append(row)
            else:
                protected_groups.add(row["group"]); protected_states.add(state_key(row))
                protected_states.add(sha(norm(render(row))))
                protected_states.add(sha(norm(legacy_render(row))))
    state_indexes, protected_indexes = set(), set()
    legacy_files = [legacy_root / f"{name}.jsonl" for name in
                    ("train", "validation", "test", "development", "calibration", "final")]
    if not (legacy_root / "train.jsonl").exists():
        raise RuntimeError("Missing legacy training inputs; cannot certify fresh holdouts")
    for path in legacy_files:
        if not path.exists():
            continue
        prior_hashes[str(path)] = digest_file(path)
        for row in read_jsonl(path):
            all_groups.add(row["group"]); state_indexes.add(row["s"])
            if path.stem != "train":
                protected_groups.add(row["group"]); protected_indexes.add(row["s"])
    text_path = legacy_root / "texts.jsonl"
    prior_hashes[str(text_path)] = digest_file(text_path)
    for index, text in enumerate(read_jsonl(text_path)):
        if index in state_indexes:
            key = sha(norm(text)); all_states.add(key)
            if index in protected_indexes:
                protected_states.add(key)
    return dict(all_groups=all_groups, protected_groups=protected_groups,
                all_states=all_states, protected_states=protected_states,
                v3_rows=v3_rows, replay=replay, hashes=prior_hashes)


def banking_pools(v3_root):
    """Source train is training-only; previously unused official test supplies holdouts."""
    train_path = old.fetch(v3_root, "banking77", "train.csv")
    test_path = old.fetch(v3_root, "banking77", "test.csv")
    train = list(csv.DictReader(train_path.open()))
    labels = sorted({r["category"] for r in train})
    if len(labels) != 77:
        raise RuntimeError("Unexpected BANKING77 source label count")
    pools = defaultdict(list)
    for partition, rows in (("train", train), ("test", list(csv.DictReader(test_path.open())))):
        for index, record in enumerate(rows):
            label, text = record["category"], record["text"]
            group = sha("banking:" + norm(text))
            value = int(sha(f"{SEED}:bank-split:{group}")[:8], 16) % 100
            split = "train" if partition == "train" else "development" if value < 25 else "calibration" if value < 40 else "final"
            words = set(label.replace("_", " ").split())
            alternatives = sorted((x for x in labels if x != label),
                                  key=lambda x: (-len(words & set(x.replace("_", " ").split())), sha(group + x)))[:3]
            candidates = [x.replace("_", " ").replace("?", "").strip() for x in [label] + alternatives]
            pools[(split, "intent_routing")].append(make_row(
                "banking77", "intent_routing", f"{partition}:{index}", group, split, text,
                "Which support topic best matches this customer's request?", candidates, 0,
                workflow_group="banking_intent:" + label, source_partition=partition))
    return pools, {str(p): digest_file(p) for p in (train_path, test_path)}


def nli_pools(v3_root):
    import pyarrow.parquet as pq
    path = old.fetch(v3_root, "multi_nli", "data/train-00000-of-00001.parquet")
    pools = defaultdict(list)
    options = ["The evidence supports the claim.", "The evidence leaves the claim unresolved.",
               "The evidence contradicts the claim."]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["pairID", "premise", "hypothesis", "genre", "label"]):
        for record in batch.to_pylist():
            genre, label = record["genre"], record["label"]
            if genre not in {"government", "slate", "telephone", "travel"} or label not in (0, 1, 2):
                continue
            group = sha("premise:" + norm(record["premise"]))
            split = ("train" if genre in {"telephone", "travel"} else "final" if genre == "government"
                     else "calibration" if int(sha(f"{SEED}:nli:{group}")[:8], 16) % 3 == 0 else "development")
            state = "Evidence:\n" + record["premise"] + "\n\nClaim:\n" + record["hypothesis"]
            identity = str(record["pairID"]) + ":" + sha(norm(state))[:16]
            pools[(split, "evidence_relation")].append(make_row(
                "multi_nli", "evidence_relation", identity, group, split, state,
                "Using only the evidence, determine the relation of the claim to that evidence.", options, label,
                workflow_group="nli_genre:" + genre, source_partition="train"))
    return pools, {str(path): digest_file(path)}


def synthetic_rows():
    """Verified opposite-label pairs; both members always remain in one split/group."""
    specifications = {
        "train": (1000, ("threshold", "required_flag", "veto")),
        "development": (50, ("named_minimum",)),
        "calibration": (30, ("inclusive_range",)),
        "final": (100, ("two_thresholds", "range_exception")),
    }
    for split, (pairs, templates) in specifications.items():
        for number in range(pairs):
            template = templates[number % len(templates)]
            rng = random.Random(int(sha(f"{SEED}:{split}:{number}")[:16], 16))
            lo, width = rng.randint(10, 70), rng.randint(4, 20)
            hi = lo + width
            group = sha(f"v4-rule-pair:{template}:{number}")
            for accepted in (False, True):
                flag, blocked = True, False
                amount = lo + 1 if accepted else lo - 1
                second = hi + 1
                if template == "threshold":
                    policy = f"Approve only if the quantity is at least {lo}."
                    truth = amount >= lo
                elif template == "required_flag":
                    amount = hi; flag = accepted
                    policy = f"Approval requires quantity at least {lo} and verified=true."
                    truth = amount >= lo and flag
                elif template == "veto":
                    amount = hi; blocked = not accepted
                    policy = f"Quantity at least {lo} qualifies, but blocked=true always requires rejection."
                    truth = amount >= lo and not blocked
                elif template == "named_minimum":
                    policy = f"The minimum eligible quantity is {lo}; reject anything below this minimum."
                    truth = amount >= lo
                elif template == "inclusive_range":
                    amount = lo + 1 if accepted else hi + 1
                    policy = f"Only quantities from {lo} through {hi}, including both endpoints, qualify."
                    truth = lo <= amount <= hi
                elif template == "two_thresholds":
                    policy = f"Approve when quantity is at least {lo} AND reserve is strictly greater than {hi}. Both are required."
                    truth = amount >= lo and second > hi
                else:
                    amount = lo + 1; blocked = not accepted
                    policy = f"Quantity must be between {lo} and {hi}, inclusive. Exception: reject blocked requests even when their quantity qualifies."
                    truth = lo <= amount <= hi and not blocked
                if truth != accepted:
                    raise AssertionError("Counterfactual generator oracle failed")
                state = f"Policy:\n{policy}\n\nRequest:\nquantity={amount}; reserve={second}; verified={str(flag).lower()}; blocked={str(blocked).lower()}."
                yield make_row("v4_counterfactual_rules", "rule_compliance", f"{split}:{number}:{int(accepted)}",
                               group, split, state, "Apply the stated policy. What should happen to this request?",
                               ["Approve the request.", "Reject the request."], 0 if accepted else 1,
                               workflow_group="v4_rule_template:" + template,
                               evaluation_scope="heldout_synthetic_template" if split != "train" else "synthetic_training",
                               independent_real_prompt=False, source_partition="generated")


def shingles(text):
    text = norm(text)
    if len(text) > 16000:
        text = text[:8000] + " " + text[-8000:]
    words = re.findall(r"\w+|[^\w\s]", text)
    width = min(5, len(words))
    if width == 0:
        return set()
    values = {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest(), "little")
              for i in range(len(words) - width + 1)}
    return set(heapq.nsmallest(512, values))


class NearIndex:
    """Bounded lexical screen; approximate retrieval is not semantic independence."""
    def __init__(self):
        self.buckets = defaultdict(list); self.items = []
        self.counts = Counter()

    def _keys(self, sketch):
        smallest = heapq.nsmallest(32, sketch)
        return [(i, smallest[2 * i], smallest[2 * i + 1]) for i in range(len(smallest) // 2)]

    def add(self, row):
        sketch = shingles(row["state"])
        index = len(self.items); self.items.append((sketch, row["group"]))
        for key in self._keys(sketch):
            bucket = self.buckets[key]
            if len(bucket) <= 500:
                bucket.append(index)

    def matches(self, row):
        sketch = shingles(row["state"]); candidates = set()
        for key in self._keys(sketch):
            bucket = self.buckets.get(key, [])
            if len(bucket) > 500:
                self.counts["oversized_bucket_queries_skipped"] += 1
                continue
            candidates.update(bucket)
        for index in candidates:
            other, group = self.items[index]
            self.counts["candidate_pairs_checked"] += 1
            if group == row["group"] or (sketch and other and len(sketch & other) / len(sketch | other) >= .85):
                self.counts["near_rows_removed"] += 1
                return True
        return False


def validate(rows, split):
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise RuntimeError("Duplicate row identity")
        ids.add(row["id"])
        if row["split"] != split or not all(isinstance(row[x], str) and row[x] for x in ("id", "state", "group", "workflow_group", "family", "source")):
            raise RuntimeError("Invalid data row schema")
        choices = row["candidates"]
        if not 2 <= len(choices) <= 64 or len(set(map(norm, choices))) != len(choices):
            raise RuntimeError("Invalid candidate set")
        if "target_probs" in row:
            probs = row["target_probs"]
            if len(probs) != len(choices) or abs(sum(probs) - 1) > 1e-6 or min(probs) < 0:
                raise RuntimeError("Invalid target probabilities")
        if "target" in row and not 0 <= row["target"] < len(choices):
            raise RuntimeError("Invalid target index")
        if "target" not in row and "target_probs" not in row:
            raise RuntimeError("Missing supervision")


def prepare(root="/experiment/work/decision-v4/data", v3_root="/experiment/work/decision-v3/data",
            legacy_root="/experiment/work/decision-v2", model_path="/experiment/work/model"):
    from transformers import AutoTokenizer
    started = time.monotonic(); root, v3_root, legacy_root = map(Path, (root, v3_root, legacy_root))
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "audit.json"
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if previous.get("protocol") != PROTOCOL:
            raise RuntimeError("Refusing to replace an unknown frozen v4 dataset")
        if all((root / name).exists() and digest_file(root / name) == value for name, value in previous["output_hashes"].items()):
            return previous
        raise RuntimeError("Frozen v4 partition hash mismatch")
    print("V4 data: collecting prior input/group exclusions; no labels or examples logged", flush=True)
    prior = prior_inputs(v3_root, legacy_root); counts = Counter()
    pools, source_hashes = banking_pools(v3_root)
    nli, nli_hashes = nli_pools(v3_root); source_hashes.update(nli_hashes)
    pools.update(nli)
    print(f"V4 data: parsed {sum(map(len, pools.values()))} public source candidates", flush=True)
    group_splits = defaultdict(set)
    for pool in pools.values():
        for row in pool:
            group_splits[row["group"]].add(row["split"])
    ambiguous_groups = {group for group, splits in group_splits.items() if len(splits) > 1}
    counts["source_cross_split_groups_removed"] = len(ambiguous_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder_context_limit = int(json.loads((Path(model_path) / "config.json").read_text())["max_position_embeddings"])
    @lru_cache(maxsize=8192)
    def token_length(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    def fits(row):
        state = render(row)
        return (len(state) < 60000 and all(len(x) < 24000 for x in row["candidates"])
                and token_length(state) <= 2048 and max(map(token_length, row["candidates"])) <= 768
                and all(token_length(state + "\n\nCandidate action:\n" + candidate) <= encoder_context_limit
                        for candidate in row["candidates"]))
    prior_near = NearIndex()
    for row in prior["v3_rows"]:
        prior_near.add(row)
    holdout_near = NearIndex(); splits = defaultdict(list)
    selected_groups = set(); selected_states = set()
    for split in ("final", "calibration", "development", "train"):
        for family, quota in REAL_QUOTAS[split].items():
            pool = sorted(pools[(split, family)], key=lambda r: sha(f"{SEED}:select:{r['id']}"))
            selected = []; label_counts = Counter()
            label_cap = quota // 3 if family == "evidence_relation" else None
            banned_groups = prior["protected_groups"] if split == "train" else prior["all_groups"]
            banned_states = prior["protected_states"] if split == "train" else prior["all_states"]
            for row in pool:
                group = row["group"]
                if group in ambiguous_groups or group in selected_groups or group in banned_groups or state_key(row) in selected_states or state_key(row) in banned_states or sha(norm(render(row))) in banned_states:
                    counts[f"excluded_prior_or_duplicate/{split}/{family}"] += 1; continue
                # Recover source NLI class from semantic candidate, not the shuffled index.
                label = row["candidates"][row["target"]]
                if label_cap is not None and label_counts[label] >= label_cap:
                    continue
                if split != "train" and prior_near.matches(row):
                    counts[f"near_prior_v3/{split}/{family}"] += 1; continue
                if holdout_near.matches(row):
                    counts[f"near_new_holdout/{split}/{family}"] += 1; continue
                eligible = fits(row); row["within_pilot_limits"] = eligible
                if split == "train" and not eligible:
                    counts[f"overlength_training_removed/{family}"] += 1; continue
                selected.append(row); selected_groups.add(group); selected_states.add(state_key(row)); label_counts[label] += 1
                if split != "train":
                    holdout_near.add(row)
                if len(selected) == quota:
                    break
            if len(selected) < quota:
                raise RuntimeError(f"Insufficient fresh {split}/{family}: {len(selected)} of required {quota}")
            splits[split].extend(selected)
            print(f"V4 data: {split}/{family} selected {len(selected)} prompt groups", flush=True)
    # Old validation/calibration/final states are also lexical exclusions for all
    # training, preventing near copies without ever using their target labels.
    protected_near = NearIndex()
    for row in prior["v3_rows"]:
        if row["split"] != "train":
            protected_near.add(row)
    clean_training = []
    for row in splits["train"]:
        if protected_near.matches(row):
            counts["training_near_old_holdout_removed"] += 1
        else:
            clean_training.append(row)
    splits["train"] = clean_training
    for row in synthetic_rows():
        row["within_pilot_limits"] = fits(row)
        if not row["within_pilot_limits"]:
            raise RuntimeError("Generated rule exceeds published context limits")
        splits[row["split"]].append(row)
    replay_limits = {"response_preference": 400, "retention_when2call": 400, "retention_paq": 300, "agent": 400}
    replay_counts = Counter()
    for original in sorted(prior["replay"], key=lambda r: sha(f"{SEED}:replay:{r['id']}")):
        key = "agent" if original["category"] == "agent" else original["family"]
        if key not in replay_limits or replay_counts[key] >= replay_limits[key]:
            continue
        if original["group"] in prior["protected_groups"] or state_key(original) in prior["protected_states"] or original["group"] in selected_groups or state_key(original) in selected_states:
            continue
        if holdout_near.matches(original) or protected_near.matches(original) or not fits(original):
            continue
        row = dict(original); row["id"] = "v4-replay:" + row["id"]
        row.update(split="train", evaluation_scope="training_replay", within_pilot_limits=True)
        splits["train"].append(row); selected_groups.add(row["group"]); selected_states.add(state_key(row)); replay_counts[key] += 1
    counts.update({"replay/" + key: value for key, value in replay_counts.items()})
    split_groups = {split: {r["group"] for r in rows} for split, rows in splits.items()}
    split_states = {split: {state_key(r) for r in rows} for split, rows in splits.items()}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if split_groups[left] & split_groups[right]:
                raise RuntimeError("Cross-split prompt group leakage")
            if split_states[left] & split_states[right]:
                raise RuntimeError("Cross-split normalized state leakage")
    if any(split_groups[s] & prior["all_groups"] for s in ("development", "calibration", "final")):
        raise RuntimeError("New holdout contains a previously exposed prompt group")
    if split_groups["train"] & prior["protected_groups"]:
        raise RuntimeError("Training contains a protected previous holdout group")
    all_ids = [r["id"] for rows in splits.values() for r in rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate cross-split row identity")
    output_hashes, split_audit, manifest = {}, {}, []
    for split, rows in sorted(splits.items()):
        rows.sort(key=lambda r: sha(f"{SEED}:output:{r['id']}")); validate(rows, split)
        write_jsonl(root / f"{split}.jsonl", rows)
        output_hashes[f"{split}.jsonl"] = digest_file(root / f"{split}.jsonl")
        real_groups = {r["group"] for r in rows if r["family"] in {"intent_routing", "evidence_relation"}}
        split_audit[split] = dict(cases=len(rows), prompt_groups=len(split_groups[split]),
                                 real_primary_prompt_groups=len(real_groups),
                                 families=dict(Counter(r["family"] for r in rows)),
                                 within_2048_768=sum(r["within_pilot_limits"] for r in rows),
                                 workflow_groups=len({r["workflow_group"] for r in rows}))
        manifest.extend({k: r[k] for k in ("id", "group", "workflow_group", "source", "family", "split")} for r in rows)
    if split_audit["final"]["real_primary_prompt_groups"] < 1000:
        raise RuntimeError("Final evaluation has fewer than 1000 fresh real prompt groups")
    write_jsonl(root / "split-manifest.jsonl", manifest)
    output_hashes["split-manifest.jsonl"] = digest_file(root / "split-manifest.jsonl")
    report = dict(protocol=PROTOCOL, status="ready", passed=True, passed_structural=True, seed=SEED,
                  data_code_sha256=digest_file(__file__), data_v3_utility_code_sha256=digest_file(old.__file__), sources={k: old.SOURCES[k] for k in ("banking77", "multi_nli")},
                  synthetic_license="Apache-2.0", replay_license_provenance=str(v3_root / "audit.json"),
                  source_file_sha256=source_hashes, prior_input_file_sha256=prior["hashes"],
                  prior_exposed_group_count=len(prior["all_groups"]), prior_exposed_state_hash_count=len(prior["all_states"]),
                  output_hashes=output_hashes, split_file_sha256=output_hashes, splits=split_audit, counts=dict(counts),
                  real_training_target_counts=REAL_QUOTAS["train"],
                  primary_families=["intent_routing", "evidence_relation"], group_disjoint=True,
                  input_format="schema_state(state, question) + '\\n\\nCandidate action:\\n' + candidate",
                  encoder_context_limit=encoder_context_limit,
                  old_holdout_labels_used_for_training=False, all_prior_groups_excluded_from_new_holdouts=True,
                  prior_holdout_training_conflicts=0, new_holdout_prior_exposure_conflicts=0,
                  final_labels_opened_for_model_selection=False,
                  near_duplicate_screen=dict(method="State-only 5-token shingles; bottom-512 sketches, 16 two-hash bands, Jaccard >= 0.85. Candidate retrieval is approximate.",
                      previous_v3=dict(prior_near.counts), new_holdouts=dict(holdout_near.counts), protected_old_holdouts=dict(protected_near.counts),
                      old_v2_scope="Exact group and normalized state hashes only; no all-v2 lexical screen.",
                      synthetic_scope="Generated pairs remain together; synthetic templates are intentionally correlated and excluded from lexical checks."),
                  limitations=[
                      "Fresh prompt groups, not unseen domains: all BANKING77 intents and selected NLI genres occurred during v3 development/evaluation.",
                      "Public foundation-model pretraining exposure is unknown; there is no guarantee these public corpora were unseen by Gemma.",
                      "BANKING77 four-way labels use deterministic lexical hard alternatives; this is not 77-way classification or arbitrary tool routing.",
                      "MultiNLI is class balanced and filtered to permissive nonfiction genres; source train supplies a newly reserved prompt-group holdout.",
                      "Approximate lexical checks and exact exclusions do not establish semantic independence; oversized buckets are counted when skipped.",
                      "Synthetic challenge examples are correlated counterfactual pairs from held-out templates, not independent real requests.",
                      "Replay training uses only previous training rows; old v3 final results remain exposed regression evidence, never a fresh benchmark.",
                      "Evaluation includes overlength rows and counts rejections in coverage; only training is filtered by context limits."
                  ], seconds=time.monotonic() - started)
    write_json(audit_path, report)
    print(f"V4 data ready: {split_audit['train']['cases']} training cases; {split_audit['final']['real_primary_prompt_groups']} fresh real final groups. No labels/examples logged.", flush=True)
    return report
