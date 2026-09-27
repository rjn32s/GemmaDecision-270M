"""Pinned public training data; CPU-only preparation, grouped splits, disk shards."""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import random
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

SEED = 27042
SOURCES = {
    "paq": ("sentence-transformers/paq", "74601d8d731019bc9c627ffc4271cdd640e1e748"),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"),
    "when2call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53"),
    "alfworld": ("neulab/agent-data-collection", "31a76bfb0124d77ae7322eabbb0171bf11ee2c67"),
}
FORMAT = "gemmadecision-state-action-v1"


def canonical(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def fetch(root, source, filename):
    repo, revision = SOURCES[source]
    path = root / "downloads" / source / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        print(f"Downloading public {source}/{filename}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "GemmaDecision-270M-training"})
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
        tmp.replace(path)
    return path


def format_state(context, question):
    return f"State:\n{context}\n\nDecision:\n{question}"


def prepare(root, model_path, qa_cap=250000):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "data-audit.json"
    if audit_path.exists():
        existing = json.loads(audit_path.read_text())
        if existing["qa_candidate_cap"] != qa_cap:
            raise ValueError("Prepared directory belongs to a different data configuration")
        print("Reusing frozen prepared dataset", flush=True)
        return existing
    started = time.monotonic()
    rows, counts = [], Counter()
    rng = random.Random(SEED)

    def add(source, sid, context, question, positive, negative, group_text):
        counts[source + "/considered"] += 1
        if not positive.strip() or (negative is not None and canonical(positive) == canonical(negative)):
            counts[source + "/invalid_pair"] += 1
            return
        rows.append({"source": source, "id": str(sid),
                     "state": format_state(context, question), "positive": positive,
                     "negative": negative, "group_text": canonical(group_text)})

    # Draw rows throughout four shards; this is not a sample of every PAQ shard.
    for shard in [0, 22, 44, 66]:
        path = fetch(root, "paq", f"pair/train-{shard:05d}-of-00088.parquet")
        pf = pq.ParquetFile(path)
        selected = np.asarray(sorted(rng.sample(range(pf.metadata.num_rows),
                                               min(qa_cap // 4, pf.metadata.num_rows))))
        offset = 0
        for batch in pf.iter_batches(batch_size=8192, columns=["query", "answer"]):
            ids = selected[(selected >= offset) & (selected < offset + len(batch))]
            for index, row in zip(ids, batch.take((ids - offset).tolist()).to_pylist()):
                add("paq", f"{shard}:{index}", row["query"],
                    "Choose the passage that answers the question.", row["answer"], None, row["answer"])
            offset += len(batch)
        print(f"PAQ shard {shard}: {len(rows)} candidate pairs", flush=True)

    path = fetch(root, "ultrafeedback", "data/train_prefs-00000-of-00001.parquet")
    candidates = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            counts["ultrafeedback/raw"] += 1
            if row["score_chosen"] - row["score_rejected"] < 1.0:
                counts["ultrafeedback/score_gap_below_1"] += 1
                continue
            chosen, rejected = row["chosen"], row["rejected"]
            if not chosen or not rejected or chosen[:-1] != rejected[:-1]:
                counts["ultrafeedback/history_mismatch"] += 1
                continue
            if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
                continue
            candidates.append(row)
    rng.shuffle(candidates)
    for row in candidates[:40000]:
        add("ultrafeedback", row["prompt_id"], row["prompt"],
            "Choose the more helpful, correct response to the user's request.",
            row["chosen"][-1]["content"], row["rejected"][-1]["content"], row["prompt"])

    path = fetch(root, "when2call", "train/when2call_train_pref.jsonl")
    for index, line in enumerate(path.open()):
        row = json.loads(line)
        conversation = "\n".join(m["role"] + ": " + m["content"] for m in row["messages"])
        context = "Available tools:\n" + "\n".join(row["tools"]) + "\n\nConversation:\n" + conversation
        add("when2call", index, context,
            "Choose the appropriate next response: use an available tool, clarify, answer, or explain a limitation.",
            row["chosen_response"]["content"], row["rejected_response"]["content"], conversation)

    # Only trajectories with an explicit successful terminal observation.
    path = fetch(root, "alfworld", "agenttuning_alfworld/full_std.jsonl")
    fetch(root, "alfworld", "agenttuning_alfworld/LICENSE")
    for line in path.open():
        row = json.loads(line)
        events = row["content"]
        if not any(e.get("class_") == "text_observation" and
                   "task completed successfully" in e.get("content", "").lower() for e in events):
            counts["alfworld/trajectory_without_explicit_success"] += 1
            continue
        observations = [e.get("content", "") for e in events if e.get("class_") == "text_observation"]
        initial = "\n".join(observations[:2])
        history = []
        for index, event in enumerate(events):
            kind = event.get("class_")
            if kind == "api_action":
                action = json.dumps({"function": event["function"], "arguments": event["kwargs"]}, sort_keys=True)
                # Keep the initial goal and the last six events; never future feedback.
                context = initial + "\n\nRecent history:\n" + "\n".join(history[-6:])
                add("alfworld", f"{row['id']}:{index}", context,
                    "Choose the next action to make progress toward the household task.",
                    action, None, initial)
                history.append("Action: " + action)
            elif kind == "text_observation":
                history.append("Observation: " + event.get("content", ""))

    print(f"Grouping {len(rows)} examples and removing duplicates", flush=True)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[max(i, j)] = min(i, j)

    groups, states, pair_seen = {}, {}, set()
    valid = []
    for i, row in enumerate(rows):
        group = digest(row["group_text"])
        state = digest(canonical(row["state"]))
        union(i, groups.setdefault(group, i))
        union(i, states.setdefault(state, i))
        pair = (state, digest(canonical(row["positive"])), digest(canonical(row["negative"] or "")))
        if pair in pair_seen:
            counts[row["source"] + "/exact_duplicate"] += 1
        else:
            valid.append(i)
            pair_seen.add(pair)

    # Indexed near-duplicate candidates. Approximate discovery, exact Jaccard
    # verification; do not claim exhaustive corpus-wide near-duplicate removal.
    buckets = defaultdict(list)
    shingle_cache = {}
    links = 0
    for key, i in groups.items():
        words = re.findall(r"\w+", rows[i]["group_text"])
        if len(words) < 20:
            continue
        shingles = {int.from_bytes(hashlib.blake2b(" ".join(words[k:k+5]).encode(), digest_size=8).digest(), "little")
                    for k in range(len(words)-4)}
        anchors = heapq.nsmallest(8, shingles)
        candidates = set()
        for k in range(0, len(anchors)-1, 2):
            candidates.update(buckets[(anchors[k], anchors[k+1])])
        for j in candidates:
            other = shingle_cache[j]
            if min(len(shingles), len(other)) < .8 * max(len(shingles), len(other)):
                continue
            intersection = len(shingles & other)
            if intersection / len(shingles | other) >= .8:
                union(i, j)
                links += 1
        shingle_cache[i] = shingles
        for k in range(0, len(anchors)-1, 2):
            buckets[(anchors[k], anchors[k+1])].append(i)
    del buckets, shingle_cache

    group_hash = {}
    for i in range(len(rows)):
        g = find(i)
        value = digest(rows[i]["group_text"])
        group_hash[g] = min(group_hash.get(g, value), value)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    texts, text_lookup, text_ids = [], {}, []

    def text_id(value):
        if value not in text_lookup:
            text_lookup[value] = len(texts)
            texts.append(value)
            text_ids.append(None)
        return text_lookup[value]

    packed = []
    for i in valid:
        row = rows[i]
        packed.append({"id": row["source"] + ":" + row["id"], "source": row["source"],
                       "group": group_hash[find(i)], "s": text_id(row["state"]),
                       "p": text_id(row["positive"]),
                       "n": -1 if row["negative"] is None else text_id(row["negative"])})
    del rows
    print(f"Tokenizing {len(texts)} unique texts on CPU", flush=True)
    for start in range(0, len(texts), 512):
        encoded = tokenizer(texts[start:start+512], truncation=False, add_special_tokens=True)["input_ids"]
        text_ids[start:start+len(encoded)] = encoded
        if start % 32768 == 0:
            print(f"Tokenized {start}/{len(texts)} texts", flush=True)

    output_rows = defaultdict(list)
    for row in packed:
        if len(text_ids[row["s"]]) > 1024 or any(len(text_ids[row[k]]) > 512 for k in ["p", "n"] if row[k] >= 0):
            counts[row["source"] + "/overlength_rejected"] += 1
            continue
        bucket = int(row["group"][:12], 16) % 100
        split = "train" if bucket < 90 else "validation" if bucket < 94 else "calibration" if bucket < 96 else "test"
        output_rows[split].append(row)
    used = sorted({r[k] for rs in output_rows.values() for r in rs for k in ["s", "p", "n"] if r[k] >= 0})
    remap = {old: new for new, old in enumerate(used)}
    offsets = np.zeros(len(used) + 1, dtype=np.int64)
    for index, old in enumerate(used):
        offsets[index+1] = offsets[index] + len(text_ids[old])
    ids = np.lib.format.open_memmap(root / "tokens.npy", mode="w+", dtype=np.uint32, shape=(int(offsets[-1]),))
    with (root / "texts.jsonl").open("w") as f:
        for index, old in enumerate(used):
            ids[offsets[index]:offsets[index+1]] = text_ids[old]
            f.write(json.dumps(texts[old], ensure_ascii=False) + "\n")
    ids.flush()
    np.save(root / "offsets.npy", offsets)
    signatures = np.asarray([int(digest(canonical(texts[i]))[:15], 16) for i in used], dtype=np.int64)
    np.save(root / "text-keys.npy", signatures)
    split_counts = {}
    split_groups = {}
    for split, values in output_rows.items():
        rng.shuffle(values)
        split_counts[split] = dict(Counter(r["source"] for r in values))
        split_groups[split] = {r["group"] for r in values}
        with (root / f"{split}.jsonl").open("w") as f:
            for row in values:
                f.write(json.dumps({**row, **{k: remap[row[k]] if row[k] >= 0 else -1 for k in ["s", "p", "n"]}}) + "\n")
    for a in split_groups:
        for b in split_groups:
            if a != b:
                assert not split_groups[a] & split_groups[b]
    audit = {"format": FORMAT, "seed": SEED, "sources": SOURCES, "qa_candidate_cap": qa_cap,
             "qa_shards": [0,22,44,66], "split_counts": split_counts, "filter_counts": dict(counts),
             "unique_texts": len(used), "token_count": int(offsets[-1]), "near_duplicate_links": links,
             "near_duplicate_method": "Bottom-8 hashed word-5-shingles, paired-anchor candidate index, verified Jaccard >= 0.8; approximate discovery",
             "max_state_tokens":1024, "max_action_tokens":512, "truncated_texts":0,
             "official_test_data_used":False, "typed_decisions_used":False, "jevbench_items_used":False,
             "preparation_seconds": time.monotonic()-started}
    save_json(audit_path, audit)
    print(json.dumps(audit, indent=2), flush=True)
    return audit
