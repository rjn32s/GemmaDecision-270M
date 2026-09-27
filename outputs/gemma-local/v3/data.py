"""Bounded CPU-only v3 preparation. No source test or benchmark is used.

Only prepare() downloads data; import and synthetic generation have no side effects.
The final file contains labels for a later frozen evaluation; audit/logs omit labels.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
import unicodedata
import urllib.request

SEED = 2704303
PROTOCOL = "gemmadecision-v3-data-1"
REPAIR_PROTOCOL = "gemmadecision-v3-partition-repair-1"
SOURCES = {
    "banking77": {
        "repo": "PolyAI/banking77", "revision": "90d4e2ee5521c04fc1488f065b8b083658768c57",
        "upstream_revision": "57ec275d8078af65b7731c2a98be812d844a6d6b", "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/datasets/PolyAI/banking77",
    },
    "multi_nli": {
        "repo": "nyu-mll/multi_nli", "revision": "da70db2af9d09693783c3320c4249840212ee221",
        "license": "OANC permissive terms; fiction excluded",
        "license_url": "https://cims.nyu.edu/~sbowman/multinli/paper.pdf",
        "excluded_genres": ["fiction"], "included_genres": ["government", "slate", "telephone", "travel"],
    },
    "helpsteer2": {
        "repo": "nvidia/HelpSteer2", "revision": "990b2711a36180dd19d9c94b8627844866f8982a",
        "license": "CC-BY-4.0", "license_url": "https://huggingface.co/datasets/nvidia/HelpSteer2",
    },
    "synthetic_rules": {"revision": PROTOCOL, "license": "Apache-2.0", "generator": "data.py"},
}
FILE_HASHES = {
    ("multi_nli", "data/train-00000-of-00001.parquet"): "1c1de03640b168e410aabfca19e7cc2f3dfcd7f0e126e935674e56fb102c4529",
    ("helpsteer2", "train.jsonl.gz"): "c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713",
    ("helpsteer2", "preference/preference.jsonl.gz"): "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b",
}
FAMILIES = ("intent_routing", "evidence_relation", "response_preference", "ordinal_quality", "rule_compliance", "priority_selection")
TRAIN_QUOTAS = dict(zip(FAMILIES, (800, 800, 800, 400, 600, 600)))
FINAL_QUOTAS = dict(zip(FAMILIES, (400, 400, 400, 200, 120, 120)))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024): h.update(block)
    return h.hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n"); tmp.replace(path)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path); tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def split_prompt(group):
    value = int(sha(f"{SEED}:{group}")[:12], 16) % 100
    return "train" if value < 65 else "development" if value < 80 else "calibration" if value < 90 else "final"


def fetch(root, source, filename):
    path = Path(root) / "downloads" / source / filename
    expected = FILE_HASHES.get((source, filename))
    if path.exists() and (expected is None or digest_file(path) == expected): return path
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = SOURCES[source]
    if source == "banking77" and filename.endswith(".csv"):
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{spec['upstream_revision']}/banking_data/{filename}"
    else:
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{filename}"
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "GemmaDecision-v3-research"}
        if offset: headers["Range"] = f"bytes={offset}-"
        try:
            print(f"Download {source}/{filename}, attempt {attempt+1}, resume bytes {offset}", flush=True)
            req = urllib.request.Request(url + ("?download=true" if attempt and "huggingface.co" in url else ""), headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                resumed = response.status == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Invalid download range")
                mode = "ab" if resumed else "wb"
                expected_bytes = int(response.headers.get("Content-Length", 0))
                received = 0; report = time.monotonic()
                with part.open(mode) as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block); received += len(block)
                        if time.monotonic() - report >= 15:
                            print(f"Download {source}: {(received + (offset if resumed else 0))/1e6:.1f} MB", flush=True); report=time.monotonic()
                if expected_bytes and received != expected_bytes: raise RuntimeError("Incomplete download")
            if expected and digest_file(part) != expected:
                part.unlink(); raise RuntimeError("Source hash mismatch")
            part.replace(path); return path
        except Exception as exc:
            if getattr(exc, "code", None) == 416 and part.exists(): part.unlink()
            print(f"Download retry {source}: {type(exc).__name__}", flush=True)
            if attempt == 3: raise RuntimeError(f"Download failed: {source}/{filename}") from None
            time.sleep(2 * (attempt + 1))


def make_row(source, family, identity, group, split, state, question, candidates, target=None, **extra):
    row = dict(id=f"{source}:{identity}", source=source, family=family, group=group,
               workflow_group=f"{source}:{family}", split=split, state=state, question=question,
               candidates=list(candidates), category="decision", metric="categorical", evaluation_scope="fresh_prompt")
    if target is not None: row["target"] = int(target)
    row.update(extra)
    # Candidate permutation is fixed by id and independent of gold. Ordinal levels
    # retain their natural order so expected-rating metrics remain meaningful.
    if row["metric"] != "ordinal":
        order=list(range(len(candidates))); random.Random(int(sha(str(SEED)+row["id"])[:16],16)).shuffle(order)
        row["candidates"]=[row["candidates"][i] for i in order]
        if "target" in row: row["target"]=order.index(row["target"])
        if "target_probs" in row: row["target_probs"]=[row["target_probs"][i] for i in order]
    return row


def banking_rows(root):
    records=list(csv.DictReader(fetch(root,"banking77","train.csv").open()))
    labels=sorted({r["category"] for r in records},key=lambda x:sha(f"{SEED}:intent:{x}"))
    intent_split={x:("final" if i<12 else "development" if i<22 else "calibration" if i<30 else "train") for i,x in enumerate(labels)}
    by_split={s:[x for x in labels if intent_split[x]==s] for s in set(intent_split.values())}
    for i,r in enumerate(records):
        label=r["category"]; split=intent_split[label]; text=r["text"]; group=sha("banking:"+norm(text))
        # Similar label words make alternatives plausible; all candidates belong
        # to this split's intents, so held-out intent names never enter training.
        tokens=set(label.casefold().replace("_"," ").split())
        alternatives=sorted((x for x in by_split[split] if x!=label),key=lambda x:(-len(tokens & set(x.casefold().replace("_"," ").split())),sha(group+x)))[:3]
        names=[label]+alternatives
        yield make_row("banking77","intent_routing",str(i),group,split,text,"Which support topic best matches this customer's request?",[x.replace("_"," ").replace("?","").strip() for x in names],0,
                       workflow_group="banking_intent:"+label,evaluation_scope="heldout_workflow" if split!="train" else "fresh_prompt",source_partition="train")


def nli_rows(root):
    import pyarrow.parquet as pq
    path=fetch(root,"multi_nli","data/train-00000-of-00001.parquet")
    options=["The evidence supports the claim.","The evidence leaves the claim unresolved.","The evidence contradicts the claim."]
    pf=pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4096,columns=["pairID","premise","hypothesis","genre","label"]):
        for r in batch.to_pylist():
            if r["genre"] not in {"government","slate","telephone","travel"} or r["label"] not in (0,1,2):continue
            group=sha("premise:"+norm(r["premise"]))
            split="final" if r["genre"]=="government" else "development" if r["genre"]=="slate" else "calibration" if int(group[:8],16)%10==0 else "train"
            yield make_row("multi_nli","evidence_relation",r["pairID"],group,split,"Evidence:\n"+r["premise"]+"\n\nClaim:\n"+r["hypothesis"],"Using only the evidence, determine the relation of the claim to that evidence.",options,r["label"],workflow_group="nli_genre:"+r["genre"],evaluation_scope="heldout_workflow" if r["genre"] in {"government","slate"} else "fresh_prompt",source_partition="train")


RATING_LEVELS={
    "helpfulness":["0 — Unhelpful: does not address the request.","1 — Mostly unhelpful: little useful assistance.","2 — Partially helpful: useful but with substantial gaps.","3 — Mostly helpful: addresses most of the request.","4 — Fully helpful: effectively addresses the request."],
    "correctness":["0 — Incorrect: major factual or reasoning failures.","1 — Mostly incorrect: few correct elements.","2 — Partly correct: a mixture of correct and incorrect material.","3 — Mostly correct: only minor errors or omissions.","4 — Correct: pertinent facts and reasoning without material errors."],
}


def helpsteer_rows(root):
    for i,r in enumerate(read_jsonl(fetch(root,"helpsteer2","preference/preference.jsonl.gz"))):
        if r["split"]!="train":continue
        group=sha("helpsteer:"+norm(r["prompt"])); split=split_prompt(group)
        strength=float(r["preference_strength"])
        target=0 if strength<0 else 1 if strength>0 else None
        extra=dict(source_partition="train")
        if target is None:extra.update(target_probs=[.5,.5],metric="preference_tie")
        yield make_row("helpsteer2_preference","response_preference",str(i),group,split,r["prompt"],"Which response better satisfies this request? Consider usefulness, correctness and instruction following.",[r["response_1"],r["response_2"]],target,**extra)
    for i,r in enumerate(read_jsonl(fetch(root,"helpsteer2","train.jsonl.gz"))):
        group=sha("helpsteer:"+norm(r["prompt"])); attribute="helpfulness" if int(group[-2:],16)%2==0 else "correctness"
        rating=float(r[attribute]); assert 0<=rating<=4
        probs=[0.]*5; low=int(math.floor(rating)); high=int(math.ceil(rating))
        if low==high:probs[low]=1.
        else:probs[low]=high-rating;probs[high]=rating-low
        yield make_row("helpsteer2_rating","ordinal_quality",str(i),group,split_prompt(group),"Request:\n"+r["prompt"]+"\n\nResponse:\n"+r["response"],f"Rate the response's {attribute} using the stated 0 to 4 scale.",RATING_LEVELS[attribute],int(round(rating)),target_probs=probs,metric="ordinal",expected_rating=rating,rating_attribute=attribute,source_partition="train")


def synthetic_rows():
    """Rule/template splits are fixed first; instances are not independent workflows."""
    templates={
        "train": ["all_thresholds","either_threshold","veto_threshold","priority_then_deadline","cheapest_eligible","deadline_then_priority"],
        "development": ["one_only_threshold","highest_priority_eligible"],
        "calibration": ["bounded_range","earliest_eligible"],
        "final": ["two_of_three_with_veto","priority_then_cost_eligible"],
    }
    compliance={"all_thresholds","either_threshold","veto_threshold","one_only_threshold","bounded_range","two_of_three_with_veto"}
    for split,names in templates.items():
        for template in names:
            for i in range(700 if split=="train" else 240):
                rng=random.Random(int(sha(f"{SEED}:{template}:{i}")[:16],16))
                if template in compliance:
                    a,b,c=[rng.randint(0,100) for _ in range(3)]; x,y,z=[rng.randint(20,80) for _ in range(3)]; veto=rng.choice([False,True])
                    if template=="all_thresholds": rule=f"Accept only when quantity A is at least {x} AND quantity B is at least {y}."; accept=a>=x and b>=y
                    elif template=="either_threshold":rule=f"Accept when quantity A is at least {x} OR quantity B is at least {y}; one condition is enough.";accept=a>=x or b>=y
                    elif template=="veto_threshold":rule=f"Accept when quantity A is at least {x}, unless the blocked flag is true. A blocked request must be rejected.";accept=a>=x and not veto
                    elif template=="one_only_threshold":rule=f"Accept if exactly one of these conditions is true: quantity A is at least {x}; quantity B is at least {y}. Reject when both or neither are true.";accept=(a>=x)!=(b>=y)
                    elif template=="bounded_range":lo,hi=sorted([x,y]);rule=f"Accept when quantity A lies between {lo} and {hi}, inclusive, and the blocked flag is false.";accept=lo<=a<=hi and not veto
                    else:rule=f"Accept if at least two conditions hold: A is at least {x}, B is at least {y}, C is at least {z}. However, reject any request with blocked=true regardless of these conditions.";accept=sum([a>=x,b>=y,c>=z])>=2 and not veto
                    state=f"Policy:\n{rule}\n\nRequest:\nA={a}; B={b}; C={c}; blocked={str(veto).lower()}."
                    candidates=["Accept the request.","Reject the request."];target=0 if accept else 1;question="Apply the stated policy. What should happen to this request?";family="rule_compliance"
                else:
                    items=[dict(name=f"Task {chr(65+j)}",priority=rng.randint(1,8),deadline=rng.randint(1,24),cost=rng.randint(1,90),eligible=rng.choice([True,False])) for j in range(4)]
                    if not any(x["eligible"] for x in items):items[rng.randrange(4)]["eligible"]=True
                    eligibility=template not in {"priority_then_deadline","deadline_then_priority"}
                    pool=[j for j in range(4) if items[j]["eligible"] or not eligibility]
                    if template in {"priority_then_deadline","highest_priority_eligible"}: keys=lambda j:(-items[j]["priority"],items[j]["deadline"],j);policy="Choose the highest priority; break ties by earliest deadline, then alphabetical task name."
                    elif template in {"deadline_then_priority","earliest_eligible"}:keys=lambda j:(items[j]["deadline"],-items[j]["priority"],j);policy="Choose the earliest deadline; break ties by highest priority, then alphabetical task name."
                    elif template=="cheapest_eligible":keys=lambda j:(items[j]["cost"],-items[j]["priority"],j);policy="Choose the lowest cost; break ties by highest priority, then alphabetical task name."
                    else:keys=lambda j:(-items[j]["priority"],items[j]["cost"],j);policy="Choose the highest priority; break ties by lowest cost, then alphabetical task name."
                    if eligibility:policy="Exclude every task marked eligible=false. "+policy
                    else:policy="Consider every task, regardless of its eligible flag. "+policy
                    state="Scheduling policy:\n"+policy+"\n\nTasks:\n"+"\n".join(f"{x['name']}: priority={x['priority']}, deadline={x['deadline']}, cost={x['cost']}, eligible={str(x['eligible']).lower()}" for x in items)
                    candidates=["Select "+x["name"]+"." for x in items];target=min(pool,key=keys);question="Which single task should be selected next?";family="priority_selection"
                yield make_row("synthetic_rules",family,f"{template}:{i}",sha(f"rule_instance:{template}:{i}"),split,state,question,candidates,target,
                               workflow_group="rule_template:"+template,evaluation_scope="heldout_workflow" if split!="train" else "fresh_prompt",independent_real_prompt=False)


def select_rows(rows, quota, max_per_group=1):
    result=[]; groups=Counter()
    for r in sorted(rows,key=lambda x:sha(f"{SEED}:select:{x['id']}")):
        if groups[r["group"]]>=max_per_group:continue
        result.append(r);groups[r["group"]]+=1
        if len(result)>=quota:break
    return result


def replay_rows(legacy_root, split, limits):
    legacy_root=Path(legacy_root)
    texts=[json.loads(x) for x in (legacy_root/"texts.jsonl").open()]
    rows=list(read_jsonl(legacy_root/f"{split}.jsonl"))
    sources=defaultdict(list)
    for r in rows:sources[r["source"]].append(r)
    output=[]
    for category,limit in limits.items():
        allowed=[s for s in sources if (s=="paq" if category=="qa" else s.startswith("agent_") if category=="agent" else s in {"arc","commonsense","when2call"})]
        for si,source in enumerate(sorted(allowed)):
            count=limit//len(allowed)+(si<limit%len(allowed)); pool=sources[source]
            # Deterministic, split-local alternatives; never borrow test actions.
            action_pool=sorted({r["p"] for r in pool}); selected=select_rows(pool,count,max_per_group=4 if category=="agent" else 1)
            known=defaultdict(set)
            for r in pool:known[r["s"]].add(r["p"])
            for r in selected:
                options=r.get("options") or [r["p"]]
                if len(options)==1:
                    rng=random.Random(int(sha(str(SEED)+r["id"])[:16],16));options=[r["p"]]
                    for idx in rng.sample(action_pool,min(64,len(action_pool))):
                        if idx not in known[r["s"]] and norm(texts[idx]) not in {norm(texts[x]) for x in options}:options.append(idx)
                        if len(options)==4:break
                if len(options)<2:continue
                source_split="train" if split=="train" else "development"
                output.append(make_row("v2_"+source,"retention_"+source,r["id"],r["group"],source_split,texts[r["s"]],"",[texts[i] for i in options],options.index(r["p"]),category=category,evaluation_scope="retention_exposed",workflow_group="v2_source:"+source,source_partition=split))
    return output


def repair(root, audit_report=None):
    """Repair only training data using a verified structural/lexical audit.

    No resampling or model score is involved. Evaluation split files are never
    written; final and calibration bytes must remain identical. Sources are not
    downloaded. A small loss of pool size is intentional; the sampler continues
    its fixed 50/25/25 category proportions.
    """
    root = Path(root)
    record_path = root / "partition-repair.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("protocol") != REPAIR_PROTOCOL:
            raise RuntimeError("Unknown previous partition repair")
        if all((root/name).exists() and digest_file(root/name)==value for name,value in previous["post_hashes"].items()):
            return previous
        raise RuntimeError("Repaired data hash mismatch; refusing to overwrite frozen partitions")
    if audit_report is None:
        from audit import audit
        audit_report = audit(root)
    elif isinstance(audit_report, (str, Path)):
        audit_report = json.loads(Path(audit_report).read_text())
    for name, expected in audit_report["split_file_sha256"].items():
        if digest_file(root/name) != expected:
            raise RuntimeError("Partition repair audit is stale")
    unsupported = set(audit_report.get("issue_counts", {})) - {"duplicate_row_id"}
    if unsupported:
        raise RuntimeError("Partition repair cannot safely resolve structural issue codes: " + ", ".join(sorted(unsupported)))
    screen = audit_report["near_duplicate_screen"]
    if screen.get("matches_truncated") or screen.get("candidate_pair_limit_reached"):
        raise RuntimeError("Near-match report is incomplete; refusing partial repair")
    old_audit = json.loads((root/"audit.json").read_text())
    splits = {name:list(read_jsonl(root/f"{name}.jsonl")) for name in ("train","development","calibration","final")}
    train = splits["train"]
    by_id = defaultdict(list)
    for row in train:by_id[row["id"]].append(row)
    removed_groups = set(); triggering_matches = []
    for match in screen.get("matches", []):
        if match["shingle_jaccard_estimate"] < .85:continue
        left, right = match["left"], match["right"]
        if left["split"] == right["split"]:continue
        sides = [side for side in (left,right) if side["split"] == "train"]
        if len(sides) != 1:
            raise RuntimeError("Near-match affects two immutable evaluation partitions; training-only repair is insufficient")
        identity = sides[0]["id"]
        if identity not in by_id:raise RuntimeError("Audit training ID missing from prepared data")
        removed_groups.update(row["group"] for row in by_id[identity])
        triggering_matches.append(match)
    directly_removed_groups = set(removed_groups)

    def owned_candidates(row):
        # HelpSteer responses both belong to the prompt. Ordinal legends are
        # shared rubrics. Recorded-action replay owns only its positive action;
        # sampled negatives came from other source groups and are not its data.
        if row["metric"] == "ordinal":return set()
        if row["source"] == "helpsteer2_preference":return {norm(x) for x in row["candidates"]}
        if "target" in row:return {norm(row["candidates"][row["target"]])}
        return set()

    candidate_removals = set(); contamination_passes = 0
    while True:
        removed_owned = set(); retained_owned = set()
        for row in train:
            (removed_owned if row["group"] in removed_groups else retained_owned).update(owned_candidates(row))
        # Generic actions/labels that remain a positive of an independent kept
        # group are valid source data, so do not erase every shared API or label.
        forbidden = removed_owned - retained_owned
        newly_contaminated = {
            row["group"] for row in train if row["group"] not in removed_groups
            and any(norm(candidate) in forbidden for candidate in row["candidates"])
        }
        if not newly_contaminated:break
        removed_groups.update(newly_contaminated);candidate_removals.update(newly_contaminated);contamination_passes += 1
        if sum(row["group"] in removed_groups for row in train) > 500:
            raise RuntimeError("Conservative negative-provenance repair would remove over 5% of the pilot; review required")
    kept = [row for row in train if row["group"] not in removed_groups]
    all_ids = Counter(row["id"] for rows in splits.values() for row in rows)
    renamed = []
    for row in kept:
        if all_ids[row["id"]] > 1:
            old_id = row["id"]
            content = json.dumps({key:row[key] for key in ("state","question","candidates")},sort_keys=True,ensure_ascii=False)
            row["id"] = old_id + ":input-" + sha(content)[:16]
            renamed.append(dict(old_id=old_id,new_id=row["id"],group=row["group"]))
    splits["train"] = kept
    all_repaired_ids = [row["id"] for rows in splits.values() for row in rows]
    if len(all_repaired_ids) != len(set(all_repaired_ids)):
        raise RuntimeError("Content-qualified IDs remain duplicated; refusing an ambiguous repair")
    mixtures = Counter(row["category"] for row in kept)
    if any(mixtures[k] < 100 for k in ("decision","qa","agent")):
        raise RuntimeError("Insufficient retained examples for a pilot category")
    pre_hashes = {f"{split}.jsonl":digest_file(root/f"{split}.jsonl") for split in splits}
    pre_hashes["split-manifest.jsonl"] = digest_file(root/"split-manifest.jsonl")
    write_json(root/"partition-repair-before-audit.json",old_audit)
    write_jsonl(root/"train.jsonl",kept)
    manifest = [dict(id=row["id"],group=row["group"],workflow_group=row["workflow_group"],source=row["source"],family=row["family"],split=split) for split,rows in sorted(splits.items()) for row in rows]
    write_jsonl(root/"split-manifest.jsonl",manifest)
    post_hashes = {name:digest_file(root/name) for name in pre_hashes}
    if any(post_hashes[f"{s}.jsonl"] != pre_hashes[f"{s}.jsonl"] for s in ("development","calibration","final")):
        raise RuntimeError("Evaluation partition changed unexpectedly")
    record = dict(protocol=REPAIR_PROTOCOL,status="complete",code_sha256=digest_file(__file__),
                  pre_hashes=pre_hashes,post_hashes=post_hashes,
                  audit_sha256=sha(json.dumps(audit_report,sort_keys=True)),
                  direct_training_groups_removed=sorted(directly_removed_groups),
                  candidate_contamination_groups_removed=sorted(candidate_removals),
                  removed_training_ids=[row["id"] for row in train if row["group"] in removed_groups],
                  removed_rows=len(train)-len(kept),renamed_ids=renamed,
                  triggering_matches=triggering_matches,negative_contamination_passes=contamination_passes,
                  remaining_orphaned_removed_candidate_hashes=sorted(sha(x) for x in forbidden),
                  training_pool_counts=dict(mixtures),sampling_mixture=dict(decision=.5,qa=.25,agent=.25),
                  replenished=False,immutable_evaluation_bytes_preserved=True,model_scores_used=False,
                  limitations=["Shared generic candidate text remains when it is independently owned by a retained positive group.","The bounded lexical audit does not establish semantic independence."])
    old_audit["pre_repair_data_code_sha256"] = old_audit["data_code_sha256"]
    old_audit["data_code_sha256"] = digest_file(__file__)
    old_audit["output_hashes"].update(post_hashes)
    old_audit["train_mixture"] = dict(mixtures)
    old_audit["sampling_mixture"] = dict(decision=.5,qa=.25,agent=.25)
    old_audit["repair"] = {key:record[key] for key in ("protocol","removed_rows","training_pool_counts","replenished","immutable_evaluation_bytes_preserved","model_scores_used")}
    rows = kept
    old_audit["splits"]["train"] = dict(cases=len(rows),prompt_groups=len({r["group"] for r in rows}),real_prompt_groups=len({r["group"] for r in rows if r["source"]!="synthetic_rules"}),workflow_groups=len({r["workflow_group"] for r in rows}),families=dict(Counter(r["family"] for r in rows)),within_2048_768=sum(r["within_pilot_limits"] for r in rows),ties=sum(r["metric"]=="preference_tie" for r in rows),ordinal_cases=sum(r["metric"]=="ordinal" for r in rows))
    old_audit["limitations"].append("Pre-training structural/lexical repair removed training prompt groups without replenishment; fixed 50/25/25 sampling remains, while pool counts differ slightly.")
    write_json(root/"audit.json",old_audit)
    write_json(record_path,record)
    print(f"Partition repair complete: removed {record['removed_rows']} training rows; all evaluation file bytes preserved. No example text or labels logged.",flush=True)
    return record


def prepare(root, legacy_root, model_path):
    from transformers import AutoTokenizer
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    audit_path=root/"audit.json"
    code_hash=digest_file(__file__)
    if audit_path.exists():
        old=json.loads(audit_path.read_text())
        if old.get("protocol")==PROTOCOL and old.get("repair",{}).get("protocol")==REPAIR_PROTOCOL:
            if all((root/name).exists() and digest_file(root/name)==value for name,value in old["output_hashes"].items()):return old
            raise RuntimeError("Repaired data hash mismatch; refusing to regenerate immutable evaluation partitions")
        if old.get("protocol")==PROTOCOL and old.get("data_code_sha256")==code_hash:
            if all((root/name).exists() and digest_file(root/name)==value for name,value in old["output_hashes"].items()):return old
    started=time.monotonic();counts=Counter();pools=defaultdict(list)
    for source,generator in [("banking77",banking_rows),("multi_nli",nli_rows),("helpsteer2",helpsteer_rows)]:
        print(f"Preparing source {source}",flush=True)
        # Retain bounded per-family pools by selecting before tokenization. NLI
        # streams row batches but source text fits comfortably on the CPU worker.
        for row in generator(root):pools[(row["split"],row["family"])].append(row)
        print(f"Source {source} parsed; elapsed {time.monotonic()-started:.0f}s",flush=True)
    for row in synthetic_rows():pools[(row["split"],row["family"])].append(row)
    # Remove any underlying prompt/premise assigned to multiple splits, regardless
    # of which examples will eventually be sampled. No final label is inspected.
    group_splits=defaultdict(set);state_splits=defaultdict(set)
    for pool in pools.values():
        for r in pool:
            group_splits[r["group"]].add(r["split"])
            state_splits[sha(norm(r["state"])+"\n"+norm(r["question"]))].add(r["split"])
    bad_groups={g for g,s in group_splits.items() if len(s)>1};bad_states={g for g,s in state_splits.items() if len(s)>1}
    counts["cross_split_groups_removed"]=len(bad_groups);counts["cross_split_states_removed"]=len(bad_states)
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    @lru_cache(maxsize=8192)
    def length(value):return len(tok(value,add_special_tokens=True)["input_ids"])
    def eligible(r):
        text=(r["state"].strip()+"\n\n"+r["question"].strip()).strip()
        return len(text)<60000 and all(len(x)<24000 for x in r["candidates"]) and length(text)<=2048 and max(map(length,r["candidates"]))<=768
    splits=defaultdict(list)
    for (split,family),pool in sorted(pools.items()):
        quota=TRAIN_QUOTAS[family] if split=="train" else FINAL_QUOTAS[family] if split=="final" else 120 if split=="development" else 80
        seen=set();filtered=[]
        for r in pool:
            if r["group"] in bad_groups or sha(norm(r["state"])+"\n"+norm(r["question"])) in bad_states:continue
            key=sha(norm(r["state"])+"\n"+norm(r["question"])+json.dumps(sorted(map(norm,r["candidates"]))))
            if key in seen:counts["exact_duplicate_rows_removed"]+=1;continue
            seen.add(key);filtered.append(r)
        # Overlength training is excluded; final/eval is sampled before eligibility
        # and keeps rejected cases so request coverage cannot be inflated.
        ordered=select_rows(filtered,len(filtered))
        picked=[]
        for r in ordered:
            fits=eligible(r);r["within_pilot_limits"]=fits
            if split=="train" and not fits:counts[f"train_overlength/{family}"]+=1;continue
            picked.append(r)
            if len(picked)>=quota:break
        if len(picked)<quota:raise RuntimeError(f"Insufficient {split}/{family}: {len(picked)} < {quota}")
        splits[split].extend(picked)
        print(f"Prepared {split}/{family}: {len(picked)} cases",flush=True)
    del pools
    for split,limits in [("train",dict(decision=1000,qa=2500,agent=2500)),("validation",dict(decision=120,qa=40,agent=120))]:
        rows=replay_rows(legacy_root,split,limits)
        for r in rows:
            r["within_pilot_limits"]=eligible(r)
            if split=="train" and not r["within_pilot_limits"]:raise RuntimeError("Previously eligible replay row exceeds v3 raw-text token limits")
        # Replays come only from matching legacy partitions, and remain clearly
        # separate from fresh-family development and future final evaluation.
        splits["train" if split=="train" else "development"].extend(rows)
        for cat,want in limits.items():
            actual=sum(r["category"]==cat for r in rows)
            if actual!=want:raise RuntimeError(f"Replay {split}/{cat} has {actual}, expected {want}")
    # Freeze global exact input / group disjointness, with group tied across all
    # HelpSteer preference and ordinal variants and every recorded agent task.
    groups={s:{r["group"] for r in rows} for s,rows in splits.items()}
    for a in groups:
        for b in groups:
            if a<b and groups[a]&groups[b]:raise RuntimeError(f"Cross-split groups remain: {a}/{b}")
    train_counts=Counter(r["category"] for r in splits["train"])
    if train_counts!=Counter(decision=5000,qa=2500,agent=2500):raise RuntimeError(f"Incorrect mixture: {dict(train_counts)}")
    real_final={r["group"] for r in splits["final"] if r["source"]!="synthetic_rules"}
    if len(real_final)<1000:raise RuntimeError(f"Final real prompt groups below threshold: {len(real_final)}")
    output_hashes={};split_audit={};manifest=[]
    for split,rows in sorted(splits.items()):
        rows.sort(key=lambda r:sha(f"{SEED}:output:{r['id']}"))
        write_jsonl(root/f"{split}.jsonl",rows);output_hashes[f"{split}.jsonl"]=digest_file(root/f"{split}.jsonl")
        split_audit[split]={"cases":len(rows),"prompt_groups":len(groups[split]),"real_prompt_groups":len({r["group"] for r in rows if r["source"]!="synthetic_rules"}),"workflow_groups":len({r["workflow_group"] for r in rows}),"families":dict(Counter(r["family"] for r in rows)),"within_2048_768":sum(r["within_pilot_limits"] for r in rows),"ties":sum(r["metric"]=="preference_tie" for r in rows),"ordinal_cases":sum(r["metric"]=="ordinal" for r in rows)}
        manifest.extend(dict(id=r["id"],group=r["group"],workflow_group=r["workflow_group"],source=r["source"],family=r["family"],split=split) for r in rows)
    write_jsonl(root/"split-manifest.jsonl",manifest);output_hashes["split-manifest.jsonl"]=digest_file(root/"split-manifest.jsonl")
    source_files={str(p.relative_to(root/"downloads")):digest_file(p) for p in (root/"downloads").rglob("*") if p.is_file() and not p.name.endswith(".part")}
    audit=dict(protocol=PROTOCOL,seed=SEED,legacy_audit_sha256=digest_file(Path(legacy_root)/"audit.json"),data_code_sha256=code_hash,sources=SOURCES,source_file_sha256=source_files,output_hashes=output_hashes,splits=split_audit,counts=dict(counts),train_mixture=dict(train_counts),final_real_prompt_groups=len(real_final),group_disjoint=True,final_labels_opened_for_model_selection=False,seconds=time.monotonic()-started,
               limitations=["Public pretraining exposure is unknown.","Exact normalized group/input deduplication does not prove semantic independence.","Synthetic instances sharing a rule template are correlated; workflow/template counts must accompany uncertainty.","BANKING77 is one domain and held-out intents are not independent industries.","MultiNLI excludes fiction and uses a government genre holdout; no ANLI is used.","HelpSteer ordinal rubric descriptions are concise task paraphrases, not exact annotator instructions.","Retention development rows were previously exposed during v2 and must not enter fresh-decision macro.","Recorded-action retrieval uses sampled split-local negatives, not live agent completion.","Final labels exist on the private volume but must not be read by pilot/training code."])
    write_json(audit_path,audit)
    print("V3 data preparation complete: 10000 training rows; final label-bearing examples not logged.",flush=True)
    return audit
