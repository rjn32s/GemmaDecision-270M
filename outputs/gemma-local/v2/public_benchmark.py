"""Public-only JevBench run after model freeze; no benchmark-dependent tuning."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.request
import numpy as np
from common import save

REVISION="d06ee95988da1350eb8ae5511daa0e06bfffe911"


def benchmark(deadline):
    root=Path("/experiment/work/decision-v2")
    if not (root/"package/standalone-check.json").exists():raise RuntimeError("Freeze and verify the release before benchmark exposure")
    harness=root/"jevbench-harness";hashes={}
    # adapters/__init__.py imports the other adapter modules. Keep the complete
    # pinned Python package so its normal imports work without changing upstream.
    selected={"datasets/public/easy.jsonl","datasets/public/original.jsonl","datasets/public/hard.jsonl","LICENSE"}
    with urllib.request.urlopen(f"https://codeload.github.com/fstandhartinger/jevbench/tar.gz/{REVISION}",timeout=120) as r:
        archive=r.read()
    with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():continue
            name=member.name.split("/",1)[-1]
            if name not in selected and not (name.startswith("jevbench/") and name.endswith(".py")):continue
            path=harness/name
            if not path.resolve().is_relative_to(harness.resolve()):raise ValueError("Invalid archive path")
            path.parent.mkdir(parents=True,exist_ok=True)
            content=bundle.extractfile(member).read();path.write_bytes(content)
            hashes[name]=hashlib.sha256(content).hexdigest()
    sys.path.insert(0,str(harness))
    from jevbench.tasks import load_jsonl
    from jevbench.scoring import score_task
    from jevbench_adapter import GemmaDecisionAdapter
    adapter=GemmaDecisionAdapter(model=str(root/"release"),device="cuda")
    rows=[]
    for tier in ["easy","original","hard"]:
        tasks=load_jsonl(str(harness/f"datasets/public/{tier}.jsonl"))
        for i,task in enumerate(tasks):
            if time.monotonic()>deadline-30:raise RuntimeError("Public benchmark allocation exhausted")
            response=adapter.run(task)
            scored=score_task(response.probs or {},task) if response.ok else {"valid":False,"correct":False}
            rows.append({"task_id":task.id,"tier":tier,"family":task.family,"group":task.group,
                         "eligible_for_accuracy":task.expected is not None and not task.provenance.get("exclude_reason"),
                         "ok":response.ok,"error":response.error,"latency_s":response.latency_s,
                         "probs":response.probs,"usage":response.usage,"score":scored})
            if (i+1)%25==0:print(f"JevBench public {tier}: {i+1}/{len(tasks)}",flush=True)
    report={}
    for tier in ["easy","original","hard"]:
        selected=[r for r in rows if r["tier"]==tier];eligible=[r for r in selected if r["eligible_for_accuracy"]]
        report[tier]={"attempted":len(selected),"scored":len(eligible),"accuracy":float(np.mean([r["score"]["correct"] for r in eligible])),
                      "coverage":float(np.mean([r["ok"] for r in selected])),
                      "p50_seconds":float(np.median([r["latency_s"] for r in selected])),
                      "p95_seconds":float(np.quantile([r["latency_s"] for r in selected],.95))}
    save(root/"benchmark/predictions.json",rows);save(root/"benchmark/hashes.json",hashes)
    return {"status":"complete","harness_revision":REVISION,"public_only":True,"by_tier":report,
            "official_composite_score":None,"benchmark_used_for_training_selection_or_calibration":False,
            "note":"Public per-tier accuracy only; sealed evaluation and official ranking require the independent maintainers."}
