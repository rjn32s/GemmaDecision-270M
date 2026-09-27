"""Budgeted v2 jobs. Only this directory's source files are uploaded."""
from pathlib import Path
import json
import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/experiment/code-v2")
ROOT = HERE.parents[2] if modal.is_local() else Path("/experiment")
CPU_RATE = 2*.0000131+16*.00000222
GPU_RATE = .000542+CPU_RATE
app = modal.App("gemmadecision-270m-v2")
volume = modal.Volume.from_name("gemma-270m-contrastive-experiment-20260927",create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.0","transformers==5.17.0","pyarrow==25.0.1","numpy==2.4.6","ijson==3.4.0")
         .env({"PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false","HF_HOME":"/experiment/work/huggingface",
               "HF_HUB_DISABLE_TELEMETRY":"1"}))
for file in sorted(HERE.glob("*.py")):
    image = image.add_local_file(file,"/experiment/code-v2/"+file.name)


def result_files(stage):
    root=Path("/experiment/work/decision-v2")
    output=root/stage
    files={}
    if output.exists():
        for p in output.iterdir():
            if p.is_file() and p.suffix in {".json",".safetensors"} and p.stat().st_size<50000000:
                files[p.name]=p.read_bytes()
    return files


@app.function(image=image,cpu=(2,2),memory=(16384,16384),timeout=1800,startup_timeout=600,retries=0,
              max_containers=1,volumes={"/experiment/work":volume})
def cpu_job(stage):
    import sys,time
    sys.path.insert(0,"/experiment/code-v2")
    from common import save,QWEN_REVISION,CLM_REVISION
    start=time.monotonic()
    try:
        if stage=="prepare":
            from data import prepare
            report=prepare("/experiment/work/decision-v2","/experiment/work/decision-v1","/experiment/work/model")
        elif stage=="assets":
            from huggingface_hub import snapshot_download,hf_hub_download
            snapshot_download("Qwen/Qwen3-8B",revision=QWEN_REVISION,local_dir="/experiment/work/qwen-reference",
                              allow_patterns=["*.safetensors","*.json","*.txt","LICENSE"],max_workers=3)
            hf_hub_download("Contrastive-LM/CLM-v0.1-8B","CLM_v0.1-8B.pt",revision=CLM_REVISION,
                            local_dir="/experiment/work/clm-reference")
            report={"qwen_revision":QWEN_REVISION,"clm_revision":CLM_REVISION}
        else:raise ValueError(stage)
        report["function_seconds"]=time.monotonic()-start
        report["estimated_function_usd"]=report["function_seconds"]*CPU_RATE
        save(f"/experiment/work/decision-v2/{stage}/metrics.json",report)
        return result_files(stage)
    finally:volume.commit()


@app.function(image=image,gpu="L40S",cpu=(2,2),memory=(32768,32768),timeout=3600,startup_timeout=600,
              retries=0,max_containers=1,scaledown_window=2,volumes={"/experiment/work":volume})
def gpu_job(stage, seconds, revision=""):
    import sys,time,hashlib
    sys.path.insert(0,"/experiment/code-v2")
    from common import save
    from run import run
    start=time.monotonic()
    try:
        if stage=="verify_public":
            from verify_public import verify
            report=verify(revision)
        else:
            report=run(stage,seconds)
        report["function_seconds"]=time.monotonic()-start
        report["estimated_function_usd"]=report["function_seconds"]*(GPU_RATE+16*.00000222)
        save(f"/experiment/work/decision-v2/{stage}/metrics.json",report)
        save(f"/experiment/work/decision-v2/{stage}/source-hashes.json",
             {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/experiment/code-v2').glob('*.py')})
        return result_files(stage)
    finally:volume.commit()


@app.local_entrypoint()
def main(stage:str="prepare",revision:str=""):
    import fcntl,time
    from datetime import datetime,timezone
    from modal import Workspace
    configs={"prepare":("data",1800),"assets":("reference",1800),"cache":("training",3600),
             "reference":("reference",3000),"train":("training",3600),"evaluate":("evaluation",3600),
             "package":("release",1800),"benchmark":("evaluation",1200),"verify_public":("release",900)}
    group,timeout=configs[stage]
    guard=(ROOT/"work/decision-training.lock").open("a")
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=ROOT/"work/decision-budget-ledger.json"
    state=json.loads(ledger.read_text())
    budget=state.setdefault("v2",{"baseline_metered_usd":.86,"new_budget_usd":21,
                                   "allocations":{"data":2,"reference":3,"training":10,"evaluation":4,"release":2}})
    rate=CPU_RATE if stage in {"prepare","assets"} else GPU_RATE+16*.00000222
    reservation=(timeout+300)*rate+.10
    previous=[j for j in state["jobs"] if j.get("version")==2]
    def cost(j):return j.get("estimated_function_usd",j["reserved_usd"])+(.10 if "estimated_function_usd" in j else 0)
    used=float(Workspace.from_context().billing.summary().metered_cost)
    estimated=sum(cost(j) for j in previous)
    if sum(cost(j) for j in previous if j["group"]==group)+reservation>budget["allocations"][group]:
        raise RuntimeError("Stage reservation exceeds its allocation; reconcile usage before any change")
    if max(used-budget["baseline_metered_usd"],estimated)+reservation>21:
        raise RuntimeError("New work would exceed the $21 allocation")
    job={"version":2,"stage":stage,"group":group,"reserved_usd":reservation,"status":"reserved",
         "metered_before_usd":used,"started":datetime.now(timezone.utc).isoformat()}
    state["jobs"].append(job);ledger.write_text(json.dumps(state,indent=2)+"\n")
    print(f"V2 {stage}: reservation ${reservation:.3f}; metered so far ${used:.2f}",flush=True)
    try:
        files=cpu_job.remote(stage) if stage in {"prepare","assets"} else gpu_job.with_options(timeout=timeout).remote(stage,timeout-150,revision)
        destination=HERE/"results"/stage;destination.mkdir(parents=True,exist_ok=True)
        for name,content in files.items():
            if Path(name).name!=name:raise ValueError("Invalid artifact name")
            (destination/name).write_bytes(content)
        report=json.loads(files["metrics.json"])
        job.update(status="completed",estimated_function_usd=report["estimated_function_usd"])
        print(json.dumps(report,indent=2),flush=True)
    except Exception:
        job["status"]="failed";raise
    finally:
        ledger.write_text(json.dumps(state,indent=2)+"\n");guard.close()
