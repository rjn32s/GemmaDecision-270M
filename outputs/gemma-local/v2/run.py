"""Cloud-only v2 caching, three-stage training, reference and frozen evaluation."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from common import *

ROOT=Path(os.environ.get("GEMMADECISION_WORK_DIR","/experiment/work/decision-v2"))
BASE=Path(os.environ.get("GEMMA_MODEL_PATH","/experiment/work/model"))


class Data:
    def __init__(self):
        self.tokens=np.load(ROOT/"tokens.npy",mmap_mode="r")
        self.offsets=np.load(ROOT/"offsets.npy",mmap_mode="r")
        self.lengths=np.diff(self.offsets)
        self.keys=np.load(ROOT/"text-keys.npy",mmap_mode="r")
        self.texts=[json.loads(x) for x in (ROOT/"texts.jsonl").open()]
    def rows(self,split):return list(read_jsonl(ROOT/f"{split}.jsonl"))
    def batch(self,indices):
        width=int(max(self.lengths[indices]));ids=np.zeros((len(indices),width),dtype=np.int64);mask=np.zeros_like(ids)
        for j,i in enumerate(indices):
            t=self.tokens[self.offsets[i]:self.offsets[i+1]];ids[j,:len(t)]=t;mask[j,:len(t)]=1
        return {"input_ids":torch.tensor(ids,device="cuda"),"attention_mask":torch.tensor(mask,device="cuda")}


def load_base(path=BASE):
    return AutoModel.from_pretrained(str(path),local_files_only=True,dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)


def pooled(model,batch):
    h=model(**batch,use_cache=False).last_hidden_state
    ix=batch["attention_mask"].sum(-1)-1
    return F.normalize(h[torch.arange(len(ix),device="cuda"),ix].float(),dim=-1)


def make_tasks(data,split,cap=128):
    rng=random.Random(SEED+{"validation":1,"calibration":2,"test":3}[split]);tasks=[]
    sources=defaultdict(list)
    for r in data.rows(split):sources[r["source"]].append(r)
    for source,rs in sorted(sources.items()):
        pool=sorted({r["p"] for r in rs});known=defaultdict(set)
        for r in rs:known[int(data.keys[r["s"]])].add(int(data.keys[r["p"]]))
        rs=sorted(rs,key=lambda r:r["id"]);rng.shuffle(rs);groups=Counter();selected=0
        for r in rs:
            if groups[r["group"]]>=4:continue
            options=list(r["options"])
            if len(options)==1:
                others=[p for p in pool if int(data.keys[p]) not in known[int(data.keys[r["s"]])]]
                # Recorded-next-action retrieval, not environment correctness.
                if len(others)<7:continue
                options.extend(rng.sample(others,7))
            rng.shuffle(options)
            tasks.append({"id":r["id"],"source":source,"group":r["group"],"kind":r["kind"],
                          "s":r["s"],"options":options,"target":options.index(r["p"]),
                          "state":data.texts[r["s"]],"candidates":[data.texts[i] for i in options]})
            groups[r["group"]]+=1;selected+=1
            if selected>=cap:break
    return tasks


def official_tasks(tokenizer):
    from data import fetch
    rows=list(read_jsonl(fetch(ROOT,"when2call","test/when2call_test_mcq.jsonl")))
    random.Random(SEED+4).shuffle(rows);out=[]
    for r in rows[:256]:
        state=schema_state("Available tools:\n"+"\n".join(r["tools"])+"\n\nConversation:\nuser: "+r["question"],
                           "Choose the appropriate next response: use an available tool, clarify, answer, or explain a limitation.")
        keys=list(r["answers"]);candidates=[r["answers"][k] for k in keys]
        lengths=[len(tokenizer(t)["input_ids"]) for t in [state,*candidates]]
        out.append({"id":r["uuid"],"source":"when2call_official","kind":"official","group":r.get("source_id",r["uuid"]),
                    "state":state,"candidates":candidates,"target":keys.index(r["correct_answer"]),
                    "eligible":lengths[0]<=MAX_STATE and max(lengths[1:])<=MAX_ACTION})
    return out


def cache(data,deadline):
    audit=json.loads((ROOT/"audit.json").read_text())
    if not audit["agent_gate_passed"]:raise RuntimeError("Agent data gate has not passed")
    model=load_base();order=np.argsort(data.lengths,kind="stable");count=len(order)
    path=ROOT/"embeddings.npy";progress=ROOT/"embedding-progress.json";offset=0
    if progress.exists():offset=json.loads(progress.read_text())["offset"]
    vectors=np.lib.format.open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.float16,shape=(count,640))
    begin=time.monotonic();last=begin
    with torch.inference_mode():
        while offset<count:
            if time.monotonic()>deadline-60:break
            size=64 if data.lengths[order[offset]]<=512 else 24 if data.lengths[order[offset]]<=1024 else 12
            indices=order[offset:offset+size]
            values=pooled(model,data.batch(indices))
            if not torch.isfinite(values).all():raise RuntimeError("Nonfinite embeddings")
            vectors[indices]=values.cpu().numpy();offset+=len(indices)
            if time.monotonic()-last>25:
                vectors.flush();save(progress,{"offset":offset,"total":count,"format":FORMAT})
                print(f"V2 embeddings {offset}/{count}; {time.monotonic()-begin:.0f}s",flush=True);last=time.monotonic()
    vectors.flush();save(progress,{"offset":offset,"total":count,"format":FORMAT})
    if offset<count:return {"status":"partial","encoded":offset,"total":count}
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    for split in ["validation","calibration","test"]:
        tasks=make_tasks(data,split,cap=128 if split!="calibration" else 64)
        if split=="test":tasks.extend(official_tasks(tokenizer))
        save(ROOT/f"{split}-tasks.json",tasks)
    protocol={"format":FORMAT,"seed":SEED,"state_limit":MAX_STATE,"candidate_limit":MAX_ACTION,
              "selection":"Equal-weight non-PAQ source accuracy on validation; NLL tie-breaker.",
              "test_used_for_selection":False,"agent_metric":"8-way recorded-action retrieval; random alternate demonstrations may also be valid.",
              "official_mcq":"First 256 seeded shuffled official When2Call test cases; overlength cases count as failures.",
              "baseline":"Mean conditional candidate token log likelihood, no candidate BOS/EOS; prefix suffix is Candidate response.",
              "success_target":"Non-QA macro +5pp over base likelihood, paired task-cluster bootstrap lower bound >0, gains on two sources."}
    save(ROOT/"protocol.json",protocol);save(ROOT/"cache/protocol.json",protocol)
    return {"status":"complete","texts":count,"tokens":int(data.offsets[-1]),"cache_seconds":time.monotonic()-begin,
            "tasks":{s:len(json.loads((ROOT/f"{s}-tasks.json").read_text())) for s in ["validation","calibration","test"]}}


def contrastive_loss(heads,rows,vectors,data):
    sids=[r["s"] for r in rows];pids=[r["p"] for r in rows]
    s,a=heads.project(vectors[sids],vectors[pids]);logits=s@a.T*heads.scale()
    sk=torch.tensor([int(data.keys[i]) for i in sids],device=vectors.device)
    pk=torch.tensor([int(data.keys[i]) for i in pids],device=vectors.device)
    positive=(sk[:,None]==sk[None,:])|(pk[:,None]==pk[None,:])
    groups=[r["group"] for r in rows]
    same=torch.tensor([[x==y for y in groups] for x in groups],device=vectors.device)
    masked=logits.masked_fill(same & ~positive,-torch.inf)
    pos=logits.masked_fill(~positive,-torch.inf)
    return ((torch.logsumexp(masked,1)-torch.logsumexp(pos,1)).mean()+
            (torch.logsumexp(masked,0)-torch.logsumexp(pos,0)).mean())/2


def candidate_loss(heads,rows,vectors):
    sids=[r["s"] for r in rows];maximum=max(len(r["options"]) for r in rows)
    aids=[i for r in rows for i in r["options"]]
    s,a=heads.project(vectors[sids],vectors[aids]);logits=[];offset=0
    for i,r in enumerate(rows):
        n=len(r["options"]);v=(s[i:i+1]*a[offset:offset+n]).sum(-1)*heads.scale();offset+=n
        logits.append(F.pad(v,(0,maximum-n),value=-torch.inf))
    return F.cross_entropy(torch.stack(logits),torch.tensor([r["target"] for r in rows],device=vectors.device))


def cached_logits(heads,tasks,vectors):
    out=[]
    with torch.inference_mode():
        for t in tasks:
            s,a=vectors[[t["s"]]],vectors[t["options"]]
            if heads is None:logits=(F.normalize(s,dim=-1)@F.normalize(a,dim=-1).T)[0]/.07
            else:
                s,a=heads.project(s,a);logits=(s@a.T)[0]*heads.scale()
            out.append(logits.cpu().tolist())
    return out


def summarize(tasks,logits,temperature=1.):
    sources=defaultdict(list);outcomes=[]
    for t,l in zip(tasks,logits):
        if l is None:
            hit=0.;nll=None;brier=None
        else:
            a=np.array(l,dtype=np.float64)/temperature;a-=a.max();p=np.exp(a);p/=p.sum()
            best=a==a.max();hit=float(best[t["target"]])/best.sum()
            nll=-float(np.log(max(p[t["target"]],1e-30)))
            gold=np.eye(len(p))[t["target"]];brier=float(((p-gold)**2).sum())
        r={"id":t["id"],"source":t["source"],"group":t["group"],"correct":float(hit),"nll":nll,"brier":brier,"covered":l is not None}
        outcomes.append(r);sources[t["source"]].append(r)
    report={}
    for s,rs in sources.items():
        report[s]={"accuracy":float(np.mean([r["correct"] for r in rs])),"examples":len(rs),
                   "groups":len({r["group"] for r in rs}),"coverage":float(np.mean([r["covered"] for r in rs])),
                   "nll_covered":float(np.mean([r["nll"] for r in rs if r["covered"]])) if any(r["covered"] for r in rs) else None,
                   "brier_covered":float(np.mean([r["brier"] for r in rs if r["covered"]])) if any(r["covered"] for r in rs) else None}
    decision_scores=[v["accuracy"] for s,v in report.items() if s!="paq"]
    return {"by_source":report,"nonqa_macro":float(np.mean(decision_scores)) if decision_scores else None},outcomes


def train(data,deadline):
    if not json.loads((ROOT/"audit.json").read_text())["agent_gate_passed"]:raise RuntimeError("Agent coverage gate failed")
    torch.manual_seed(SEED);rng=random.Random(SEED)
    # 640-d embeddings fit comfortably on the GPU; the frozen encoder is unloaded.
    vectors=torch.tensor(np.load(ROOT/"embeddings.npy"),device="cuda",dtype=torch.float32)
    rows=data.rows("train");by=defaultdict(list)
    for r in rows:by[r["source"]].append(r)
    val=json.loads((ROOT/"validation-tasks.json").read_text())
    heads=make_heads().to("cuda");history=[];stages={};batch=256
    sources_hard=[s for s,rs in by.items() if rs[0]["kind"]=="hard"]
    sources_agent=[s for s,rs in by.items() if rs[0]["kind"]=="agent"]
    save(ROOT/"train/training-config.json",{"seed":SEED,"batch_size":batch,"optimizer":"AdamW","weight_decay":.01,
         "max_lr":{"qa":.001,"hard":.0003,"agent":.0003},"schedule":"OneCycleLR, 10% warmup, cosine decay",
         "maximum_steps":{"qa":1500,"hard":2500,"agent":3000},"qa_replay_fraction":.4,"head_config":HEAD_CONFIG})
    for stage,steps in [("qa",1500),("hard",2500),("agent",3000)]:
        output=ROOT/"train";output.mkdir(exist_ok=True)
        optimizer=torch.optim.AdamW(heads.parameters(),lr=1e-3 if stage=="qa" else 3e-4,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3 if stage=="qa" else 3e-4,
                                                       total_steps=steps,pct_start=.1,anneal_strategy="cos")
        best=(-1.,-1e9);best_step=0;sampled=Counter();unique=defaultdict(set);stage_start=time.monotonic();stale=0
        for step in range(1,steps+1):
            if time.monotonic()>deadline-75:raise RuntimeError("Time budget insufficient to finish all three stages; checkpoints saved")
            if stage=="qa" or rng.random()<.4:source="paq"
            else:source=rng.choice(sources_hard if stage=="hard" else sources_agent)
            rs=rng.sample(by[source],min(batch,len(by[source])))
            optimizer.zero_grad(set_to_none=True)
            loss=candidate_loss(heads,rs,vectors) if rs[0]["kind"]=="hard" else contrastive_loss(heads,rs,vectors,data)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(heads.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError("Nonfinite gradient")
            optimizer.step();scheduler.step();sampled[source]+=len(rs);unique[source].update(r["id"] for r in rs)
            if step==1 or step%100==0:print(f"V2 {stage} {step}/{steps}; loss {float(loss):.4f}; elapsed {time.monotonic()-stage_start:.0f}s",flush=True)
            if step%250==0 or step==steps or step==100:
                relevant=[t for t in val if (stage!="qa" or t["source"]=="paq")]
                result,_=summarize(relevant,cached_logits(heads,relevant,vectors))
                if stage=="qa":score=result["by_source"]["paq"]["accuracy"]
                else:score=result["nonqa_macro"]
                nll=float(np.mean([x["nll_covered"] for x in result["by_source"].values()]))
                history.append({"stage":stage,"step":step,"loss":float(loss),"validation":result})
                if (score,-nll)>best:
                    best=(score,-nll);best_step=step;stale=0
                    save_file({k:v.detach().cpu().contiguous() for k,v in heads.state_dict().items()},str(output/f"{stage}.safetensors"))
                else:stale+=1
                save(output/"history.json",history)
                print(f"V2 {stage} validation {score:.4f}; best step {best_step}",flush=True)
                if stale>=6 and step>=1500:break
        heads.load_state_dict(load_file(str(output/f"{stage}.safetensors"),device="cuda"))
        stages[stage]={"steps":step,"best_step":best_step,"selection_score":best[0],"seconds":time.monotonic()-stage_start,
                       "sampled":dict(sampled),"unique":{s:len(x) for s,x in unique.items()}}
        save(output/"stages.json",stages)
    save(ROOT/"train/config.json",{"head_config":HEAD_CONFIG,"format":FORMAT,"model_revision":MODEL_REVISION})
    return {"status":"complete","stages":stages,"head_parameters":sum(p.numel() for p in heads.parameters()),
            "backbone_frozen":True,"selected_stage":"agent","test_used":False,"peak_cuda_gib":torch.cuda.max_memory_allocated()/2**30}


def clm_logits(tasks,deadline):
    from clm_heads import make_head
    path=Path(os.environ.get("CLM_QWEN_PATH",str(ROOT.parent/"qwen-reference")))
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True);tok.padding_side="right"
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=load_base(path)
    ck=torch.load(str(Path(os.environ.get("CLM_HEAD_PATH",str(ROOT.parent/"clm-reference/CLM_v0.1-8B.pt")))),map_location="cpu",weights_only=True)
    cfg=ck["cfg"]
    opts=dict(width=cfg["width"],depth=cfg["depth"],proj=ck.get("projection_dim",cfg.get("projection_dim",512)),
              hidden=cfg.get("hidden_size",4096),activation=cfg.get("activation","gelu"),
              layernorm=cfg.get("layernorm",False),residual=cfg.get("residual",False))
    sh,ah=make_head(**opts).to("cuda"),make_head(**opts).to("cuda")
    sh.load_state_dict(ck["state_head"]);ah.load_state_dict(ck["action_head"]);sh.eval();ah.eval()
    scale=torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100).item()
    texts=sorted({t for r in tasks if r.get("eligible",True) for t in [r["state"],*r["candidates"]]});vectors={}
    with torch.inference_mode():
        for start in range(0,len(texts),8):
            if time.monotonic()>deadline-45:raise RuntimeError("Reference evaluation time budget exhausted")
            batch=texts[start:start+8]
            if max(len(tok(t)["input_ids"]) for t in batch)>8192:raise RuntimeError("Reference input exceeds 8192 tokens")
            v=encode_texts(model,tok,batch,batch_size=8)
            vectors.update(zip(batch,v))
            if start%160==0:print(f"CLM reference embeddings {start}/{len(texts)}",flush=True)
        logits=[]
        for t in tasks:
            if not t.get("eligible",True):logits.append(None);continue
            s=torch.tensor(vectors[t["state"]][None],device="cuda")
            a=torch.tensor(np.stack([vectors[c] for c in t["candidates"]]),device="cuda")
            z=(F.normalize(sh(s),dim=-1)@F.normalize(ah(a),dim=-1).T)[0]*scale
            logits.append(z.cpu().tolist())
    del model,sh,ah,ck;torch.cuda.empty_cache()
    return logits,{"head_config":opts,"qwen_revision":QWEN_REVISION,"clm_revision":CLM_REVISION,
                   "backend":"Transformers BF16 SDPA, last non-padding token and L2 normalization; raw text without chat template.",
                   "backend_caveat":"Reference architecture/weights reproduced with Transformers; vLLM numerical parity not separately certified."}


def reference(deadline):
    tasks=json.loads((ROOT/"validation-tasks.json").read_text())
    # A fixed first 32 validation examples per source; no final test scoring here.
    counts=Counter();subset=[]
    for r in tasks:
        if counts[r["source"]]<32:subset.append(r);counts[r["source"]]+=1
    logits,info=clm_logits(subset,deadline)
    report,out=summarize(subset,logits)
    save(ROOT/"reference/predictions.json",[{"id":t["id"],"source":t["source"],"logits":l} for t,l in zip(subset,logits)])
    save(ROOT/"reference/reference-info.json",info)
    return {"status":"complete","development_only":True,"results":report,"reference":info}


def run(stage,seconds):
    if not torch.cuda.is_available():raise RuntimeError("Cloud CUDA required; no Mac model execution")
    torch.set_num_threads(2);deadline=time.monotonic()+seconds
    if stage=="reference":return reference(deadline)
    if stage=="evaluate":
        from evaluate import evaluate
        return evaluate(deadline)
    if stage=="package":
        from package import verify_package
        return verify_package(deadline)
    if stage=="benchmark":
        from public_benchmark import benchmark
        return benchmark(deadline)
    data=Data()
    if stage=="cache":return cache(data,deadline)
    if stage=="train":return train(data,deadline)
    raise ValueError(stage)
