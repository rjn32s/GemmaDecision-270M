"""Frozen evaluation; test outcomes cannot select model, prompt or temperature."""
from collections import defaultdict
import json
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file
from common import *


def likelihood(model,tokenizer,tasks,deadline):
    scores=[]
    with torch.inference_mode():
        for index,t in enumerate(tasks):
            if not t.get("eligible",True):scores.append(None);continue
            if time.monotonic()>deadline-60:raise RuntimeError("Likelihood evaluation exceeded allocation")
            prefix=tokenizer(t["state"]+"\n\nCandidate response:\n",add_special_tokens=True)["input_ids"]
            out=[]
            for candidate in t["candidates"]:
                action=tokenizer(candidate,add_special_tokens=False)["input_ids"]
                if not action:raise RuntimeError("Empty candidate tokens")
                ids=torch.tensor([prefix+action],device="cuda")
                h=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state[0]
                h=h[len(prefix)-1:len(prefix)+len(action)-1]
                total=0.
                for start in range(0,len(action),32):
                    z=F.linear(h[start:start+32],model.get_input_embeddings().weight)
                    cap=getattr(model.config,"final_logit_softcapping",None)
                    if cap is not None:z=(z/cap).tanh()*cap
                    z=z.float();labels=torch.tensor(action[start:start+32],device="cuda")
                    total+=float((z[torch.arange(len(labels),device="cuda"),labels]-torch.logsumexp(z,-1)).sum())
                out.append(total/len(action))
            scores.append(out)
            if (index+1)%100==0:print(f"Gemma likelihood {index+1}/{len(tasks)}",flush=True)
    return scores


def project_tasks(tasks,vectors,heads=None):
    out=[]
    with torch.inference_mode():
        for t in tasks:
            if not t.get("eligible",True):out.append(None);continue
            s=torch.tensor(vectors[t["state"]][None],device="cuda")
            a=torch.tensor(np.stack([vectors[c] for c in t["candidates"]]),device="cuda")
            if heads is None:z=(F.normalize(s,dim=-1)@F.normalize(a,dim=-1).T)[0]/.07
            else:
                s,a=heads.project(s,a);z=(s@a.T)[0]*heads.scale()
            out.append(z.cpu().tolist())
    return out


def evaluate(deadline):
    from run import ROOT,BASE,load_base,summarize,clm_logits
    output=ROOT/"evaluate";output.mkdir(exist_ok=True)
    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"train").glob("*.safetensors")}
    if set(checkpoints)!={"qa.safetensors","hard.safetensors","agent.safetensors"}:raise RuntimeError("All three stage checkpoints required")
    save(output/"frozen-checkpoints.json",checkpoints)
    calibration=json.loads((ROOT/"calibration-tasks.json").read_text())
    tasks=json.loads((ROOT/"test-tasks.json").read_text())
    save(output/"protocol.json",json.loads((ROOT/"protocol.json").read_text()))
    save(output/"task-identities.json",[{k:t[k] for k in ["id","group","source","target"]}|{"eligible":t.get("eligible",True)} for t in tasks])
    model=load_base();tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tokenizer.padding_side="right"
    texts=sorted({text for t in tasks+calibration if t.get("eligible",True) for text in [t["state"],*t["candidates"]]})
    vectors={}
    for start in range(0,len(texts),128):
        chunk=texts[start:start+128];v=encode_texts(model,tokenizer,chunk,batch_size=16);vectors.update(zip(chunk,v))
    logits={"gemma_cosine":project_tasks(tasks,vectors)}
    heads=make_heads().to("cuda").eval().requires_grad_(False)
    for stage in ["qa","hard","agent"]:
        heads.load_state_dict(load_file(str(ROOT/f"train/{stage}.safetensors"),device="cuda"))
        logits[stage]=project_tasks(tasks,vectors,heads)
    cal=project_tasks(calibration,vectors,heads)
    temperatures=np.exp(np.linspace(np.log(.1),np.log(10),161))
    def cal_loss(temp):
        losses=defaultdict(list)
        for t,z in zip(calibration,cal):
            a=np.array(z)/temp;a-=a.max();losses[t["source"]].append(float(np.log(np.exp(a).sum())-a[t["target"]]))
        return float(np.mean([np.mean(v) for v in losses.values()]))
    temperature=float(min(temperatures,key=cal_loss))
    save(output/"calibration.json",{"temperature":temperature,"calibration_only":True,"raw_macro_nll":cal_loss(1.),"calibrated_macro_nll":cal_loss(temperature)})
    logits["gemma_likelihood"]=likelihood(model,tokenizer,tasks,deadline-900)
    # Loaded-model latency includes tokenization, embedding and head scoring.
    timings=[];permutation_errors=[]
    with torch.inference_mode():
        for t in [t for t in tasks if t.get("eligible",True)][:32]:
            torch.cuda.synchronize();start=time.monotonic()
            v=encode_texts(model,tokenizer,[t["state"],*t["candidates"]],batch_size=8)
            s,a=heads.project(torch.tensor(v[:1],device="cuda"),torch.tensor(v[1:],device="cuda"))
            z=(s@a.T)[0]*heads.scale();torch.cuda.synchronize();timings.append(time.monotonic()-start)
            permutation_errors.append(float((z-(s@a.flip(0).T)[0].flip(0)*heads.scale()).abs().max()))
    del model,heads;torch.cuda.empty_cache()
    logits["clm8b"],reference_info=clm_logits(tasks,deadline)
    save(output/"reference-info.json",reference_info)
    results={};outcomes={}
    for name,values in logits.items():results[name],outcomes[name]=summarize(tasks,values)
    results["agent_calibrated"],_=summarize(tasks,logits["agent"],temperature)
    # Source-stratified paired bootstrap over task groups; not individual turns.
    grouped=defaultdict(lambda:defaultdict(list))
    for i,t in enumerate(tasks):
        if t["source"]!="paq":grouped[t["source"]][t["group"]].append(i)
    rng=np.random.default_rng(SEED+10);intervals={}
    for baseline in ["gemma_likelihood","gemma_cosine","clm8b"]:
        differences=np.array([a["correct"]-b["correct"] for a,b in zip(outcomes["agent"],outcomes[baseline])])
        boots=[]
        for repeat in range(1000):
            macros=[]
            for groups in grouped.values():
                blocks=list(groups.values());ids=[i for j in rng.integers(0,len(blocks),len(blocks)) for i in blocks[j]]
                macros.append(float(differences[ids].mean()))
            boots.append(float(np.mean(macros)))
        intervals[baseline]={"difference":results["agent"]["nonqa_macro"]-results[baseline]["nonqa_macro"],
                             "bootstrap_95":np.quantile(boots,[.025,.975]).tolist()}
    gains=sum(results["agent"]["by_source"][s]["accuracy"]>results["gemma_likelihood"]["by_source"][s]["accuracy"] for s in grouped)
    success=intervals["gemma_likelihood"]["difference"]>=.05 and intervals["gemma_likelihood"]["bootstrap_95"][0]>0 and gains>=2
    predictions=[{**{k:t[k] for k in ["id","group","source","target"]},"logits":{s:v[i] for s,v in logits.items()}} for i,t in enumerate(tasks)]
    save(output/"predictions.json",predictions)
    report={"status":"complete","systems":results,"paired_nonpaq_intervals":intervals,"quality_target_met":success,
            "temperature":temperature,"decision_sources_improved":gains,
            "latency":{"p50_seconds":float(np.median(timings)),"p95_seconds":float(np.quantile(timings,.95)),
                       "scope":"Loaded model, no embedding cache; tokenization and encoding included; excludes model load and network.",
                       "max_candidate_permutation_error":max(permutation_errors)},
            "limitations":["Non-PAQ macro combines factual choices, tool preferences and recorded-action retrieval; it is not JevBench.",
                           "Agent distractors may be alternate valid actions; retrieval agreement is not actual task completion.",
                           "Terminal positive trajectories use source completion flags, not independently rerun environment rewards.",
                           "CLM reference uses original weights with Transformers embedding backend; exact vLLM numerical parity is unverified.",
                           "No direct JEV API comparison; no general JEV parity claim."]}
    save(output/"metrics.json",report)
    return report
