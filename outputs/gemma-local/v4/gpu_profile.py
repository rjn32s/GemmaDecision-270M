"""Cloud-only, matched-workload full-backbone Gemma-270M GPU profiling.

Reads training rows only. All parameters and AdamW master/state tensors are
float32; BF16 autocast accelerates forward/backward. Profiling updates are thrown
away. No checkpoint is saved and no input dataset is modified.
"""
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from common import schema_state

SEED = 2704205
STATE_CAP, CANDIDATE_CAP = 2048, 768
WARMUP_STEPS, MEASURED_STEPS = 3, 9
TIERS = {"short": (1, 256, 192), "medium": (384, 768, 640), "long": (1536, 3072, 2304)}
SEPARATOR = "\n\nCandidate action:\n"


def make_head():
    return nn.Sequential(nn.LayerNorm(640), nn.Linear(640,512), nn.GELU(), nn.Linear(512,1))


def joint_text(rendered, candidate): return rendered + SEPARATOR + candidate


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()).hexdigest()


def _target(row):
    if "target_probs" in row: value = np.asarray(row["target_probs"], dtype=np.float64)
    else:
        value = np.zeros(len(row["candidates"]), dtype=np.float64); value[int(row["target"])] = 1.
    if len(value)!=len(row["candidates"]) or not np.isfinite(value).all() or (value<0).any() or abs(value.sum()-1)>1e-5:
        raise ValueError("Invalid training target distribution")
    return value.tolist()


def _select(tokenizer, training_path, deadline):
    pools = {name:[] for name in TIERS}; counts = Counter(); token_cache = {}
    def tokens(text):
        if text not in token_cache: token_cache[text] = tokenizer(text, add_special_tokens=True)["input_ids"]
        return token_cache[text]
    print("GPU profile: selecting fixed batches from training rows only",flush=True)
    with training_path.open() as stream:
        for number, line in enumerate(stream):
            if not line.strip(): continue
            if number%512==0 and time.monotonic()>deadline-10: raise TimeoutError("Deadline during training-only batch selection")
            row=json.loads(line); counts["training_rows"]+=1
            if not 2<=len(row["candidates"])<=4:
                counts["excluded_candidate_count"]+=1;continue
            rendered=schema_state(row["state"],row.get("question",""))
            if len(tokens(rendered))>STATE_CAP or any(len(tokens(candidate))>CANDIDATE_CAP for candidate in row["candidates"]):
                counts["excluded_length"]+=1;continue
            pairs=[joint_text(rendered,candidate) for candidate in row["candidates"]]
            ids=[tokens(pair) for pair in pairs];maximum=max(map(len,ids))
            prepared={"id":row["id"],"group":row["group"],"source":row["source"],"ids":ids,
                      "targets":_target(row),"joint_texts":pairs,"maximum_joint_tokens":maximum}
            for name,(low,high,target) in TIERS.items():
                if low<=maximum<=high:pools[name].append(prepared)
    batches={};seen=set()
    for name,(_,_,target) in TIERS.items():
        ordered=sorted(pools[name],key=lambda row:(abs(row["maximum_joint_tokens"]-target),
                         hashlib.sha256((str(SEED)+'|'+row['id']).encode()).hexdigest()))
        chosen=[]
        for row in ordered:
            if row["group"] in seen:continue
            chosen.append(row);seen.add(row["group"])
            if len(chosen)==4:break
        if len(chosen)!=4:raise RuntimeError(f"Only {len(chosen)} distinct training groups available for {name}; workload not reduced")
        batches[name]=chosen
    protocol={"seed":SEED,"training_file_sha256":hashlib.sha256(training_path.read_bytes()).hexdigest(),
              "tier_definitions":{name:{"minimum_joint_tokens":lo,"maximum_joint_tokens":hi,"target_joint_tokens":target}
                                  for name,(lo,hi,target) in TIERS.items()},
              "selection_counts":dict(counts),"batches":{},"warmup_tiers":list(TIERS),
              "measured_tiers":list(TIERS)*3,"prompt_groups_per_update":4,"microbatches_per_update":1,
              "state_limit":STATE_CAP,"candidate_limit":CANDIDATE_CAP,"truncation":False}
    for name,rows in batches.items():
        protocol["batches"][name]={"ids":[row["id"] for row in rows],"sources":[row["source"] for row in rows],
            "joint_lengths":[[len(ids) for ids in row["ids"]] for row in rows],
            "candidate_counts":[len(row["ids"]) for row in rows],
            "nonpadding_tokens":sum(len(ids) for row in rows for ids in row["ids"]),
            "padded_shape":[sum(len(row["ids"]) for row in rows),max(len(ids) for row in rows for ids in row["ids"])],
            "batch_sha256":_digest([{key:row[key] for key in ["id","group","source","joint_texts","targets"]} for row in rows])}
    protocol["workload_sha256"]=_digest(protocol)
    print("GPU profile fixed shapes: "+json.dumps({name:value['padded_shape'] for name,value in protocol['batches'].items()}),flush=True)
    return batches,protocol


def _batch(rows, pad_token_id):
    sequences=[ids for row in rows for ids in row["ids"]];width=max(map(len,sequences))
    ids=torch.full((len(sequences),width),pad_token_id,dtype=torch.long,device="cuda");mask=torch.zeros_like(ids)
    targets=[]
    for index,sequence in enumerate(sequences):
        ids[index,:len(sequence)]=torch.tensor(sequence,device="cuda");mask[index,:len(sequence)]=1
    for row in rows:targets.append(torch.tensor(row["targets"],dtype=torch.float32,device="cuda"))
    return ids,mask,targets


def _loss(scores, targets):
    losses=[];offset=0
    for target in targets:
        values=scores[offset:offset+len(target)];offset+=len(target)
        losses.append(-(target*F.log_softmax(values.float(),dim=-1)).sum())
    if offset!=len(scores):raise RuntimeError("Scalar scores do not match candidate groups")
    return torch.stack(losses).mean()


def run(seconds=90):
    if not torch.cuda.is_available():raise RuntimeError("Full-backbone profiling runs on a cloud CUDA GPU only")
    if not 30<=seconds<=180:raise ValueError("GPU profile allocation must be30–180seconds")
    if not torch.cuda.is_bf16_supported():raise RuntimeError("This matched profile requires BF16 hardware support")
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();deadline=started+seconds
    base=Path(os.environ.get("GEMMA_MODEL_PATH","/experiment/work/model"))
    training_path=Path(os.environ.get("V4_TRAINING_FILE","/experiment/work/decision-v3/data/train.jsonl"))
    if training_path.name!='train.jsonl':raise ValueError("Profiler may only read the training split")
    properties=torch.cuda.get_device_properties(torch.cuda.current_device())
    report={"status":"running","allocated_seconds":seconds,"hardware":{"name":properties.name,
             "total_vram_bytes":properties.total_memory,"compute_capability":[properties.major,properties.minor]},
            "runtime":{"torch":torch.__version__,"cuda":torch.version.cuda,"python":platform.python_version()},
            "configuration":{"seed":SEED,"encoder_parameters":"all trainable float32",
                "autocast":"bfloat16","head":"LayerNorm640→Linear512→GELU→Linear1",
                "optimizer":"AdamW, fused=True","encoder_lr":1e-5,"head_lr":3e-4,"weight_decay":.01,
                "warmup_updates":WARMUP_STEPS,"measured_updates":MEASURED_STEPS,
                "microbatches_per_update":1,"activation_checkpointing":False,"tf32":False},
            "updates":[],"checkpoint_saved":False,"profiling_updates_discarded":True,
            "development_calibration_final_access":False,
            "timing_scope":"Synchronized forward, loss, backward, finite global norm/clip and fused optimizer step on GPU-resident fixed batches. Input staging, tokenization, zero_grad and model initialization are excluded from update latency but included in elapsed_seconds."}
    encoder=head=optimizer=None
    try:
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        batches,protocol=_select(tokenizer,training_path,deadline);report["protocol"]=protocol
        if time.monotonic()>deadline-10:raise TimeoutError("Deadline before model initialization")
        encoder=AutoModel.from_pretrained(base,local_files_only=True,dtype=torch.float32,
                                         attn_implementation="sdpa").to("cuda").train().requires_grad_(True)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        head=make_head().to(device="cuda",dtype=torch.float32).train()
        parameters=list(encoder.parameters())+list(head.parameters())
        if not all(parameter.requires_grad and parameter.dtype==torch.float32 for parameter in parameters):
            raise RuntimeError("Full-backbone FP32 trainability check failed")
        optimizer=torch.optim.AdamW([{"params":list(encoder.parameters()),"lr":1e-5},
                                     {"params":list(head.parameters()),"lr":3e-4}],weight_decay=.01,fused=True)
        sample_name,sample=next((name,parameter) for name,parameter in encoder.named_parameters() if name.endswith('self_attn.q_proj.weight'))
        before=sample.detach().flatten()[:64].clone()
        report["initial_encoder_sample_sha256"]=hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest()
        head_hasher=hashlib.sha256()
        for parameter in head.parameters(): head_hasher.update(parameter.detach().cpu().numpy().tobytes())
        report["initial_head_sha256"]=head_hasher.hexdigest()
        report["base_config_sha256"]=hashlib.sha256((base/"config.json").read_bytes()).hexdigest()
        report["trainable_parameters"]={"encoder":sum(p.numel() for p in encoder.parameters()),
                                         "head":sum(p.numel() for p in head.parameters()),"total":sum(p.numel() for p in parameters)}
        staged={name:_batch(rows,tokenizer.pad_token_id) for name,rows in batches.items()}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        sequence=list(TIERS)+list(TIERS)*3
        for index,tier in enumerate(sequence):
            if time.monotonic()>deadline-5:raise TimeoutError("Deadline before completing the fixed update workload")
            ids,mask,targets=staged[tier];optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                hidden=encoder(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                last=mask.sum(-1)-1
                vectors=F.normalize(hidden[torch.arange(len(ids),device='cuda'),last].float(),dim=-1)
                logits=head(vectors).squeeze(-1);loss=_loss(logits,targets)
            if not torch.isfinite(loss):raise RuntimeError("Non-finite full-backbone loss")
            loss.backward()
            if index==0:
                grad=float(sample.grad.float().norm()) if sample.grad is not None else 0.
                if grad<=0:raise RuntimeError("Encoder gradient sample is zero")
                report["encoder_gradient_check"]={"parameter":sample_name,"gradient_norm":grad}
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if not torch.isfinite(norm):raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            if index==0:
                changed=bool(not torch.equal(before,sample.detach().flatten()[:64]))
                if not changed:raise RuntimeError("Encoder sample weights did not change")
                report["encoder_weight_sample_changed"]=changed
                states=list(optimizer.state.values())
                report["optimizer_states"]={"parameter_tensors":len(states),
                    "expected_gradient_parameter_tensors":sum(p.grad is not None for p in parameters),
                    "state_tensor_bytes":sum(value.numel()*value.element_size() for state in states for value in state.values() if torch.is_tensor(value)),
                    "fp32_adam_moments":all(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32 for state in states)}
                if len(states)!=sum(p.grad is not None for p in parameters) or not report['optimizer_states']['fp32_adam_moments']:
                    raise RuntimeError("FP32 AdamW state materialization check failed")
            update={"index":index+1,"warmup":index<WARMUP_STEPS,"tier":tier,
                    "seconds":elapsed,"loss":float(loss.detach()),
                    "nonpadding_tokens":protocol['batches'][tier]['nonpadding_tokens'],
                    "padded_shape":protocol['batches'][tier]['padded_shape']}
            report['updates'].append(update)
            print(f"Full-backbone profile {index+1}/{len(sequence)} {tier}: {elapsed:.3f}s; loss {update['loss']:.4f}; {torch.cuda.max_memory_allocated()/2**30:.2f}GiB peak",flush=True)
        measured=[row for row in report['updates'] if not row['warmup']]
        report['summary']={"measured_updates":len(measured),"median_update_seconds":float(np.median([x['seconds'] for x in measured])),
            "total_measured_seconds":sum(x['seconds'] for x in measured),
            "actual_nonpadding_tokens":sum(x['nonpadding_tokens'] for x in measured),
            "nonpadding_tokens_per_second":sum(x['nonpadding_tokens'] for x in measured)/sum(x['seconds'] for x in measured),
            "by_tier":{tier:{"updates":sum(x['tier']==tier for x in measured),
                "median_seconds":float(np.median([x['seconds'] for x in measured if x['tier']==tier])),
                "nonpadding_tokens_per_second":sum(x['nonpadding_tokens'] for x in measured if x['tier']==tier)/sum(x['seconds'] for x in measured if x['tier']==tier)} for tier in TIERS}}
        report['status']='complete'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='failed',failure='cuda_out_of_memory',failure_detail=str(error)[:500],workload_reduced=False)
    except TimeoutError as error:
        report.update(status='failed',failure='deadline',failure_detail=str(error),workload_reduced=False)
    except Exception as error:
        report.update(status='failed',failure=type(error).__name__,failure_detail=str(error)[:800],workload_reduced=False)
    finally:
        report['peak_allocated_gpu_bytes']=torch.cuda.max_memory_allocated()
        report['peak_reserved_gpu_bytes']=torch.cuda.max_memory_reserved()
        report['elapsed_seconds']=time.monotonic()-started
        del optimizer,head,encoder
        gc.collect();torch.cuda.empty_cache()
    report['interpretation']='Matched training-step hardware profile, not an accuracy experiment. Compare only complete runs with identical workload hashes; model updates are discarded.'
    return report
