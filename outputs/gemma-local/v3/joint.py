"""Bounded frozen-Gemma ablation with request-conditioned candidate embeddings.

Each candidate is encoded together with the same state/question. A shared scalar
head ranks the resulting embeddings. Candidate order never enters model input.
Reads training and development only; does not publish or open final/calibration.
"""
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file, load_file
from common import save, MODEL_REVISION

SEED = 2704203
HEAD_CONFIG = {"hidden": 640, "width": 512}
FORMAT = "gemmadecision-joint-v1"
SEPARATOR = "\n\nCandidate action:\n"
ROOT = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
BASE = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))


def joint_text(rendered_state, candidate):
    """The exact serving contract: raw rendered state, fixed separator, candidate."""
    return rendered_state + SEPARATOR + candidate


def make_joint_head(config=None):
    config = config or HEAD_CONFIG
    return nn.Sequential(nn.LayerNorm(config["hidden"]), nn.Linear(config["hidden"], config["width"]),
                         nn.GELU(), nn.Linear(config["width"], 1))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _head_save(head, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().contiguous().cpu() for name, value in head.state_dict().items()},
              str(destination / "head.safetensors"))


class _Tokenizer:
    def __init__(self, tokenizer): self.tokenizer = tokenizer; self.cache = {}
    def ids(self, text):
        if text not in self.cache: self.cache[text] = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        return self.cache[text]


def _data(tokenizer, steps, output):
    import train
    if train.ROOT != ROOT: raise RuntimeError("Joint and matched training roots differ")
    previous_full = os.environ.get("V3_FULL_DEVELOPMENT")
    previous_name = os.environ.get("V3_OUTPUT_NAME")
    previous_steps = train.STEPS
    try:
        # load_rows only writes joint-data.json under this explicit namespace.
        os.environ["V3_FULL_DEVELOPMENT"] = "1"; os.environ["V3_OUTPUT_NAME"] = "joint"
        train.STEPS = steps
        schedule, development, matched = train.load_rows(_Tokenizer(tokenizer))
    finally:
        train.STEPS = previous_steps
        if previous_full is None: os.environ.pop("V3_FULL_DEVELOPMENT", None)
        else: os.environ["V3_FULL_DEVELOPMENT"] = previous_full
        if previous_name is None: os.environ.pop("V3_OUTPUT_NAME", None)
        else: os.environ["V3_OUTPUT_NAME"] = previous_name
    comparison_file = ROOT / "frozen_extension/metrics.json"
    if not comparison_file.exists(): raise RuntimeError("Completed separate-encoder comparison is required before the joint ablation")
    separate = json.loads(comparison_file.read_text())
    expected = separate["manifest"]
    keys = ["training_data_sha256", "development_data_sha256", "development_ids_sha256", "schedule_sha256", "requested_steps", "seed"]
    for key in keys:
        if matched[key] != expected[key]: raise RuntimeError(f"Joint and separate-encoder exposure differ: {key}")
    if separate["status"] != "complete" or separate["steps"] != steps:
        raise RuntimeError("Separate-encoder arm has not completed matched training exposure")
    manifest = {**matched, "format": FORMAT, "head_config": HEAD_CONFIG, "base_revision": MODEL_REVISION,
        "input_format": "rendered_state + '\\n\\nCandidate action:\\n' + candidate", "separator": SEPARATOR,
        "head_initialization": "New scalar head with PyTorch default initialization, seed2704203; unlike the prior CLM heads, it cannot reuse a differently shaped checkpoint.",
        "comparison_source_sha256": _sha(comparison_file), "matched_exposure_verified": True,
        "encoder_frozen": True, "embedding_cache_dtype": "float32", "final_and_calibration_opened": False,
        "head_parameter_count": sum(parameter.numel() for parameter in make_joint_head().parameters()),
        "separate_head_parameter_count": separate.get("checks", {}).get("trainable_head_parameters"),
        "optimization": {"batch": 8, "lr": .0003, "weight_decay": .01, "warmup_steps": 20,
                         "clip_grad_norm": 1., "validation_interval": 250,
                         "objective": "Candidate soft-target cross entropy; ordinal and preference ties retain their distributions"}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Joint experiment manifest changed; refusing incompatible cache/checkpoint reuse")
    save(manifest_path, manifest)
    print(f"Joint data: {steps} matched batches, {steps*8} row exposures, {len(development)} development cases", flush=True)
    return schedule, development, manifest


def _prepare_cache(schedule, development, tokenizer, output):
    rows = [row for batch in schedule for row in batch] + development
    # Dict deduplication does not inspect target values; inputs use only text fields.
    texts = sorted({joint_text(row["rendered_state"], candidate) for row in rows if row["eligible"] for candidate in row["candidates"]})
    indices = {text: index for index, text in enumerate(texts)}
    for row in rows:
        row["joint_indices"] = [indices[joint_text(row["rendered_state"], candidate)] for candidate in row["candidates"]] if row["eligible"] else []
    hasher = hashlib.sha256()
    for text in texts: hasher.update(text.encode()); hasher.update(b"\0")
    metadata = {"format": FORMAT, "text_sha256": hasher.hexdigest(), "count": len(texts), "base_revision": MODEL_REVISION}
    cache = output / "cache"; cache.mkdir(parents=True, exist_ok=True)
    protocol = cache / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != metadata: raise RuntimeError("Joint embedding inputs changed")
    save(protocol, metadata)
    token_path, offset_path = cache / "tokens.npy", cache / "offsets.npy"
    if not (token_path.exists() and offset_path.exists()):
        pieces = []; offsets = [0]; started = time.monotonic()
        for start in range(0, len(texts), 256):
            batch = tokenizer(texts[start:start+256], add_special_tokens=True)["input_ids"]
            for tokens in batch: pieces.append(np.asarray(tokens, dtype=np.uint32)); offsets.append(offsets[-1] + len(tokens))
            if start == 0 or start % 4096 == 0:
                print(f"Joint tokenization {min(start+256,len(texts))}/{len(texts)}; {time.monotonic()-started:.0f}s", flush=True)
        np.save(token_path, np.concatenate(pieces)); np.save(offset_path, np.asarray(offsets, dtype=np.int64))
    tokens = np.load(token_path, mmap_mode="r"); offsets = np.load(offset_path, mmap_mode="r")
    if len(offsets) != len(texts)+1 or offsets[-1] != len(tokens): raise RuntimeError("Joint token cache is incomplete")
    lengths = np.diff(offsets)
    save(cache / "statistics.json", {"texts": len(texts), "total_tokens": int(len(tokens)),
        "maximum_joint_tokens": int(lengths.max()), "state_limit": 2048, "candidate_limit": 768,
        "truncation": False, "joint_texts_beyond_3072": int((lengths > 3072).sum())})
    print(f"Joint cache: {len(texts)} texts / {len(tokens)} tokens; longest {int(lengths.max())}; no truncation", flush=True)
    return tokens, offsets, lengths


def _cache(tokens, offsets, lengths, tokenizer, output, deadline):
    cache = output / "cache"; progress_path = cache / "progress.json"; vector_path = cache / "embeddings.npy"
    count = len(lengths); order = np.argsort(lengths, kind="stable")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"offset": 0, "total": count}
    if progress["total"] != count: raise RuntimeError("Joint cache row count changed")
    offset = int(progress["offset"])
    if offset > 0 and not vector_path.exists(): raise RuntimeError("Cache progress exists without its embedding file")
    vectors = np.lib.format.open_memmap(vector_path, mode="r+" if vector_path.exists() else "w+", dtype=np.float32, shape=(count,640))
    if vectors.shape != (count,640) or vectors.dtype != np.float32: raise RuntimeError("Joint cache has unexpected shape or precision")
    if offset == count: return vectors, progress
    if time.monotonic() > deadline - 60: return None, {"offset": offset, "total": count, "status": "partial_cache"}
    model = AutoModel.from_pretrained(BASE, local_files_only=True, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
    if int(lengths.max()) > model.config.max_position_embeddings: raise RuntimeError("A joint input exceeds the encoder architecture's limit")
    first = next(model.parameters()); anchor = first.detach().flatten()[:256].clone()
    encoded_tokens = 0; started = time.monotonic(); latest = started
    print(f"Joint encoding resume {offset}/{count}", flush=True)
    with torch.inference_mode():
        while offset < count and time.monotonic() < deadline - 60:
            initial_length = int(lengths[order[offset]])
            size = min(64, max(1, 16384 // max(initial_length,1)))
            ids = order[offset:offset+size]
            while len(ids)>1 and len(ids)*int(lengths[ids].max())>16384: ids=ids[:-1]
            width = int(lengths[ids].max())
            values = np.full((len(ids),width), tokenizer.pad_token_id, dtype=np.int64); mask = np.zeros_like(values)
            for batch_index, index in enumerate(ids):
                sequence = tokens[offsets[index]:offsets[index+1]]
                values[batch_index,:len(sequence)] = sequence; mask[batch_index,:len(sequence)] = 1
            input_ids = torch.tensor(values, device="cuda"); attention_mask = torch.tensor(mask, device="cuda")
            hidden = model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
            last = attention_mask.sum(-1)-1
            embedded = F.normalize(hidden[torch.arange(len(ids),device="cuda"),last].float(),dim=-1)
            if not torch.isfinite(embedded).all(): raise RuntimeError("Non-finite joint embeddings")
            vectors[ids] = embedded.cpu().numpy(); offset += len(ids); encoded_tokens += int(lengths[ids].sum())
            if time.monotonic()-latest >= 20:
                vectors.flush(); save(progress_path, {"offset":offset,"total":count})
                print(f"Joint embeddings {offset}/{count}; {encoded_tokens/(time.monotonic()-started):.0f} tokens/s; {time.monotonic()-started:.0f}s", flush=True)
                latest = time.monotonic()
    unchanged = bool(torch.equal(first.detach().flatten()[:256],anchor))
    if not unchanged or any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Encoder freeze check failed")
    vectors.flush()
    progress = {"offset": offset, "total": count, "status": "complete" if offset==count else "partial_cache",
                "encoder_parameter_sample_unchanged": unchanged, "trainable_encoder_parameters":0,
                "invocation_encoded_tokens":encoded_tokens,"invocation_seconds":time.monotonic()-started,
                "peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30}
    save(progress_path, progress)
    del model, first
    gc.collect(); torch.cuda.empty_cache()
    return vectors if offset==count else None, progress


def _logits(head, rows, vectors):
    flat = [index for row in rows for index in row["joint_indices"]]
    values = head(vectors[flat]).squeeze(-1)
    result=[]; offset=0
    for row in rows:
        count=len(row["joint_indices"]); result.append(values[offset:offset+count]); offset+=count
    return result


@torch.no_grad()
def _evaluate(head, rows, vectors):
    from train import score_predictions
    head.eval(); values=[]
    for offset in range(0,len(rows),128):
        block=rows[offset:offset+128]; eligible=[row for row in block if row["eligible"]]
        scored=_logits(head,eligible,vectors) if eligible else []
        iterator=iter(scored)
        values.extend(next(iterator).float().cpu().tolist() if row["eligible"] else None for row in block)
    summary,outcomes=score_predictions(rows,values)
    return summary,values


def _checkpoint(head, optimizer, output, step, history, best, manifest):
    path=output/f"step-{step:05d}"; _head_save(head,path)
    torch.save({"optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all()},path/"optimizer.pt")
    progress={"step":step,"checkpoint":str(path),"best":best,"history":history,
              "manifest_sha256":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()}
    save(output/"progress.json",progress)


def _train(schedule, development, manifest, numpy_vectors, output, deadline, cache_info):
    from train import objective
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    head=make_joint_head().to("cuda")
    vectors=torch.from_numpy(np.asarray(numpy_vectors)).to("cuda")
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
    progress_path=output/"progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else None
    start_step=0; history=[]; best=None
    if progress:
        expected=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        if progress["manifest_sha256"]!=expected: raise RuntimeError("Joint checkpoint controls changed")
        path=Path(progress["checkpoint"]);head.load_state_dict(load_file(str(path/"head.safetensors"),device="cuda"))
        state=torch.load(path/"optimizer.pt",weights_only=True,map_location="cuda");optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start_step=progress["step"];history=progress["history"];best=progress["best"]
    if not history:
        metrics,values=_evaluate(head,development,vectors)
        history.append({"step":0,"development":metrics})
        best={"step":0,"score":metrics["fresh_family_macro"],"nll":metrics["fresh_family_nll"]}
        _head_save(head,output/"best");save(output/"best-predictions.json",values)
    first_matrix=next(parameter for parameter in head.parameters() if parameter.ndim==2)
    anchor=first_matrix.detach().clone();step=start_step;last_checkpoint=start_step;losses=[];gradient_norm=None
    started=time.monotonic()
    for index in range(start_step,len(schedule)):
        if time.monotonic()>deadline-35:break
        head.train();optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = .0003*min(1.,(index+1)/20)
        batch=schedule[index];loss=objective(_logits(head,batch,vectors),batch)
        if not torch.isfinite(loss):raise RuntimeError("Non-finite joint training loss")
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
            raise RuntimeError("Non-finite scalar-head gradients")
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if gradient_norm is None:
            gradient_norm=float(norm)
            if gradient_norm<=0:raise RuntimeError("Scalar head received no training gradient")
        optimizer.step();step=index+1;losses.append(float(loss.detach()))
        if step%100==0:
            print(f"Joint head {step}/{len(schedule)}; loss {np.mean(losses[-100:]):.4f}; elapsed {time.monotonic()-started:.0f}s",flush=True)
        if step%250==0 or step==len(schedule):
            metrics,values=_evaluate(head,development,vectors)
            history.append({"step":step,"loss":float(np.mean(losses[-250:])),"development":metrics})
            if (metrics["fresh_family_macro"],-metrics["fresh_family_nll"])>(best["score"],-best["nll"]):
                best={"step":step,"score":metrics["fresh_family_macro"],"nll":metrics["fresh_family_nll"]}
                _head_save(head,output/"best");save(output/"best-predictions.json",values)
            _checkpoint(head,optimizer,output,step,history,best,manifest);last_checkpoint=step
    if step!=last_checkpoint or not progress_path.exists():
        metrics,values=_evaluate(head,development,vectors)
        history.append({"step":step,"loss":float(np.mean(losses)) if losses else None,"development":metrics})
        if (metrics["fresh_family_macro"],-metrics["fresh_family_nll"])>(best["score"],-best["nll"]):
            best={"step":step,"score":metrics["fresh_family_macro"],"nll":metrics["fresh_family_nll"]}
            _head_save(head,output/"best");save(output/"best-predictions.json",values)
        _checkpoint(head,optimizer,output,step,history,best,manifest)
    changed=bool(not torch.equal(first_matrix.detach(),anchor)) if step>start_step else None
    if step>start_step and not changed:raise RuntimeError("Scalar head did not update")
    head.load_state_dict(load_file(str(output/"best/head.safetensors"),device="cuda"));head.eval()
    selected_metrics,selected_logits=_evaluate(head,development,vectors)
    sample=next(row for row in development if row["eligible"])
    with torch.no_grad():
        expected=_logits(head,[sample],vectors)[0]
        reverse=dict(sample,joint_indices=list(reversed(sample["joint_indices"])))
        reordered=_logits(head,[reverse],vectors)[0].flip(0)
        restored=make_joint_head().to("cuda").eval()
        restored.load_state_dict(load_file(str(output/"best/head.safetensors"),device="cuda"))
        reload_error=float((_logits(restored,[sample],vectors)[0]-expected).abs().max())
        permutation_error=float((reordered-expected).abs().max())
    if reload_error>1e-5 or permutation_error>1e-5:raise RuntimeError("Joint head reload or order invariance failed")
    report={"status":"complete" if step==len(schedule) else "partial_training","steps":step,"requested_steps":len(schedule),
            "best":best,"history":history,"selected_development":selected_metrics,"cache":cache_info,
            "checks":{"head_gradient_norm":gradient_norm,"head_changed_this_invocation":changed,
                      "reload_max_logit_error":reload_error,"candidate_permutation_max_logit_error":permutation_error,
                      "frozen_encoder_trainable_parameters":0,"head_parameters":sum(p.numel() for p in head.parameters())},
            "manifest":manifest,"selected_head_sha256":_sha(output/"best/head.safetensors"),
            "training_seconds":time.monotonic()-started,"new_updates":step-start_step,
            "architecture":"Frozen Gemma jointly encodes state/question and one candidate; shared scalar head ranks candidates.",
            "limitations":["Development selection only; no untouched-final or benchmark result.",
                "The scalar head has a different initialization and parameterization from the pretrained contrastive heads; this is an architecture-and-head ablation.",
                "Joint encoding repeats the context for each candidate and cannot globally cache candidate embeddings.",
                "Agent agreement remains recorded-action retrieval, not a live agent completion reward."],
            "final_and_calibration_opened":False,"publication_on_hold":True}
    save(output/"selected-predictions.json",selected_logits)
    save(output/"metrics.json",report)
    return report


def run(seconds=780,steps=3000):
    if not torch.cuda.is_available():raise RuntimeError("Joint model execution is authorized on Modal GPU only")
    if steps!=3000:raise ValueError("This bounded matched-exposure ablation is fixed to 3000 updates")
    if not 120<=seconds<=900:raise ValueError("Joint allocation must be120–900seconds")
    start=time.monotonic();deadline=start+seconds;torch.set_num_threads(2)
    output=ROOT/"joint";output.mkdir(parents=True,exist_ok=True)
    print("Joint ablation starting: frozen Gemma270M, full development, no final/calibration access",flush=True)
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tokenizer.padding_side="right"
    schedule,development,manifest=_data(tokenizer,steps,output)
    tokens,offsets,lengths=_prepare_cache(schedule,development,tokenizer,output)
    vectors,cache_info=_cache(tokens,offsets,lengths,tokenizer,output,deadline)
    if vectors is None:
        report={"status":"partial_cache","cache":cache_info,"manifest":manifest,"steps":0,
                "final_and_calibration_opened":False,"publication_on_hold":True}
        save(output/"metrics.json",report);return report
    report=_train(schedule,development,manifest,vectors,output,deadline,cache_info)
    report["total_seconds"]=time.monotonic()-start;save(output/"metrics.json",report)
    return report
