"""Bounded implementation-speed probe; no development or final-data access.

Fresh temporary models run the same first three training batches for each variant.
No trained checkpoint, optimizer state, source data or existing training artifact
is written. Training labels enter CE only; speed selection never uses accuracy.
"""
from collections import defaultdict, Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

VARIANTS = (
    {"name": "checkpoint_on_micro2", "checkpointing": True, "microbatch": 2},
    {"name": "checkpoint_off_micro2", "checkpointing": False, "microbatch": 2},
    {"name": "checkpoint_on_micro8", "checkpointing": True, "microbatch": 8},
    {"name": "checkpoint_off_micro8", "checkpointing": False, "microbatch": 8},
)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _batches(train_path, seed):
    """Mirror training's sampling with source balance, without reading dev rows."""
    pools = defaultdict(lambda: defaultdict(list)); excluded = Counter()
    with Path(train_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train": raise RuntimeError("Profile input contains a non-training row")
            if row.get("within_pilot_limits") is not True:
                excluded[row["source"]] += 1; continue
            pools[row["category"]][row["source"]].append(row)
    for category in ("decision", "qa", "agent"):
        if not pools[category]: raise RuntimeError("Profile lacks a required training category")
        for source in pools[category]: pools[category][source].sort(key=lambda row: row["id"])
    rng = random.Random(seed); batches = []
    for _ in range(3):
        batch = []
        for category, count in (("decision", 4), ("qa", 2), ("agent", 2)):
            sources = sorted(pools[category])
            for _ in range(count):
                source = rng.choice(sources); batch.append(rng.choice(pools[category][source]))
        rng.shuffle(batch); batches.append(batch)
    return batches, dict(excluded)


def _parameter_digest(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _variant(config, batches, base, initial_heads, seed, deadline):
    # Imports are deliberately deferred; importing this file never loads a model.
    import torch
    from common import schema_state
    from model import DecisionModel
    from train import objective

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    model = DecisionModel(base, initial_heads, arm="lora")
    if config["checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.encoder.gradient_checkpointing_disable()
    encoder = [(name, parameter) for name, parameter in model.encoder.named_parameters() if parameter.requires_grad]
    encoder_params = [parameter for _, parameter in encoder]
    heads = list(model.heads.parameters())
    parameters = heads + encoder_params
    initial_adapter_hash = _parameter_digest(encoder)
    zero_b = all(bool(torch.count_nonzero(parameter.detach()) == 0) for name, parameter in encoder if "lora_B" in name)
    if not zero_b: raise RuntimeError("Profile requires a fresh zero-output LoRA initialization")
    frozen = next(parameter for parameter in model.encoder.parameters() if not parameter.requires_grad)
    frozen_anchor = frozen.detach().flatten()[:256].clone()
    # Candidate and state text are the sole model inputs. Only the selected 24
    # training rows need tokenization; prep has already marked all pool lengths.
    prepared = []
    for batch in batches:
        current = []
        for row in batch:
            item = dict(row)
            item["rendered_state"] = schema_state(row["state"], row.get("question", ""))
            item["lengths"] = [len(model.ids(text)) for text in [item["rendered_state"], *item["candidates"]]]
            if item["lengths"][0] > 2048 or max(item["lengths"][1:]) > 768:
                raise RuntimeError("Profile selected a row outside the pilot length contract")
            current.append(item)
        prepared.append(current)
    optimizer = torch.optim.AdamW([
        {"params": heads, "lr": .0003}, {"params": encoder_params, "lr": .00005}
    ], weight_decay=.01)
    model.train()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.monotonic() - started
    steps = []; complete = True
    for index, batch in enumerate(prepared):
        if time.monotonic() >= deadline - 5:
            complete = False; break
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1., (index + 1) / 20)
        for group, peak in zip(optimizer.param_groups, (.0003, .00005)): group["lr"] = peak * warmup
        torch.cuda.synchronize(); tick = time.monotonic(); token_start = model.tokens_processed
        total_loss = 0.
        for offset in range(0, 8, config["microbatch"]):
            micro = batch[offset:offset + config["microbatch"]]
            logits = model.logits_batch(micro, gradient=True)
            loss = objective(logits, micro) * len(micro) / 8
            if not bool(torch.isfinite(loss)): raise RuntimeError("Non-finite profile loss")
            loss.backward(); total_loss += float(loss.detach())
        finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
        if not finite: raise RuntimeError("Non-finite profile gradients")
        head_norm = float(sum((p.grad.float().square().sum() for p in heads if p.grad is not None), torch.zeros((), device="cuda")).sqrt())
        encoder_norm = float(sum((p.grad.float().square().sum() for p in encoder_params if p.grad is not None), torch.zeros((), device="cuda")).sqrt())
        if not math.isfinite(head_norm) or not math.isfinite(encoder_norm) or head_norm <= 0 or encoder_norm <= 0: raise RuntimeError("Profile requires finite nonzero head and adapter gradients")
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        optimizer.step()
        torch.cuda.synchronize(); elapsed = time.monotonic() - tick
        steps.append({"update": index + 1, "warmup": index == 0, "seconds": elapsed,
                      "loss": total_loss, "head_gradient_norm": head_norm,
                      "adapter_gradient_norm": encoder_norm, "pre_clip_gradient_norm": pre_clip_norm,
                      "finite_gradients": finite,
                      "scheduled_tokens": sum(sum(row["lengths"]) for row in batch),
                      "encoder_forward_tokens": model.tokens_processed - token_start})
    measured = [step for step in steps if not step["warmup"]]
    measured_seconds = sum(step["seconds"] for step in measured)
    if not bool(torch.equal(frozen.detach().flatten()[:256], frozen_anchor)):
        raise RuntimeError("Frozen backbone parameter sample changed during implementation probe")
    adapter_updated = any(bool(torch.count_nonzero(parameter.detach()) > 0) for name, parameter in encoder if "lora_B" in name)
    if steps and not adapter_updated: raise RuntimeError("Temporary adapter did not update")
    return {
        **config, "status": "complete" if complete and len(steps) == 3 else "partial_deadline",
        "setup_seconds": setup_seconds, "steps": steps, "warmup_updates": min(len(steps), 1),
        "measured_updates": len(measured), "measured_seconds": measured_seconds,
        "measured_seconds_per_update": measured_seconds / len(measured) if measured else None,
        "measured_updates_per_second": len(measured) / measured_seconds if measured_seconds else None,
        "encoder_forward_tokens_per_second": sum(step["encoder_forward_tokens"] for step in measured) / measured_seconds if measured_seconds else None,
        "scheduled_tokens_per_second": sum(step["scheduled_tokens"] for step in measured) / measured_seconds if measured_seconds else None,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 2**30,
        "initial_adapter_sha256": initial_adapter_hash, "zero_lora_B_at_initialization": zero_b,
        "adapter_B_updated": adapter_updated, "frozen_backbone_sample_unchanged": True,
        "checkpointing_effective": bool(model.encoder.is_gradient_checkpointing),
    }


def run(seconds=120):
    """Compare implementations only; never resume or change an experiment arm."""
    if not isinstance(seconds, (int, float)) or not 15 <= seconds <= 120:
        raise ValueError("Profile seconds must be between 15 and 120")
    started = time.monotonic(); deadline = started + seconds
    import torch
    from common import save
    from train import SEED, ROOT, DATA_ROOT, BASE, V2_ROOT
    if not torch.cuda.is_available(): raise RuntimeError("This profile must run on a cloud CUDA GPU")
    output = ROOT / "profile"; output.mkdir(parents=True, exist_ok=True)
    train_path = DATA_ROOT / "train.jsonl"; initial_heads = V2_ROOT / "train" / "hard.safetensors"
    batches, exclusions = _batches(train_path, SEED)
    identities = [[row["id"] for row in batch] for batch in batches]
    provenance = {
        "seed": SEED, "train_data_sha256": _file_hash(train_path),
        "initial_head_sha256": _file_hash(initial_heads),
        "profile_code_sha256": _file_hash(__file__),
        "model_code_sha256": _file_hash(Path(__file__).with_name("model.py")),
        "training_code_sha256": _file_hash(Path(__file__).with_name("train.py")),
        "schedule_ids_sha256": _json_hash(identities), "schedule_ids": identities,
        "category_counts_per_batch": [dict(Counter(row["category"] for row in batch)) for batch in batches],
        "training_metadata_length_exclusions": exclusions,
    }
    report = {"status": "partial", "purpose": "Implementation throughput only; no accuracy/model selection",
              "budget_seconds": seconds, "provenance": provenance, "variants": [],
              "gpu": torch.cuda.get_device_name(), "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "torch_version": torch.__version__, "development_opened": False,
              "calibration_opened": False, "final_opened": False, "trained_checkpoints_written": False,
              "initialization": "Fresh v2 hard-decision heads and same-seed zero-B LoRA for every variant",
              "math": "Candidate-distribution CE averaged over 8 rows; same 4/2/2 mixture, source balance, AdamW, first-three-update LR warmup, and gradient clip 1 as pilot",
              "measurement": "First update warms model/optimizer; next two identical updates timed with CUDA synchronize. Tokenization/model setup excluded from step time; finite-gradient checks included.",
              "limitations": ["Three short batches measure implementation behavior, not convergence or generalization.", "BF16 and different sequence batching may produce small rounding differences.", "A single timing probe has noise; longer production runs should verify sustained throughput.", "Deadline checks are between optimizer updates; an in-flight update finishes before stopping."]}
    save(output / "metrics.json", report)
    for config in VARIANTS:
        if time.monotonic() >= deadline - 10: break
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        try:
            result = _variant(config, batches, BASE, initial_heads, SEED, deadline)
        except torch.cuda.OutOfMemoryError:
            result = {**config, "status": "out_of_memory"}
        report["variants"].append(result)
        # _variant returns only scalars; temporary models/optimizers become
        # unreachable before the next fresh implementation is constructed.
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        print(f"Profile {config['name']}: {result['status']}; measured updates {result.get('measured_updates', 0)}", flush=True)
        report["seconds"] = time.monotonic() - started
        save(output / "metrics.json", report)
    completed = [variant for variant in report["variants"] if variant["status"] == "complete"]
    baseline = next((v for v in completed if v["name"] == "checkpoint_on_micro2"), None)
    if baseline:
        for variant in completed:
            differences = [abs(step["loss"] - ref["loss"]) for step, ref in zip(variant["steps"], baseline["steps"])]
            tolerances = [max(.005, abs(ref["loss"]) * .01) for ref in baseline["steps"]]
            variant["loss_absolute_differences_from_baseline"] = differences
            variant["loss_agreement_tolerances"] = tolerances
            variant["loss_agreement_diagnostic_passed"] = all(delta <= tolerance for delta, tolerance in zip(differences, tolerances))
            variant["same_adapter_initialization"] = variant["initial_adapter_sha256"] == baseline["initial_adapter_sha256"]
            variant["speedup_vs_baseline"] = baseline["measured_seconds_per_update"] / variant["measured_seconds_per_update"]
        candidates = [v for v in completed if v["loss_agreement_diagnostic_passed"] and v["same_adapter_initialization"]]
        report["fastest_numerically_agreeing_implementation"] = min(candidates, key=lambda v:v["measured_seconds_per_update"])["name"] if candidates else None
    report["status"] = "complete" if len(completed) == len(VARIANTS) else "partial"
    report["seconds"] = time.monotonic() - started
    save(output / "metrics.json", report)
    return report
