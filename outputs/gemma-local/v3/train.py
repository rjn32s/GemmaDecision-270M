"""Bounded, cloud-only v3 screening experiment; never reads final or calibration.

Stages: baselines, frozen, lora, compare. V3_STEPS defaults to 300 and may be
extended using the same deterministic schedule. Both arms see exactly the same
row batches. No publication or benchmark submission occurs here.
"""
from collections import defaultdict, Counter
import gc
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
from safetensors.torch import load_file
from common import save, read_jsonl, schema_state, HEAD_CONFIG
from model import DecisionModel, LORA_CONFIG, adapter_state

SEED = 2704203
ROOT = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
DATA_ROOT = Path(os.environ.get("V3_DATA_ROOT", str(ROOT / "data")))
BASE = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
V2_ROOT = Path(os.environ.get("V2_ROOT", "/experiment/work/decision-v2"))
MAX_STATE, MAX_ACTION = 2048, 768
STEPS = int(os.environ.get("V3_STEPS", "300"))
BATCH = 8


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def select_development(rows, cap=24):
    families = defaultdict(list)
    for row in rows: families[row["family"]].append(row)
    selected = []
    for family, candidates in sorted(families.items()):
        candidates.sort(key=lambda row: hashlib.sha256((str(SEED) + "|" + row["id"]).encode()).hexdigest())
        counts = Counter(); taken = 0
        for row in candidates:
            if counts[row["group"]] >= 2: continue
            selected.append(row); counts[row["group"]] += 1; taken += 1
            if taken >= cap: break
    return selected


def target_distribution(row):
    if "target_probs" in row:
        result = np.asarray(row["target_probs"], dtype=np.float64)
    else:
        result = np.zeros(len(row["candidates"]), dtype=np.float64)
        result[int(row["target"])] = 1.
    if len(result) != len(row["candidates"]) or not np.isfinite(result).all() or (result < 0).any() or abs(result.sum() - 1.) > 1e-5:
        raise ValueError(f"Invalid target distribution for {row['id']}")
    return result


def score_predictions(rows, logits):
    """Expected exact-choice credit for ambiguous labels; rejected rows score zero.

A 50/50 preference tie gives any deterministic top choice 0.5 expected credit,
not an arbitrary first-candidate label. CE and Brier preserve the full soft label.
Ordinal MAE compares expected predicted rating with the annotated mean rating.
"""
    if len(rows) != len(logits): raise ValueError("Prediction count does not match the development population")
    by_family = defaultdict(list); outcomes = []
    for row, values in zip(rows, logits):
        target = target_distribution(row)
        covered = values is not None
        record = {"id": row["id"], "group": row["group"], "family": row["family"],
                  "scope": row.get("evaluation_scope", "fresh_prompt"), "covered": covered,
                  "workflow_group": row.get("workflow_group", row["group"]), "source": row["source"],
                  "metric": row.get("metric", "categorical"),
                  "credit": 0., "nll": None, "brier": None, "ordinal_mae": None}
        if covered:
            z = np.asarray(values, dtype=np.float64)
            if len(z) != len(target) or not np.isfinite(z).all(): raise RuntimeError("Invalid decision logits")
            z -= z.max(); logp = z - np.log(np.exp(z).sum()); p = np.exp(logp)
            maxima = np.flatnonzero(np.isclose(z, z.max(), atol=1e-7, rtol=0))
            record.update(credit=float(target[maxima].mean()), nll=float(-(target * logp).sum()),
                          brier=float(((p - target) ** 2).sum()))
            if row.get("metric") == "ordinal":
                ratings = np.asarray(row.get("candidate_values", list(range(len(p)))), dtype=np.float64)
                expected = float(row.get("expected_rating", (target * ratings).sum()))
                record["ordinal_mae"] = float(abs((p * ratings).sum() - expected))
        by_family[row["family"]].append(record); outcomes.append(record)
    def aggregate(values):
        valid = [row for row in values if row["covered"]]
        categorical = [row for row in values if row["metric"] not in ["ordinal", "preference_tie"]]
        ordinal = [row["ordinal_mae"] for row in valid if row["ordinal_mae"] is not None]
        return {"count": len(values), "prompt_groups": len({row["group"] for row in values}),
                "evaluation_clusters": len({row["workflow_group"] if row["source"] == "synthetic_rules" else row["group"] for row in values}),
                "workflow_groups": len({row["workflow_group"] for row in values}),
                "classification_count": len(categorical),
                "accuracy": float(np.mean([row["credit"] for row in categorical])) if categorical else None,
                "soft_label_top_choice_credit": float(np.mean([row["credit"] for row in values])),
                "coverage": len(valid) / len(values),
                "nll_covered": float(np.mean([row["nll"] for row in valid])) if valid else None,
                "brier_covered": float(np.mean([row["brier"] for row in valid])) if valid else None,
                "ordinal_mae_covered": float(np.mean(ordinal)) if ordinal else None}
    family_results = {family: aggregate(values) for family, values in by_family.items()}
    fresh = {family: [row for row in values if row["scope"] != "retention_exposed" and row["metric"] not in ["ordinal", "preference_tie"]] for family, values in by_family.items()}
    fresh = {family: values for family, values in fresh.items() if values}
    scope_results = {}
    for scope in sorted({row["scope"] for row in outcomes}):
        groups = defaultdict(list)
        for row in outcomes:
            if row["scope"] == scope: groups[row["family"]].append(row)
        summaries = [aggregate(values) for values in groups.values()]
        classification = [value["accuracy"] for value in summaries if value["accuracy"] is not None]
        scope_results[scope] = {"count": sum(map(len, groups.values())),
            "classification_count": sum(value["classification_count"] for value in summaries),
            "macro_accuracy": float(np.mean(classification)) if classification else None,
            "macro_coverage": float(np.mean([value["coverage"] for value in summaries]))}
    ties = [row for row in outcomes if row["metric"] == "preference_tie"]
    result = {"by_family": family_results, "by_scope": scope_results,
              "fresh_family_macro": float(np.mean([aggregate(values)["accuracy"] for values in fresh.values()])) if fresh else None,
              "fresh_family_nll": float(np.mean([aggregate(values)["nll_covered"] for values in fresh.values() if any(row['covered'] for row in values)])) if fresh else None,
              "ordinal_mae_by_family": {name: values["ordinal_mae_covered"] for name, values in family_results.items() if values["ordinal_mae_covered"] is not None},
              "preference_ties": aggregate(ties) if ties else None,
              "coverage": float(np.mean([row["covered"] for row in outcomes])),
              "classification_macro_policy": "Equal fresh categorical families; ordinal ratings and preference ties excluded. Ordinal MAE and tie proper scores reported separately.",
              "tie_policy": "Expected target credit, averaging exactly tied predicted maxima; preference ties retain soft targets and are excluded from primary accuracy."}
    return result, outcomes


def load_rows(model):
    # Deliberate allowlist: final.jsonl/calibration.jsonl are never opened here.
    train = list(read_jsonl(DATA_ROOT / "train.jsonl"))
    all_development = list(read_jsonl(DATA_ROOT / "development.jsonl"))
    development = (all_development if os.environ.get('V3_FULL_DEVELOPMENT') == '1'
                   else select_development(all_development))
    excluded = Counter()
    for row in train + development:
        row["rendered_state"] = schema_state(row["state"], row.get("question", ""))
        target_distribution(row)
        if not 2 <= len(row["candidates"]) <= 64 or len(set(row["candidates"])) != len(row["candidates"]):
            raise ValueError(f"Invalid candidates for {row['id']}")
        row["lengths"] = [len(model.ids(text)) for text in [row["rendered_state"], *row["candidates"]]]
        row["eligible"] = row["lengths"][0] <= MAX_STATE and max(row["lengths"][1:]) <= MAX_ACTION
    eligible = []
    for row in train:
        if row["eligible"]: eligible.append(row)
        else: excluded[row["source"]] += 1
    pools = defaultdict(lambda: defaultdict(list))
    for row in eligible: pools[row.get("category", row.get("kind", "decision"))][row["source"]].append(row)
    for category in ["decision", "qa", "agent"]:
        if category not in pools: raise RuntimeError(f"Missing training category {category}")
        for source in pools[category]: pools[category][source].sort(key=lambda row: row["id"])
    rng = random.Random(SEED)
    schedule = []
    for _ in range(STEPS):
        batch = []
        for category, count in [("decision", 4), ("qa", 2), ("agent", 2)]:
            sources = sorted(pools[category])
            for _ in range(count):
                source = rng.choice(sources); batch.append(rng.choice(pools[category][source]))
        rng.shuffle(batch); schedule.append(batch)
    identities = [row["id"] for row in development]
    manifest = {"seed": SEED, "requested_steps": STEPS, "batch_size": BATCH,
        "development_population": "full" if os.environ.get('V3_FULL_DEVELOPMENT') == '1' else "pilot_subset",
        "mixture": {"decision": .5, "qa": .25, "agent": .25}, "source_sampling": "Uniform sources within category; with replacement",
        "training_rows": len(eligible), "training_length_exclusions": dict(excluded),
        "development_ids": identities, "development_ids_sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
        "training_data_sha256": hashlib.sha256((DATA_ROOT / "train.jsonl").read_bytes()).hexdigest(),
        "development_data_sha256": hashlib.sha256((DATA_ROOT / "development.jsonl").read_bytes()).hexdigest(),
        "schedule_sha256": digest([[row["id"] for row in batch] for batch in schedule]),
        "scheduled_unique_rows": len({row["id"] for batch in schedule for row in batch}),
        "scheduled_source_counts": dict(Counter(row["source"] for batch in schedule for row in batch)),
        "initial_head_sha256": hashlib.sha256((V2_ROOT / "train/hard.safetensors").read_bytes()).hexdigest(),
        "limits": {"state": MAX_STATE, "candidate": MAX_ACTION}, "test_labels_opened": False}
    manifest_name = os.environ.get('V3_OUTPUT_NAME', 'pilot') + '-data.json'
    save(ROOT / manifest_name, manifest)
    return schedule, development, manifest


def vector_cache(model, rows, deadline):
    directory = ROOT / "base-cache"; directory.mkdir(parents=True, exist_ok=True)
    index_path, vector_path = directory / "index.json", directory / "vectors.npy"
    vectors = {}
    if index_path.exists() and vector_path.exists():
        index = json.loads(index_path.read_text()); array = np.load(vector_path)
        if len(index) != len(array): raise RuntimeError("Embedding cache index mismatch")
        vectors.update(zip(index, array))
    texts = sorted({text for row in rows if row["eligible"] for text in [row["rendered_state"], *row["candidates"]]} - set(vectors), key=lambda text: len(model.ids(text)))
    start = time.monotonic(); latest = start
    model.eval()
    for offset in range(0, len(texts), 32):
        if time.monotonic() > deadline - 60: break
        chunk = texts[offset:offset + 32]
        with torch.no_grad(): encoded = model.encode(chunk).cpu().numpy()
        vectors.update(zip(chunk, encoded))
        if time.monotonic() - latest > 25:
            print(f"Base cache added {offset + len(chunk)}/{len(texts)} texts; {time.monotonic()-start:.0f}s", flush=True)
            latest = time.monotonic()
    np.save(vector_path, np.stack(list(vectors.values())))
    save(index_path, list(vectors))
    missing = len(set(texts) - set(vectors))
    if missing: return None, {"status": "partial_cache", "missing_texts": missing}
    return {text: torch.tensor(value, device="cuda") for text, value in vectors.items()}, None


@torch.no_grad()
def evaluate_model(model, rows, vectors=None):
    model.eval(); values = []
    for offset in range(0, len(rows), 4):
        block = rows[offset:offset + 4]
        eligible = [row for row in block if row["eligible"]]
        predictions = model.logits_batch(eligible, vectors=vectors) if eligible else []
        iterator = iter(predictions)
        values.extend(next(iterator).float().cpu().tolist() if row["eligible"] else None for row in block)
    summary, outcomes = score_predictions(rows, values)
    return summary, values, outcomes


def likelihood(model, rows, deadline, output):
    # Same declared conditional token-likelihood baseline as v2. No head or adapter.
    scores = json.loads(output.read_text()) if output.exists() else []
    with torch.no_grad():
        for index in range(len(scores), len(rows)):
            if time.monotonic() > deadline - 45: break
            row = rows[index]
            if not row["eligible"]: scores.append(None); continue
            prefix = model.tokenizer(row["rendered_state"] + "\n\nCandidate response:\n", add_special_tokens=True)["input_ids"]
            values = []
            for candidate in row["candidates"]:
                action = model.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                inputs = torch.tensor([prefix + action], device="cuda")
                h = model.encoder(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).last_hidden_state[0]
                h = h[len(prefix)-1:len(prefix)+len(action)-1]
                total = 0.
                for at in range(0, len(action), 16):
                    z = F.linear(h[at:at+16], model.encoder.get_input_embeddings().weight)
                    cap = getattr(model.encoder.config, "final_logit_softcapping", None)
                    if cap is not None: z = (z / cap).tanh() * cap
                    z = z.float(); labels = torch.tensor(action[at:at+16], device="cuda")
                    total += float((z[torch.arange(len(labels), device="cuda"), labels] - torch.logsumexp(z, -1)).sum())
                values.append(total / len(action))
            scores.append(values)
            if (index+1) % 24 == 0:
                save(output, scores); print(f"Gemma likelihood {index+1}/{len(rows)}", flush=True)
    save(output, scores)
    return scores


def baselines(deadline):
    output = ROOT / "baselines"; output.mkdir(exist_ok=True, parents=True)
    model = DecisionModel(BASE, V2_ROOT / "train/hard.safetensors")
    _, development, manifest = load_rows(model)
    provenance_path = output / "likelihood-provenance.json"
    provenance = {key: manifest[key] for key in ["development_ids_sha256", "development_data_sha256"]}
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise RuntimeError("Likelihood baseline provenance changed; refusing stale predictions")
    if (output / "likelihood-progress.json").exists() and not provenance_path.exists():
        raise RuntimeError("Likelihood progress is missing its provenance")
    save(provenance_path, provenance)
    vectors, partial = vector_cache(model, development, deadline)
    if partial: return partial
    results = {}; predictions = {}
    for name, checkpoint in [("v2_hard_initialization", "hard"), ("v2_unchanged", "agent")]:
        model.heads.load_state_dict(load_file(str(V2_ROOT / f"train/{checkpoint}.safetensors"), device="cuda"))
        summary, values, _ = evaluate_model(model, development, vectors)
        results[name] = summary; predictions[name] = values
    values = likelihood(model, development, deadline, output / "likelihood-progress.json")
    if len(values) != len(development):
        save(output / "head-metrics.json", results)
        return {"status": "partial_baseline", "likelihood_completed": len(values), "total": len(development)}
    results["gemma_likelihood"], _ = score_predictions(development, values); predictions["gemma_likelihood"] = values
    save(output / "predictions.json", {"ids": manifest["development_ids"], "logits": predictions})
    report = {"status": "complete", "development_ids_sha256": manifest["development_ids_sha256"], "systems": results,
              "selection_data_only": True, "final_labels_opened": False}
    save(output / "metrics.json", report)
    return report


def objective(logits, rows):
    terms = []
    for values, row in zip(logits, rows):
        target = torch.tensor(target_distribution(row), dtype=torch.float32, device=values.device)
        # Cross entropy is KL + fixed target entropy, and retains preference ties.
        terms.append(-(target * F.log_softmax(values.float(), dim=-1)).sum())
    return torch.stack(terms).mean()


def checkpoint(model, optimizer, output, step, history, best, manifest, checks):
    destination = output / f"step-{step:05d}"
    model.save(destination)
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, destination / "optimizer.pt")
    progress = {"step": step, "checkpoint": str(destination), "best": best, "history": history,
                "manifest": manifest, "checks": checks, "arm": model.arm}
    save(output / "progress.json", progress)
    return progress


def train_arm(arm, deadline):
    output_name = os.environ.get('V3_OUTPUT_NAME', arm)
    if output_name not in {'frozen','lora','frozen_extension'}: raise ValueError('Unknown training output')
    output = ROOT / output_name; output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    progress = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else None
    path = Path(progress["checkpoint"]) if progress else None
    model = DecisionModel(BASE, path / "heads.safetensors" if path else V2_ROOT / "train/hard.safetensors", arm,
                          path / "adapter.safetensors" if path and arm == "lora" else None)
    schedule, development, manifest = load_rows(model)
    if progress:
        for key in ["training_data_sha256", "development_data_sha256", "initial_head_sha256", "development_ids_sha256"]:
            if progress["manifest"][key] != manifest[key]: raise RuntimeError(f"Resume provenance changed: {key}")
        previous_steps = progress["manifest"]["requested_steps"]
        prefix = digest([[row["id"] for row in batch] for batch in schedule[:previous_steps]])
        if STEPS < previous_steps or prefix != progress["manifest"]["schedule_sha256"]: raise RuntimeError("Training schedule changed across resume")
    vectors = None
    if arm == "frozen":
        rows = [row for batch in schedule for row in batch] + development
        vectors, partial = vector_cache(model, rows, deadline)
        if partial: return partial
    params = [{"params": list(model.heads.parameters()), "lr": .0003}]
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    if arm == "lora": params.append({"params": encoder_params, "lr": .00005})
    optimizer = torch.optim.AdamW(params, weight_decay=.01)
    history = progress["history"] if progress else []
    best = progress["best"] if progress else None
    checks = progress["checks"] if progress else {}
    start_step = progress["step"] if progress else 0
    if progress:
        state = torch.load(path / "optimizer.pt", weights_only=True, map_location="cuda")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    if not history:
        initial, values, _ = evaluate_model(model, development, vectors)
        history.append({"step": 0, "development": initial})
        best = {"step": 0, "score": initial["fresh_family_macro"], "nll": initial["fresh_family_nll"]}
        model.save(output / "best")
        save(output / "initial-predictions.json", values)
        save(output / "best-predictions.json", values)
    started = time.monotonic(); starting_tokens = model.tokens_processed
    head_anchor_name, head_anchor_parameter = next((name, parameter) for name, parameter in model.heads.named_parameters() if parameter.ndim == 2)
    anchor = head_anchor_parameter.detach().clone()
    base_parameter = next(parameter for name, parameter in model.encoder.named_parameters() if not parameter.requires_grad)
    base_anchor = base_parameter.detach().flatten()[:256].clone()
    last_saved = start_step; losses = []; timing = []; step = start_step; training_encoder_tokens = 0
    for step_index in range(start_step, STEPS):
        # Reserve time for one development run, checkpoint, and independent reload.
        if time.monotonic() > deadline - 150: break
        model.train(); optimizer.zero_grad(set_to_none=True)
        warmup = min(1., (step_index + 1) / 20)
        for group, peak in zip(optimizer.param_groups, [.0003, .00005]): group["lr"] = peak * warmup
        batch = schedule[step_index]; total_loss = 0.; tick = time.monotonic(); token_tick = model.tokens_processed
        for at in range(0, BATCH, 2):
            micro = batch[at:at + 2]
            values = model.logits_batch(micro, vectors=vectors, gradient=arm == "lora")
            loss = objective(values, micro) * len(micro) / BATCH
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            loss.backward(); total_loss += float(loss.detach())
        all_trainable = list(model.heads.parameters()) + encoder_params
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in all_trainable):
            raise RuntimeError("Non-finite training gradients")
        if step_index == 0:
            checks["head_gradient_norm"] = float(sum((parameter.grad.float().square().sum() for parameter in model.heads.parameters() if parameter.grad is not None)).sqrt())
            checks["encoder_gradient_norm"] = float(sum((parameter.grad.float().square().sum() for parameter in encoder_params if parameter.grad is not None), torch.tensor(0., device="cuda")).sqrt())
            if checks["head_gradient_norm"] <= 0 or (arm == "lora" and checks["encoder_gradient_norm"] <= 0):
                raise RuntimeError("Expected trainable module has no gradient")
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.)
        optimizer.step(); step = step_index + 1
        losses.append(total_loss); timing.append(time.monotonic() - tick)
        training_encoder_tokens += model.tokens_processed - token_tick
        if step % 10 == 0:
            print(f"V3 {arm}: step {step}/{STEPS}; loss {np.mean(losses[-10:]):.4f}; {np.mean(timing[-10:]):.2f}s/step; {torch.cuda.max_memory_allocated()/2**30:.2f}GiB peak", flush=True)
        if step % 50 == 0 or step == STEPS:
            summary, values, _ = evaluate_model(model, development, vectors)
            history.append({"step": step, "loss": float(np.mean(losses[-50:])), "development": summary})
            score, nll = summary["fresh_family_macro"], summary["fresh_family_nll"]
            if (score, -nll) > (best["score"], -best["nll"]):
                best = {"step": step, "score": score, "nll": nll}; model.save(output / "best")
                save(output / "best-predictions.json", values)
            checkpoint(model, optimizer, output, step, history, best, manifest, checks); last_saved = step
    if step != last_saved or not (output / "progress.json").exists():
        # Save every completed update even when the deadline ends a partial block.
        summary, values, _ = evaluate_model(model, development, vectors)
        history.append({"step": step, "loss": float(np.mean(losses)) if losses else None, "development": summary})
        if (summary["fresh_family_macro"], -summary["fresh_family_nll"]) > (best["score"], -best["nll"]):
            best = {"step": step, "score": summary["fresh_family_macro"], "nll": summary["fresh_family_nll"]}
            model.save(output / "best"); save(output / "best-predictions.json", values)
        checkpoint(model, optimizer, output, step, history, best, manifest, checks)
    checks["base_parameter_sample_unchanged"] = bool(torch.equal(base_parameter.detach().flatten()[:256], base_anchor))
    checks["heads_changed_this_invocation"] = bool(not torch.equal(dict(model.heads.named_parameters())[head_anchor_name].detach(), anchor)) if step > start_step else None
    checks["trainable_encoder_parameters"] = sum(parameter.numel() for parameter in encoder_params)
    checks["trainable_head_parameters"] = sum(parameter.numel() for parameter in model.heads.parameters())
    if arm == "lora":
        checks["adapter_B_nonzero"] = any(bool(value.abs().max() > 0) for name, value in adapter_state(model.encoder).items() if "lora_B" in name)
        if step and not checks["adapter_B_nonzero"]: raise RuntimeError("LoRA B parameters did not update")
    if not checks["base_parameter_sample_unchanged"]: raise RuntimeError("Frozen base parameter changed")
    if step > start_step and not checks["heads_changed_this_invocation"]: raise RuntimeError("Heads did not update")
    report = {"status": "complete" if step == STEPS else "partial", "arm": arm, "steps": step, "requested_steps": STEPS,
        "best": best, "history": history, "checks": checks, "manifest": manifest, "head_config": HEAD_CONFIG,
        "lora_config": LORA_CONFIG if arm == "lora" else None,
        "new_optimizer_steps": step - start_step, "measured_seconds_per_step": float(np.mean(timing)) if timing else None,
        "encoder_forward_tokens_training": training_encoder_tokens,
        "encoder_forward_tokens_per_training_second": training_encoder_tokens / sum(timing) if timing else None,
        "scheduled_training_tokens": sum(sum(row["lengths"]) for batch in schedule[start_step:step] for row in batch),
        "total_encoder_input_tokens_including_development": model.tokens_processed - starting_tokens,
        "invocation_seconds": time.monotonic() - started, "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
        "interpretation": ("Small development-only screening pilot, below one full data pass; failure to improve does not establish a model capacity limit."
                           if STEPS <= 300 else "Longer frozen-head run selected on development data; task-level results do not establish general decision expertise."),
        "final_labels_opened": False}
    # An independent object reload checks selected adapter + heads, not a live copy.
    if time.monotonic() < deadline - 35:
        model.heads.load_state_dict(load_file(str(output / "best/heads.safetensors"), device="cuda"))
        if arm == "lora":
            from model import load_adapter
            load_adapter(model.encoder, output / "best/adapter.safetensors")
        model.eval(); sample = next(row for row in development if row["eligible"])
        with torch.no_grad():
            before = model.logits(sample).cpu()
            shuffled = dict(sample, candidates=list(reversed(sample["candidates"])))
            permutation_error = float((model.logits(shuffled).flip(0).cpu() - before).abs().max())
        del optimizer, model, vectors, encoder_params
        gc.collect(); torch.cuda.empty_cache()
        restored = DecisionModel(BASE, output / "best/heads.safetensors", arm,
                                 output / "best/adapter.safetensors" if arm == "lora" else None).eval()
        with torch.no_grad(): reloaded = restored.logits(sample).cpu()
        checks["selected_reload_max_logit_error"] = float((before - reloaded).abs().max())
        checks["candidate_permutation_max_logit_error"] = permutation_error
        if checks["selected_reload_max_logit_error"] > 1e-4 or permutation_error > 1e-4:
            raise RuntimeError("Checkpoint reload or candidate permutation invariance failed")
        checks["reload_verified"] = True
    else: checks["reload_verified"] = False
    save(output / "metrics.json", report)
    return report


def compare():
    reports = {arm: json.loads((ROOT / arm / "metrics.json").read_text()) for arm in ["frozen", "lora"]}
    baseline = json.loads((ROOT / "baselines/metrics.json").read_text())
    left, right = reports["frozen"], reports["lora"]
    same = left["steps"] == right["steps"] and left["manifest"]["schedule_sha256"] == right["manifest"]["schedule_sha256"]
    if left["manifest"]["development_ids_sha256"] != right["manifest"]["development_ids_sha256"]:
        raise RuntimeError("Development populations differ")
    if baseline["development_ids_sha256"] != left["manifest"]["development_ids_sha256"]:
        raise RuntimeError("Baseline development population differs")
    reference = None
    if (ROOT / "reference/metrics.json").exists():
        reference = json.loads((ROOT / "reference/metrics.json").read_text())
        if reference.get("development_ids_sha256", reference.get("selected_ids_sha256")) != left["manifest"]["development_ids_sha256"]:
            raise RuntimeError("Reference development population differs")
    report = {"status": "complete" if same and all(value["status"] == "complete" for value in reports.values()) else "incomplete_comparison",
              "equal_training_exposure": same, "arms": reports, "baselines": baseline, "reference": reference,
              "selection_rule": "Higher equal-family fresh categorical development accuracy; categorical NLL tie-breaker. Preference ties and ordinal rows excluded from primary accuracy; retention, tie proper scores and ordinal MAE reported separately.",
              "final_labels_opened": False, "publication_on_hold": True,
              "interpretation": "This screening pilot can justify a longer run; it cannot establish JEV parity or a 270M capacity limit."}
    if same:
        report["provisional_arm"] = max(reports, key=lambda name: (reports[name]["best"]["score"], -reports[name]["best"]["nll"]))
    rows = select_development(list(read_jsonl(DATA_ROOT / "development.jsonl")))
    if hashlib.sha256(json.dumps([row["id"] for row in rows], separators=(",", ":")).encode()).hexdigest() != left["manifest"]["development_ids_sha256"]:
        raise RuntimeError("Current development rows differ from scored population")
    selected = {}
    for arm in reports:
        predictions = json.loads((ROOT / arm / "best-predictions.json").read_text())
        selected[arm], _ = score_predictions(rows, predictions)
    report["selected_checkpoint_metrics"] = selected
    report["development_deltas"] = {}
    for arm, metrics in selected.items():
        deltas = {}
        for name in ["gemma_likelihood", "v2_unchanged"]:
            base = baseline["systems"][name]
            deltas[name] = {"fresh_macro": metrics["fresh_family_macro"] - base["fresh_family_macro"],
                "by_family": {family: values["accuracy"] - base["by_family"][family]["accuracy"]
                              for family, values in metrics["by_family"].items()
                              if values["accuracy"] is not None and base["by_family"][family]["accuracy"] is not None}}
        report["development_deltas"][arm] = deltas
    save(ROOT / "compare/metrics.json", report)
    return report


def run(stage, seconds=1500):
    if not torch.cuda.is_available(): raise RuntimeError("V3 model execution is authorized on Modal GPU only")
    ROOT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + seconds
    torch.set_num_threads(2)
    if stage == "baselines": return baselines(deadline)
    if stage in ["frozen", "lora"]: return train_arm(stage, deadline)
    if stage == "compare": return compare()
    raise ValueError(stage)
