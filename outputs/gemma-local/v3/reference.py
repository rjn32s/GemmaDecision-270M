"""Bounded released-CLM comparison on v3 development data only.

No model, data, or network activity occurs at import. The reference consumes the
same fixed selection and scoring helpers as the two student arms. Predictions
are evidence about teacher suitability, never replacements for verified labels.
"""
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time


SEED = 2704203
MAX_STATE = 2048
MAX_ACTION = 768
MAX_REFERENCE_TOKENS = 8192
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CLM_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"


def _digest_ids(rows):
    ids = [r["id"] for r in rows]
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _head_options(checkpoint, hidden_size):
    """Interpret only the known architecture fields from the safe checkpoint."""
    cfg = checkpoint["cfg"]
    if not isinstance(cfg, dict):
        raise ValueError("CLM configuration must be a dictionary")
    opts = dict(
        hidden=cfg.get("hidden_size", 4096),
        width=cfg["width"], depth=cfg["depth"],
        proj=checkpoint.get("projection_dim", cfg.get("projection_dim", 512)),
        activation=cfg.get("activation", "gelu"),
        layernorm=cfg.get("layernorm", False), residual=cfg.get("residual", False),
    )
    for field in ("hidden", "width", "depth", "proj"):
        if type(opts[field]) is not int or opts[field] <= 0:
            raise ValueError(f"Invalid CLM architecture field: {field}")
    if opts["hidden"] != hidden_size:
        raise ValueError("CLM head width does not match Qwen hidden size")
    if opts["activation"] not in ("gelu", "relu", "silu"):
        raise ValueError("Unexpected CLM activation")
    if any(type(opts[x]) is not bool for x in ("layernorm", "residual")):
        raise ValueError("CLM architecture flags must be bools")
    return opts


def _check_checkpoint(checkpoint):
    import torch
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary CLM checkpoint")
    for name in ("state_head", "action_head"):
        weights = checkpoint[name]
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Missing {name} weights")
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Non-tensor weight in {name}.{key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite weight in {name}.{key}")
    scale = torch.as_tensor(checkpoint["logit_scale"]).float()
    if scale.numel() != 1 or not torch.isfinite(scale).all():
        raise ValueError("CLM logit scale must be one finite scalar")
    # Clamp before exponentiation to avoid overflow in malformed checkpoints.
    return scale.clamp(max=math.log(100.0)).exp().item()


def _prepare_inputs(rows, gemma_tokenizer, reference_tokenizer):
    from common import schema_state
    inputs, reasons = [], []
    for row in rows:
        candidates = row["candidates"]
        if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 64
                or any(not isinstance(x, str) or not x.strip() for x in candidates)
                or len(set(candidates)) != len(candidates)):
            inputs.append(None); reasons.append("invalid_candidates"); continue
        # No target, rating, provenance, source, or generator fields enter input.
        state = schema_state(row["state"], row["question"])
        texts = [state, *candidates]
        lengths = [len(gemma_tokenizer(t, add_special_tokens=True)["input_ids"])
                   for t in texts]
        if lengths[0] > MAX_STATE or max(lengths[1:]) > MAX_ACTION:
            inputs.append(None); reasons.append("matched_gemma_length_limit"); continue
        if any(len(reference_tokenizer(t, add_special_tokens=True)["input_ids"])
               > MAX_REFERENCE_TOKENS for t in texts):
            inputs.append(None); reasons.append("reference_internal_length_limit"); continue
        inputs.append({"state": state, "candidates": candidates})
        reasons.append(None)
    return inputs, reasons


def score_reference(rows, deadline, output):
    """Return logits in row order, plus explicit failures; never drop examples."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer
    from clm_heads import make_head
    from common import encode_texts, save

    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    qwen = Path(os.environ.get("CLM_QWEN_PATH", str(root.parent / "qwen-reference")))
    head = Path(os.environ.get("CLM_HEAD_PATH", str(root.parent / "clm-reference/CLM_v0.1-8B.pt")))
    gemma = Path(os.environ.get("GEMMA_MODEL_PATH", str(root.parent / "model")))
    if not torch.cuda.is_available():
        raise RuntimeError("CLM reference must run on the allocated cloud CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(qwen, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma, local_files_only=True)
    inputs, reasons = _prepare_inputs(rows, gemma_tokenizer, tokenizer)
    if time.monotonic() >= deadline - 60:
        raise TimeoutError("Reference time budget exhausted during input validation")
    print(f"CLM reference: {len(rows)} development cases, "
          f"{sum(x is not None for x in inputs)} eligible", flush=True)

    # weights_only=True is deliberate: no fallback to arbitrary pickle loading.
    checkpoint = torch.load(head, map_location="cpu", weights_only=True)
    scale = _check_checkpoint(checkpoint)
    model = AutoModel.from_pretrained(
        qwen, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite Qwen parameter: {name}")
    options = _head_options(checkpoint, model.config.hidden_size)
    state_head = make_head(**options).to("cuda").eval().requires_grad_(False)
    action_head = make_head(**options).to("cuda").eval().requires_grad_(False)
    state_head.load_state_dict(checkpoint["state_head"], strict=True)
    action_head.load_state_dict(checkpoint["action_head"], strict=True)
    del checkpoint

    texts = sorted({t for row in inputs if row is not None
                    for t in [row["state"], *row["candidates"]]})
    # Length sorting lowers padding cost and makes the bounded pass predictable.
    texts.sort(key=lambda t: (len(tokenizer(t, add_special_tokens=True)["input_ids"]), t))
    vectors = {}
    timed_out = False
    last_print = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            if time.monotonic() >= deadline - 30:
                timed_out = True
                break
            batch = texts[start:start + 8]
            encoded = encode_texts(model, tokenizer, batch, batch_size=8)
            if not np.isfinite(encoded).all():
                raise RuntimeError("Non-finite reference embeddings")
            vectors.update(zip(batch, encoded))
            if start == 0 or time.monotonic() - last_print >= 20:
                print(f"CLM reference embeddings {min(start + 8, len(texts))}/{len(texts)}", flush=True)
                last_print = time.monotonic()
        logits = []
        for i, row in enumerate(inputs):
            if row is None:
                logits.append(None)
                continue
            if any(t not in vectors for t in [row["state"], *row["candidates"]]):
                reasons[i] = "reference_deadline"
                logits.append(None)
                continue
            states = torch.tensor(vectors[row["state"]][None], device="cuda")
            actions = torch.tensor(np.stack([vectors[t] for t in row["candidates"]]), device="cuda")
            projected_s = F.normalize(state_head(states), dim=-1)
            projected_a = F.normalize(action_head(actions), dim=-1)
            scores = (projected_s @ projected_a.T)[0] * scale
            if not torch.isfinite(scores).all():
                raise RuntimeError("Non-finite CLM projected logits")
            logits.append(scores.cpu().tolist())

    info = {
        "model": "Contrastive-LM/CLM-v0.1-8B", "head_config": options,
        "qwen_revision": QWEN_REVISION, "clm_revision": CLM_REVISION,
        "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
        "logit_scale": scale, "deadline_reached": timed_out,
        "unique_texts": len(texts), "encoded_texts": len(vectors),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "backend": "Transformers BF16 SDPA; L2-normalized last non-padding token; raw text, no chat template",
        "backend_caveat": "Released architecture and weights; numerical parity with vLLM is not independently certified.",
        "length_policy": {"matched_gemma_state": MAX_STATE, "matched_gemma_candidate": MAX_ACTION,
                          "qwen_internal_limit": MAX_REFERENCE_TOKENS, "truncation": False},
        "failure_counts": dict(Counter(x for x in reasons if x is not None)),
    }
    save(output / "reference-info.json", info)
    del model, state_head, action_head, vectors
    gc.collect(); torch.cuda.empty_cache()
    return logits, reasons, info


def run(seconds=900):
    """Public Modal entry point. This function never reads calibration/final data."""
    from common import read_jsonl, save
    from train import select_development, score_predictions

    if seconds < 90:
        raise ValueError("Reference needs at least a 90-second bounded allocation")
    started = time.monotonic()
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    output = root / "reference"
    output.mkdir(parents=True, exist_ok=True)
    development_path = root / "data/development.jsonl"
    rows = select_development(list(read_jsonl(development_path)), cap=24)
    if not rows:
        raise ValueError("No development cases available for CLM comparison")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate selected development IDs")
    protocol = {
        "development_only": True, "seed": SEED, "cap_per_family": 24,
        "max_rows_per_group": 2, "selected_ids": [r["id"] for r in rows],
        "selected_ids_sha256": _digest_ids(rows),
        "development_file_sha256": hashlib.sha256(development_path.read_bytes()).hexdigest(),
        "selection": "Shared train.select_development; SHA256('2704203|'+id), label independent",
        "no_final_or_calibration_access": True,
        "teacher_policy": "Independent labels are authoritative. No automatic pseudo-labeling or teacher promotion.",
    }
    save(output / "protocol.json", protocol)
    logits, reasons, info = score_reference(rows, started + seconds, output)
    metrics, outcomes = score_predictions(rows, logits)
    predictions = [{"id": r["id"], "group": r["group"], "family": r["family"],
                    "source": r["source"], "evaluation_scope": r.get("evaluation_scope"),
                    "logits": z, "failure_reason": reason}
                   for r, z, reason in zip(rows, logits, reasons)]
    save(output / "predictions.json", predictions)
    save(output / "outcomes.json", outcomes)
    report = {
        "status": "partial" if info["deadline_reached"] else "complete",
        "development_only": True, "selected_ids_sha256": protocol["selected_ids_sha256"],
        "results": metrics, "reference": info,
        "teacher_eligible": None,
        "teacher_eligibility_reason": "Pending paired comparison with student arms on these independent labels; confidence alone is insufficient.",
        "function_seconds": time.monotonic() - started,
    }
    save(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(run(args.seconds), indent=2))
