"""Shared training/inference contract. No network or model loading at import."""
import hashlib
import json
import math
from pathlib import Path
import unicodedata

SEED = 2704202
FORMAT = "gemmadecision-clm-v2"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CLM_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
HEAD_CONFIG = dict(hidden=640, width=1024, depth=3, proj=512,
                   activation="gelu", layernorm=True, residual=False)
MAX_STATE, MAX_ACTION = 2048, 768


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def make_heads(config=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from clm_heads import make_head
    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config or HEAD_CONFIG
            self.state_head = make_head(**self.config)
            self.action_head = make_head(**self.config)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
        def project(self, s, a):
            # The upstream serving encoder returns L2-normalized last-token vectors.
            return (F.normalize(self.state_head(F.normalize(s.float(), dim=-1)), dim=-1),
                    F.normalize(self.action_head(F.normalize(a.float(), dim=-1)), dim=-1))
        def scale(self):
            return self.logit_scale.exp().clamp(max=100.)
    return Heads()


def split_group(group):
    value = int(sha(group)[:12], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "calibration" if value < 95 else "test"


def schema_state(state, instructions):
    from clm_schema import state_text
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
            if isinstance(parsed, (dict, list)): state = parsed
        except ValueError: pass
    return state_text(state, instructions)


def encode_texts(model, tokenizer, texts, batch_size=32):
    import torch
    from torch.nn import functional as F
    import numpy as np
    output = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    # HF tokenization is used for both models; raw text, no chat template or EOS.
    lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            batch = tokenizer([texts[i] for i in ids], padding=True, return_tensors="pt").to(model.device)
            h = model(**batch, use_cache=False).last_hidden_state
            ix = batch["attention_mask"].sum(-1)-1
            v = F.normalize(h[torch.arange(len(ids), device=model.device), ix].float(), dim=-1)
            if not torch.isfinite(v).all(): raise RuntimeError("Non-finite encoder output")
            output[ids] = v.cpu().numpy()
    return output
