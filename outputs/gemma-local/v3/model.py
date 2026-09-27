"""GemmaDecision v3 encoder/head contract and explicit low-rank adapters.

No model is loaded at import. LoRA modifies q_proj and v_proj only; the original
weights stay frozen. Last-token pooling and L2 normalization match v2 serving.
"""
import contextlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from common import make_heads, schema_state

LORA_CONFIG = {"rank": 8, "alpha": 16, "targets": ["q_proj", "v_proj"], "dropout": 0.0}


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        self.original = original.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features, device=original.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=original.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** .5)
    def forward(self, value):
        update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return self.original(value) + (update * (self.alpha / self.rank)).to(value.dtype)


def install_lora(encoder, config=None):
    config = config or LORA_CONFIG
    replaced = []
    for name, module in list(encoder.named_modules()):
        if name.rsplit(".", 1)[-1] in config["targets"] and isinstance(module, nn.Linear):
            parent_path, leaf = name.rsplit(".", 1)
            setattr(encoder.get_submodule(parent_path), leaf,
                    LoRALinear(module, config["rank"], config["alpha"]))
            replaced.append(name)
    if not replaced: raise RuntimeError("No encoder projection modules matched the LoRA targets")
    return replaced


def adapter_state(encoder):
    return {name: value.detach().contiguous().cpu() for name, value in encoder.named_parameters() if "lora_" in name}


def load_adapter(encoder, path):
    tensors = load_file(str(path), device="cuda")
    parameters = dict(encoder.named_parameters())
    expected = {name for name in parameters if "lora_" in name}
    if set(tensors) != expected: raise RuntimeError("Adapter tensor names do not match the configured architecture")
    with torch.no_grad():
        for name, value in tensors.items(): parameters[name].copy_(value)


class DecisionModel:
    def __init__(self, base, initialization, arm="frozen", adapter=None):
        self.arm = arm
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.encoder = AutoModel.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval().requires_grad_(False)
        self.heads = make_heads().to("cuda")
        self.heads.load_state_dict(load_file(str(initialization), device="cuda"))
        self.lora_modules = []
        if arm == "lora":
            self.lora_modules = install_lora(self.encoder)
            if adapter: load_adapter(self.encoder, adapter)
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif arm != "frozen": raise ValueError(arm)
        self.token_cache = {}
        self.tokens_processed = 0

    def ids(self, text):
        if text not in self.token_cache:
            self.token_cache[text] = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        return self.token_cache[text]

    def train(self):
        self.heads.train()
        self.encoder.train(self.arm == "lora")
        return self

    def eval(self):
        self.heads.eval(); self.encoder.eval()
        return self

    def encode(self, texts, gradient=False):
        """Right-padded last nonpadding token; no truncation, raw text, BOS only."""
        order = sorted(range(len(texts)), key=lambda index: len(self.ids(texts[index])))
        output = [None] * len(texts)
        at = 0
        context = contextlib.nullcontext if gradient else torch.no_grad
        with context():
            while at < len(order):
                maximum = len(self.ids(texts[order[at]]))
                size = min(8 if not gradient else 4, max(1, 4096 // max(maximum, 1)))
                indices = order[at:at + size]
                # Recalculate after grouping because the last sequence can be longer.
                while len(indices) > 1 and len(indices) * max(len(self.ids(texts[i])) for i in indices) > 4096:
                    indices = indices[:-1]
                width = max(len(self.ids(texts[i])) for i in indices)
                batch = torch.full((len(indices), width), self.tokenizer.pad_token_id, dtype=torch.long, device="cuda")
                mask = torch.zeros_like(batch)
                for row, index in enumerate(indices):
                    ids = self.ids(texts[index]); batch[row, :len(ids)] = torch.tensor(ids, device="cuda")
                    mask[row, :len(ids)] = 1
                hidden = self.encoder(input_ids=batch, attention_mask=mask, use_cache=False).last_hidden_state
                last = mask.sum(-1) - 1
                vectors = F.normalize(hidden[torch.arange(len(indices), device="cuda"), last].float(), dim=-1)
                if not torch.isfinite(vectors).all(): raise RuntimeError("Non-finite encoder output")
                for row, index in enumerate(indices): output[index] = vectors[row]
                self.tokens_processed += int(mask.sum())
                at += len(indices)
        return torch.stack(output)

    def logits(self, row, vectors=None, gradient=False):
        texts = [row["rendered_state"], *row["candidates"]]
        encoded = self.encode(texts, gradient=gradient) if vectors is None else torch.stack([vectors[text] for text in texts])
        state, actions = self.heads.project(encoded[:1], encoded[1:])
        return (state @ actions.T)[0] * self.heads.scale()

    def logits_batch(self, rows, vectors=None, gradient=False):
        if vectors is None:
            texts = list(dict.fromkeys(text for row in rows for text in [row["rendered_state"], *row["candidates"]]))
            values = self.encode(texts, gradient=gradient)
            vectors = dict(zip(texts, values))
        return [self.logits(row, vectors=vectors) for row in rows]

    def save(self, destination):
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().contiguous().cpu() for name, value in self.heads.state_dict().items()}, str(destination / "heads.safetensors"))
        if self.arm == "lora": save_file(adapter_state(self.encoder), str(destination / "adapter.safetensors"))
