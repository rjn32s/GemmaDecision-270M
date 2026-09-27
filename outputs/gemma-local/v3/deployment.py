"""Offline CLM-style ranking with Gemma-270M and separate decision heads.

Example:
    model = GemmaDecisionRanker("./GemmaDecision-270M", device="auto")
    model.rank("The customer was charged twice", {
        "billing": "Investigate a duplicate payment",
        "technical": "Troubleshoot an application crash",
    }, question="Choose the appropriate support queue.")

Returned scores are uncalibrated ranking scores, not probabilities, confidence
estimates, or scores comparable between unrelated requests. No generation or
network access occurs during model loading or ranking.

A self-contained directory contains official Gemma config/tokenizer/weight files,
decision_heads.safetensors, decision_config.json, this module, common.py,
clm_heads.py and clm_schema.py. An unmerged LoRA package additionally contains
adapter.safetensors and explicit lora_config in decision_config.json.
"""
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from common import make_heads, schema_state


class _LoRALinear(nn.Module):
    """Inference equivalent of the training adapter; no train.py import needed."""
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features, dtype=torch.float32, device=original.weight.device), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, dtype=torch.float32, device=original.weight.device), requires_grad=False)

    def forward(self, value):
        update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return self.original(value) + (update * (self.alpha / self.rank)).to(value.dtype)


def _adapter_configuration(config):
    if not isinstance(config, dict): raise ValueError("A LoRA package requires an explicit lora_config object")
    rank = int(config["rank"]); alpha = float(config["alpha"]); targets = list(config["targets"])
    if rank < 1 or alpha <= 0 or not targets or any(not isinstance(name, str) for name in targets):
        raise ValueError("Invalid adapter configuration")
    if float(config.get("dropout", 0.)) != 0.: raise ValueError("This deployment contract requires the trained zero-dropout adapter")
    return rank, alpha, targets


def _attach_adapter(encoder, path, config):
    rank, alpha, targets = _adapter_configuration(config)
    adapted = []
    for name, module in list(encoder.named_modules()):
        if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear):
            parent, leaf = name.rsplit(".", 1)
            setattr(encoder.get_submodule(parent), leaf, _LoRALinear(module, rank, alpha))
            adapted.append(name)
    if not adapted: raise ValueError("No encoder projections match adapter targets")
    state = load_file(str(path), device="cpu")
    parameters = dict(encoder.named_parameters())
    expected = {name for name in parameters if "lora_" in name}
    if set(state) != expected: raise ValueError("Adapter tensors do not match the encoder architecture")
    with torch.no_grad():
        for name, value in state.items():
            if parameters[name].shape != value.shape: raise ValueError(f"Adapter tensor shape mismatch: {name}")
            if not torch.isfinite(value).all(): raise ValueError("Non-finite adapter weights")
            parameters[name].copy_(value.to(parameters[name].device))
    return adapted


def fold_lora_into_encoder(encoder, adapter_path, lora_config):
    """Optional packager helper: fold into a plain loaded encoder, in place.

    Intended for cloud packaging. Save the resulting encoder with save_pretrained
    and set adapter_mode='merged' in decision_config.json. BF16 rounding can make
    merged outputs differ from runtime LoRA; validate score/ranking agreement on
    development cases before choosing the merged package.
    """
    rank, alpha, targets = _adapter_configuration(lora_config)
    if any(isinstance(module, _LoRALinear) or hasattr(module, "lora_A") for module in encoder.modules()):
        raise ValueError("Fold requires a plain base encoder; an attached adapter would apply LoRA twice")
    state = load_file(str(adapter_path), device="cpu")
    modules = dict(encoder.named_modules())
    names = sorted(name[:-len(".lora_A")] for name in state if name.endswith(".lora_A"))
    expected = {key for name in names for key in [name + ".lora_A", name + ".lora_B"]}
    if not names or set(state) != expected: raise ValueError("Adapter file does not contain complete A/B pairs")
    required_names = {name for name, module in modules.items() if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets}
    if set(names) != required_names: raise ValueError("Adapter does not cover every configured projection target")
    # Validate every pair before mutating any weights.
    pairs = []
    for name in names:
        module = modules.get(name)
        a, b = state[name + ".lora_A"], state[name + ".lora_B"]
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            raise ValueError(f"Adapter target is missing or unsupported: {name}")
        if a.shape != (rank, module.in_features) or b.shape != (module.out_features, rank):
            raise ValueError(f"Adapter shape mismatch: {name}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError("Non-finite adapter weights")
        pairs.append((name, module, a, b))
    with torch.no_grad():
        for name, module, a, b in pairs:
            delta = (b.float() @ a.float()) * (alpha / rank)
            module.weight.copy_((module.weight.float() + delta.to(module.weight.device)).to(module.weight.dtype))
    return {"merged_modules": [name for name, _, _, _ in pairs], "rank": rank, "alpha": alpha,
            "requires_development_agreement_check": True}


class GemmaDecisionRanker:
    """Local rank(state, candidates, question='') with CPU, MPS or CUDA serving."""
    score_semantics = "Uncalibrated scaled cosine ranking score; higher is preferred. Not a probability or confidence estimate."

    def __init__(self, path, device="auto", candidate_cache_size=128):
        self.path = Path(path)
        self.config = json.loads((self.path / "decision_config.json").read_text())
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "mps", "cuda"}: raise ValueError("Supported devices are cpu, mps and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable on this machine")
        if self.device.type == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS is unavailable on this machine")
        # Float32 on CPU/MPS avoids assuming local BF16 support. GPU matches training.
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        self.encoder = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                                                dtype=dtype, attn_implementation="sdpa").to(self.device).eval().requires_grad_(False)
        mode = self.config.get("adapter_mode", "none")
        if mode == "runtime":
            _attach_adapter(self.encoder, self.path / self.config.get("adapter_file", "adapter.safetensors"), self.config.get("lora_config"))
        elif mode not in {"none", "merged"}: raise ValueError("adapter_mode must be none, runtime or merged")
        self.heads = make_heads(self.config["head_config"]).to(self.device).eval().requires_grad_(False)
        self.heads.load_state_dict(load_file(str(self.path / self.config.get("head_file", "decision_heads.safetensors")), device="cpu"), strict=True)
        self.max_state_tokens = int(self.config.get("max_state_tokens", 2048))
        self.max_action_tokens = int(self.config.get("max_action_tokens", 768))
        self.candidate_cache_size = max(0, int(candidate_cache_size))
        self._candidate_cache = OrderedDict()

    def clear_candidate_cache(self):
        self._candidate_cache.clear()

    def _tokens(self, text):
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def _encode(self, token_rows):
        order = sorted(range(len(token_rows)), key=lambda index: len(token_rows[index]))
        values = [None] * len(token_rows); offset = 0
        while offset < len(order):
            indices = order[offset:offset + 4]
            while len(indices) > 1 and len(indices) * max(len(token_rows[index]) for index in indices) > 4096:
                indices = indices[:-1]
            width = max(len(token_rows[index]) for index in indices)
            tokens = torch.full((len(indices), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(tokens)
            for index, original in enumerate(indices):
                ids = token_rows[original]
                tokens[index, :len(ids)] = torch.tensor(ids, device=self.device)
                mask[index, :len(ids)] = 1
            hidden = self.encoder(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
            last = mask.sum(-1) - 1
            encoded = F.normalize(hidden[torch.arange(len(indices), device=self.device), last].float(), dim=-1)
            if not torch.isfinite(encoded).all(): raise RuntimeError("Non-finite model encoding")
            for index, original in enumerate(indices): values[original] = encoded[index]
            offset += len(indices)
        return torch.stack(values)

    @torch.inference_mode()
    def rank(self, state, candidates, question=""):
        """Return descending [{candidate, text, score}], with stable input-order ties.

        candidates may be a mapping of labels to descriptions or a list of text.
        Every candidate is scored independently of candidate order. Requests that
        exceed the declared token limits raise ValueError; no text is truncated.
        """
        if isinstance(candidates, Mapping):
            labels, texts = list(candidates), list(candidates.values())
            if not all(isinstance(label, str) for label in labels): raise ValueError("Candidate labels must be strings")
        elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            texts = list(candidates); labels = [str(index) for index in range(len(texts))]
        else: raise ValueError("Candidates must be a label-to-description mapping or a sequence of strings")
        if not 2 <= len(texts) <= 64: raise ValueError("Provide 2–64 candidates")
        if not all(isinstance(text, str) and text.strip() for text in texts): raise ValueError("Candidate descriptions must be nonempty strings")
        if len(set(texts)) != len(texts): raise ValueError("Candidate descriptions must be distinct")
        rendered = schema_state(state, question)
        if not rendered.strip(): raise ValueError("Provide a nonempty state or question")
        state_tokens = self._tokens(rendered)
        if len(state_tokens) > self.max_state_tokens:
            raise ValueError(f"State has {len(state_tokens)} tokens; the declared limit is {self.max_state_tokens}")
        actions = {}; missing_texts = []; missing_tokens = []
        for text in texts:
            if text in self._candidate_cache:
                vector, length = self._candidate_cache[text]
                self._candidate_cache.move_to_end(text)
                actions[text] = vector.to(self.device)
            else:
                tokens = self._tokens(text); length = len(tokens)
                missing_texts.append(text); missing_tokens.append(tokens)
            if length > self.max_action_tokens:
                raise ValueError(f"A candidate has {length} tokens; the declared limit is {self.max_action_tokens}")
        encoded = self._encode([state_tokens, *missing_tokens])
        for text, tokens, vector in zip(missing_texts, missing_tokens, encoded[1:]):
            actions[text] = vector
            if self.candidate_cache_size:
                self._candidate_cache[text] = (vector.cpu(), len(tokens))
                while len(self._candidate_cache) > self.candidate_cache_size: self._candidate_cache.popitem(last=False)
        projected_state, projected_actions = self.heads.project(encoded[:1], torch.stack([actions[text] for text in texts]))
        scores = ((projected_state @ projected_actions.T)[0] * self.heads.scale()).float().cpu().tolist()
        if not all(math.isfinite(score) for score in scores): raise RuntimeError("Non-finite ranking score")
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [{"candidate": labels[index], "text": texts[index], "score": scores[index]} for index in order]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rank local candidate actions with GemmaDecision; scores are not probabilities.")
    parser.add_argument("--model", required=True, help="Path to the complete local model directory")
    parser.add_argument("--input", required=True, help="JSON file with state, candidates, and optional question")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    model = GemmaDecisionRanker(args.model, device=args.device)
    result = model.rank(request["state"], request["candidates"], request.get("question", ""))
    print(json.dumps({"ranking": result, "score_semantics": model.score_semantics}, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
