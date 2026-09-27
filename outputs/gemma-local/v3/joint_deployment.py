"""Offline scalar joint-encoder ranking with Gemma-270M.

Load a complete local directory containing the official Gemma encoder/tokenizer,
joint_config.json, joint_head.safetensors, common.py and clm_schema.py. Each
candidate is encoded with its state in a separate forward pass. Returned scores
are raw, uncalibrated ranking scores, not probabilities or confidence estimates.
No model is loaded at import and no network access is used.
"""
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from common import schema_state


def build_joint_text(state, question, candidate):
    """Exact training format; candidate text is preserved without stripping."""
    return schema_state(state, question) + "\n\nCandidate action:\n" + candidate


class GemmaJointRanker:
    """Local rank(state, candidates, question='') on CPU, MPS or CUDA."""

    score_semantics = (
        "Uncalibrated scalar ranking score; higher is preferred. "
        "Not a probability, confidence estimate or score comparable across requests."
    )

    def __init__(self, path, device="auto"):
        self.path = Path(path)
        if not self.path.is_dir():
            raise ValueError("Model path must be a complete local directory")
        self.config = json.loads((self.path / "joint_config.json").read_text())
        self.max_state_tokens = int(self.config.get("max_state_tokens", 2048))
        self.max_action_tokens = int(self.config.get("max_action_tokens", 768))
        head_config = self.config.get("head_config", {"hidden": 640, "width": 512})
        hidden, width = int(head_config["hidden"]), int(head_config["width"])
        if min(self.max_state_tokens, self.max_action_tokens, hidden, width) < 1:
            raise ValueError("Token limits and head dimensions must be positive")
        head_file = self.config.get("head_file", "joint_head.safetensors")
        if not isinstance(head_file, str) or Path(head_file).name != head_file:
            raise ValueError("head_file must name a file in the local model directory")

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "mps", "cuda"}:
            raise ValueError("Supported devices are cpu, mps and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable on this machine")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this machine")
        dtype = (torch.bfloat16 if self.device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="sdpa",
        ).to(self.device).eval().requires_grad_(False)
        if self.encoder.config.hidden_size != hidden:
            raise ValueError("Scalar head width does not match the encoder hidden size")
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
            nn.Linear(width, 1),
        ).to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        weights = load_file(str(self.path / head_file), device="cpu")
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Non-finite scalar head weights")
        self.head.load_state_dict(weights, strict=True)

    def _tokens(self, text):
        return self.tokenizer(text, add_special_tokens=True,
                              truncation=False, padding=False)["input_ids"]

    def _score(self, tokens):
        # A single pair per forward pass makes candidate order and list size
        # irrelevant to the encoder input, padding, and numerical batch shape.
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        hidden = self.encoder(input_ids=ids, attention_mask=mask,
                              use_cache=False).last_hidden_state
        last = mask.sum(-1) - 1
        vector = F.normalize(hidden[torch.arange(len(ids), device=self.device),
                                    last].float(), dim=-1)
        if not torch.isfinite(vector).all():
            raise RuntimeError("Non-finite model encoding")
        score = float(self.head(vector).reshape(-1)[0].item())
        if not math.isfinite(score):
            raise RuntimeError("Non-finite ranking score")
        return score

    @torch.inference_mode()
    def rank(self, state, candidates, question=""):
        """Return descending [{candidate, text, score}], retaining input-order ties.

        Candidates may be a label-to-description mapping or a sequence of text
        (sequence labels are string indices). State and candidate limits are
        checked separately, including tokenizer special tokens. Every combined
        pair is encoded in full; text is never truncated. Scores do not receive
        a softmax or depend on any other candidate in the request.
        """
        if isinstance(candidates, Mapping):
            labels, texts = list(candidates), list(candidates.values())
            if not all(isinstance(label, str) for label in labels):
                raise ValueError("Candidate labels must be strings")
        elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            texts = list(candidates)
            labels = [str(index) for index in range(len(texts))]
        else:
            raise ValueError("Candidates must be a mapping or a sequence of strings")
        if not 2 <= len(texts) <= 64:
            raise ValueError("Provide 2–64 candidates")
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Candidate descriptions must be nonempty strings")
        if len(set(texts)) != len(texts):
            raise ValueError("Candidate descriptions must be distinct")
        rendered = schema_state(state, question)
        if not rendered.strip():
            raise ValueError("Provide a nonempty state or question")
        state_length = len(self._tokens(rendered))
        if state_length > self.max_state_tokens:
            raise ValueError(f"State has {state_length} tokens; the limit is {self.max_state_tokens}")
        pairs = []
        context_limit = getattr(self.encoder.config, "max_position_embeddings", None)
        # Validate the complete request before performing any encoder work.
        for text in texts:
            action_length = len(self._tokens(text))
            if action_length > self.max_action_tokens:
                raise ValueError(f"A candidate has {action_length} tokens; the limit is {self.max_action_tokens}")
            tokens = self._tokens(rendered + "\n\nCandidate action:\n" + text)
            if context_limit is not None and len(tokens) > context_limit:
                raise ValueError(f"A joint input has {len(tokens)} tokens; the encoder limit is {context_limit}")
            if not tokens:
                raise ValueError("A joint input encoded to an empty token sequence")
            pairs.append(tokens)
        scores = [self._score(tokens) for tokens in pairs]
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [{"candidate": labels[index], "text": texts[index], "score": scores[index]}
                for index in order]


# Keeps callers of the existing offline package class compatible.
GemmaDecisionRanker = GemmaJointRanker


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rank local candidates with a scalar joint Gemma encoder.")
    parser.add_argument("--model", required=True, help="Complete local model directory")
    parser.add_argument("--input", required=True, help="JSON file with state, candidates and optional question")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    model = GemmaJointRanker(args.model, device=args.device)
    result = model.rank(request["state"], request["candidates"], request.get("question", ""))
    print(json.dumps({"ranking": result, "score_semantics": model.score_semantics},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
