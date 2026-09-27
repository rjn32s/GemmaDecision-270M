"""Portable GemmaDecision inference. Candidate scoring, not text generation."""
from pathlib import Path
import json
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel,AutoTokenizer
from common import make_heads, encode_texts, schema_state
from clm_schema import candidates,answer_from_probs


class GemmaDecision:
    def __init__(self,path,device="cpu"):
        self.path=Path(path)
        self.config=json.loads((self.path/"decision_config.json").read_text())
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.tokenizer.padding_side="right"
        self.encoder=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16 if device=="cuda" else torch.float32,
                                               attn_implementation="sdpa").to(device).eval().requires_grad_(False)
        self.heads=make_heads(self.config["head_config"]).to(device).eval().requires_grad_(False)
        self.heads.load_state_dict(load_file(str(self.path/"decision_heads.safetensors"),device=device))

    @classmethod
    def from_pretrained(cls,repo_id="rajan2k/GemmaDecision-270M",revision=None,device="cpu"):
        from huggingface_hub import snapshot_download
        path=repo_id if Path(repo_id).is_dir() else snapshot_download(repo_id,revision=revision,
              allow_patterns=["*.json","*.safetensors","tokenizer.model","*.txt"])
        return cls(path,device)

    @torch.inference_mode()
    def rank(self,state,question,options):
        if not isinstance(options,dict) or not 2<=len(options)<=64:raise ValueError("Supply 2–64 named candidate descriptions")
        labels=list(options);texts=[str(options[k]) for k in labels]
        if any(not t.strip() for t in texts) or len(set(texts))!=len(texts):raise ValueError("Candidate descriptions must be distinct nonempty text")
        state=schema_state(state,question)
        lengths=[len(self.tokenizer(t)["input_ids"]) for t in [state,*texts]]
        if lengths[0]>self.config["max_state_tokens"] or max(lengths[1:])>self.config["max_action_tokens"]:
            raise ValueError("Input exceeds the documented token limit; inference does not silently truncate")
        vectors=encode_texts(self.encoder,self.tokenizer,[state,*texts],batch_size=8)
        s,a=self.heads.project(torch.tensor(vectors[:1],device=self.device),torch.tensor(vectors[1:],device=self.device))
        logits=(s@a.T)[0]*self.heads.scale()/self.config["temperature"]
        p=logits.softmax(-1).cpu().tolist()
        return dict(zip(labels,p))

    def decide(self,request):
        answers={}
        for name,q in request["questions"].items():
            keys,texts=candidates(q)
            p=self.rank(request["state"],q.get("instructions",""),dict(zip(keys,texts)))
            answers[name]=answer_from_probs(q,keys,[p[k] for k in keys])
        return {"model":"GemmaDecision-270M","answers":answers,
                "probability_status":"Calibrated on the documented development mixture; new-domain calibration is not established."}
