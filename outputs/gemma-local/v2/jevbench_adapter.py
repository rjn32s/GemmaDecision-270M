"""JevBench in-process adapter; never reads task.expected or provenance."""
import time
from decision_model import GemmaDecision
from clm_schema import candidates
from common import schema_state


class GemmaDecisionAdapter:
    name="gemmadecision_270m"
    cost_basis="self_hosted_gpu"
    price_input_per_m=None
    price_output_per_m=None

    def __init__(self,model="rajan2k/GemmaDecision-270M",revision=None,device="cuda",**kwargs):
        self.model=model
        self.engine=GemmaDecision.from_pretrained(model,revision=revision,device=device)

    def reserve_estimate(self,task):return None

    def run(self,task):
        from jevbench.adapters.base import DecisionResult
        import torch
        if self.engine.device=="cuda":torch.cuda.synchronize()
        start=time.perf_counter()
        try:
            q={"type":task.question["type"],"instructions":task.question.get("instructions","")}
            if task.question.get("criteria") is not None:q["criteria"]=task.question["criteria"]
            if q["type"]=="noul" and isinstance(q.get("criteria"),dict):
                q["criteria"]={"false":q["criteria"].get("false",q["criteria"].get("no")),
                               "true":q["criteria"].get("true",q["criteria"].get("yes"))}
            keys,texts=candidates(q)
            p=self.engine.rank(task.state,q["instructions"],dict(zip(keys,texts)))
            if q["type"]=="noul":p={"no":p["false"],"yes":p["true"]}
            if set(p)!=set(task.labels):raise ValueError("Exact benchmark labels do not match the rubric")
            tokens=sum(len(self.engine.tokenizer(t)["input_ids"]) for t in [schema_state(task.state,q["instructions"]),*texts])
            if self.engine.device=="cuda":torch.cuda.synchronize()
            return DecisionResult(adapter=self.name,ok=True,probs=p,probs_source="native",model=self.model,
                                  status=200,latency_s=time.perf_counter()-start,
                                  usage={"input_tokens":tokens,"output_tokens":0})
        except ValueError as exc:
            return DecisionResult(adapter=self.name,ok=False,probs_source="native",model=self.model,
                                  status=422,error=str(exc),latency_s=time.perf_counter()-start)
