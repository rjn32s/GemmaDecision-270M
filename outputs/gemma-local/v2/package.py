"""Build and exercise a standalone release folder on the cloud GPU."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from common import *


def verify_package(deadline):
    root=Path("/experiment/work/decision-v2")
    release=root/"release"
    release.mkdir(exist_ok=True)
    results=json.loads((root/"evaluate/metrics.json").read_text())
    for p in Path("/experiment/work/model").iterdir():
        if p.is_file() and (p.suffix in {".json",".safetensors",".model"}):shutil.copy2(p,release/p.name)
    for name in ["common.py","clm_heads.py","clm_schema.py","decision_model.py"]:
        shutil.copy2(Path(__file__).parent/name,release/name)
    shutil.copy2(root/"train/agent.safetensors",release/"decision_heads.safetensors")
    config={"model_name":"GemmaDecision-270M","version":2,"format":FORMAT,"base_model":"google/gemma-3-270m",
            "base_revision":MODEL_REVISION,"head_config":HEAD_CONFIG,"temperature":results["temperature"],
            "max_state_tokens":MAX_STATE,"max_action_tokens":MAX_ACTION,"selected_stage":"agent",
            "quality_target_met":results["quality_target_met"],"experimental":True}
    save(release/"decision_config.json",config)
    program='''
import json, math
from decision_model import GemmaDecision
m=GemmaDecision('.',device='cuda')
request={'state':{'customer':'My invoice was charged twice.'},'questions':{
 'department':{'type':'choice','instructions':'Which department should handle this?',
 'criteria':{'billing':'Charges, invoices, refunds','technical':'Software errors and outages','shipping':'Delivery and tracking'}},
 'urgent':{'type':'noul','instructions':'Does this require immediate emergency action?'},
 'frustration':{'type':'score','instructions':'How frustrated is the customer?','criteria':['Calm','Frustrated','Very angry']}}}
r=m.decide(request)
p=r['answers']['department']['probabilities']
assert all(math.isfinite(x) for x in p.values()) and abs(sum(p.values())-1)<1e-5
assert 0<=r['answers']['urgent']['noul']<=1
assert 0<=r['answers']['frustration']['score']<=2
q=request['questions']['department']
reverse=m.rank(request['state'],q['instructions'],dict(reversed(list(q['criteria'].items()))))
error=max(abs(p[k]-reverse[k]) for k in p)
assert error<1e-4
print(json.dumps({'status':'passed','request':request,'response':r,'candidate_order_max_error':error}))
'''
    env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONPATH":str(release)}
    result=subprocess.run(["python","-c",program],cwd=release,env=env,capture_output=True,text=True,timeout=300)
    if result.returncode:
        raise RuntimeError("Standalone verification failed: "+result.stderr[-2500:])
    report=json.loads(result.stdout.strip().splitlines()[-1])
    save(root/"package/standalone-check.json",report)
    save(root/"package/decision_config.json",config)
    return {"status":"complete","standalone_load":"passed","checks":report,
            "release_files":[p.name for p in release.iterdir()],"private_volume_dependency":False,
            "note":"Fresh subprocess loaded only the standalone release folder, with network disabled."}
