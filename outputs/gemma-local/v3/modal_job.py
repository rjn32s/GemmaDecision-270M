"""Authorized v3 research jobs. No publication or benchmark submission commands."""
from pathlib import Path
import json
import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path('/experiment/code-v3')
PROJECT = HERE.parents[2] if modal.is_local() else Path('/experiment')
CPU_RATE = 2 * .0000131 + 16 * .00000222
GPU_RATE = .000542 + 2 * .0000131 + 32 * .00000222
app = modal.App('gemmadecision-270m-v3')
volume = modal.Volume.from_name('gemma-270m-contrastive-experiment-20260927', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.0','transformers==5.17.0','pyarrow==25.0.1','numpy==2.4.6','ijson==3.4.0')
         .env({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false',
               'HF_HOME':'/experiment/work/huggingface','HF_HUB_DISABLE_TELEMETRY':'1',
               'V3_ROOT':'/experiment/work/decision-v3','GEMMA_MODEL_PATH':'/experiment/work/model'}))
for file in sorted(HERE.glob('*.py')):
    image = image.add_local_file(file, '/experiment/code-v3/' + file.name)
for name in ('common.py','clm_heads.py','clm_schema.py'):
    image = image.add_local_file(HERE.parent/'v2'/name, '/experiment/code-v3/' + name)


def collect(stage):
    root=Path('/experiment/work/decision-v3')/stage
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.suffix in {'.json','.safetensors','.py','.md','.txt'}
            and not any(part.startswith('step-') for part in p.relative_to(root).parts)
            and p.stat().st_size<60_000_000}


def record(stage,report,start,rate):
    import time,hashlib
    from common import save
    report['function_seconds']=time.monotonic()-start
    report['estimated_function_usd']=report['function_seconds']*rate
    root=Path('/experiment/work/decision-v3')/stage
    save(root/'metrics.json',report)
    save(root/'source-hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path('/experiment/code-v3').glob('*.py')})
    return collect(stage)


@app.function(image=image,cpu=(2,2),memory=(16384,16384),timeout=1200,startup_timeout=600,
              retries=0,max_containers=1,volumes={'/experiment/work':volume})
def cpu_job(stage,arm=''):
    import sys,time
    sys.path.insert(0,'/experiment/code-v3')
    start=time.monotonic()
    try:
        if stage=='prepare':
            from data import prepare
            report=prepare('/experiment/work/decision-v3/data','/experiment/work/decision-v2','/experiment/work/model')
        elif stage=='assets':
            from huggingface_hub import snapshot_download,hf_hub_download
            from common import QWEN_REVISION,CLM_REVISION
            snapshot_download('Qwen/Qwen3-8B',revision=QWEN_REVISION,local_dir='/experiment/work/qwen-reference',
                              allow_patterns=['*.safetensors','*.json','*.txt','LICENSE'],max_workers=3)
            hf_hub_download('Contrastive-LM/CLM-v0.1-8B','CLM_v0.1-8B.pt',revision=CLM_REVISION,
                            local_dir='/experiment/work/clm-reference')
            report={'status':'complete','qwen_revision':QWEN_REVISION,'clm_revision':CLM_REVISION}
        elif stage=='compare':
            from train import compare
            report=compare()
        elif stage=='extension_compare':
            from extension_compare import run
            report=run()
        elif stage=='audit':
            from audit import audit
            report=audit('/experiment/work/decision-v3/data')
        elif stage=='repair':
            from data import repair
            report=repair('/experiment/work/decision-v3/data')
        elif stage=='package':
            from package_local import run
            report=run(arm,seconds=450)
        elif stage=='package_joint':
            from package_joint import run
            report=run(seconds=450)
        elif stage=='freeze':
            from freeze import run
            report=run(arm)
        else: raise ValueError(stage)
        return record(stage,report,start,CPU_RATE)
    finally: volume.commit()


@app.function(image=image,gpu='L40S',cpu=(2,2),memory=(32768,32768),timeout=1800,startup_timeout=600,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/experiment/work':volume})
def gpu_job(stage,seconds,steps):
    import sys,time,os,hashlib
    sys.path.insert(0,'/experiment/code-v3')
    import torch
    torch.set_num_threads(2)
    os.environ['V3_STEPS']=str(steps)
    start=time.monotonic()
    try:
        if stage in {'frozen','lora','frozen_extension','joint','baselines','full_development','profile'} and Path('/experiment/work/decision-v3/final-evaluation/final-labels-opened.json').exists():
            raise RuntimeError('Final labels are open; this experiment cannot resume tuning or selection')
        audit_path=Path('/experiment/work/decision-v3/audit/metrics.json')
        audit_report=json.loads(audit_path.read_text())
        if not audit_report['passed_structural'] or audit_report['near_duplicate_review_required']:
            raise RuntimeError('Dataset audit must pass before GPU execution')
        for name,expected in audit_report['split_file_sha256'].items():
            actual=hashlib.sha256((Path('/experiment/work/decision-v3/data')/name).read_bytes()).hexdigest()
            if actual!=expected: raise RuntimeError('Dataset changed since successful audit')
        if stage=='reference':
            from reference import run
            report=run(seconds)
        elif stage=='full_development':
            from full_development import run
            report=run(seconds)
        elif stage=='profile':
            from training_profile import run
            report=run(min(seconds,120))
        elif stage=='frozen_extension':
            os.environ['V3_FULL_DEVELOPMENT']='1'
            os.environ['V3_OUTPUT_NAME']=stage
            from train import run
            report=run('frozen',seconds)
        elif stage=='joint':
            from joint import run
            report=run(seconds)
        elif stage=='final_evaluation':
            from evaluate import run
            report=run(seconds)
        else:
            from train import run
            report=run(stage,seconds)
        return record('final-evaluation' if stage=='final_evaluation' else stage,report,start,GPU_RATE)
    finally: volume.commit()


@app.local_entrypoint()
def main(stage:str='prepare',steps:int=300,arm:str=''):
    import fcntl
    from datetime import datetime,timezone
    from modal import Workspace
    configs={'prepare':1200,'repair':600,'audit':600,'assets':900,'baselines':900,'reference':600,
             'frozen':900,'lora':1800,'compare':300,'full_development':900,'extension_compare':300,
             'profile':300,'package':600,'package_joint':600,'freeze':300,'final_evaluation':1800,
             'frozen_extension':900,'joint':900}
    if stage not in configs: raise ValueError('Research pilot stage only')
    if stage=='package' and arm not in {'frozen','lora','frozen_extension','v2_unchanged','v2_hard'}:
        raise ValueError('Select a reviewed checkpoint explicitly for packaging')
    if stage=='freeze' and arm not in {'joint','frozen_extension'}:
        raise ValueError('Select the reviewed development choice explicitly')
    group='extension' if stage=='extension_compare' else 'pilot'
    if stage in {'freeze','final_evaluation','package_joint'}: group='evaluation'
    if steps!=300:
        decision_file='joint-decision.json' if stage=='joint' else 'extension-decision.json'
        decision=json.loads((HERE/decision_file).read_text())
        if stage!=decision['arm'] or decision['decision']!='extend' or not 300<steps<=decision['max_steps']:
            raise ValueError('Extension must match the recorded development decision')
        group='extension'
    cpu_stage=stage in {'prepare','repair','audit','assets','compare','extension_compare','package','package_joint','freeze'}
    timeout=configs[stage];rate=CPU_RATE if cpu_stage else GPU_RATE
    reservation=(timeout+600)*rate+.10
    guard=(PROJECT/'work/decision-training.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=PROJECT/'work/decision-budget-ledger.json'
    state=json.loads(ledger.read_text())
    budget=state.setdefault('v3',{'baseline_metered_usd':2.12,'new_budget_usd':20,
                                 'allocations':{'pilot':5,'extension':7,'larger_comparison':4,'evaluation':4},
                                 'publication_on_hold':True,'github_account':'rjn32s'})
    previous=[j for j in state['jobs'] if j.get('version')==3]
    def cost(j): return j.get('estimated_function_usd',j['reserved_usd'])+(.10 if 'estimated_function_usd' in j else 0)
    estimate=sum(cost(j) for j in previous)
    used=float(Workspace.from_context().billing.summary().metered_cost)
    group_estimate=sum(cost(j) for j in previous if j['group']==group)
    metered_delta=max(0.,used-budget['baseline_metered_usd'])
    group_cap=budget['allocations'][group]
    if stage=='joint' and sum(cost(j) for j in previous if j['stage']=='joint')+reservation>1.50:
        raise RuntimeError('Candidate-conditioned comparison exceeds its $1.50 cap')
    if group_estimate+reservation>group_cap:
        raise RuntimeError(f'{group} reservation exceeds ${group_cap}; do not silently expand the allocation.')
    if max(used-budget['baseline_metered_usd'],estimate)+reservation>20:
        raise RuntimeError('V3 would exceed the authorized $20 new-work cap')
    if max(used,budget['baseline_metered_usd']+estimate)+reservation>30:
        raise RuntimeError('Experiment would exceed the original $30 total cap')
    job={'version':3,'stage':stage,'steps':steps,'group':group,'reserved_usd':reservation,
         'status':'reserved','metered_before_usd':used,'started':datetime.now(timezone.utc).isoformat()}
    state['jobs'].append(job);ledger.write_text(json.dumps(state,indent=2)+'\n')
    print(f'V3 {stage}: reservation ${reservation:.3f}; metered ${used:.2f}; {group} cap ${group_cap}; no publication',flush=True)
    try:
        files=(cpu_job.with_options(timeout=timeout).remote(stage,arm) if cpu_stage
               else gpu_job.with_options(timeout=timeout).remote(stage,timeout-120,steps))
        destination=HERE/'results'/stage;destination.mkdir(parents=True,exist_ok=True)
        for name,content in files.items():
            p=destination/name
            if not p.resolve().is_relative_to(destination.resolve()): raise ValueError('Invalid artifact path')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        report=json.loads(files['metrics.json'])
        job.update(status='completed',estimated_function_usd=report['estimated_function_usd'],result_status=report.get('status'))
        brief={key:report[key] for key in ('status','steps','best','passed_structural',
               'near_duplicate_review_required','function_seconds','estimated_function_usd') if key in report}
        if 'systems' in report:
            brief['development_macro']={name:value.get('fresh_family_macro') for name,value in report['systems'].items()}
        print(json.dumps(brief,indent=2),flush=True)
        print(f'Full report: {destination / "metrics.json"}',flush=True)
    except BaseException:
        job['status']='failed';raise
    finally:
        ledger.write_text(json.dumps(state,indent=2)+'\n');guard.close()
