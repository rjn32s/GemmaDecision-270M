"""Final authorized training/evaluation jobs; publication occurs separately."""
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent if modal.is_local() else Path('/experiment/code-v4')
PROJECT=HERE.parents[2] if modal.is_local() else Path('/experiment')
CPU_RATE=2*.0000131+16*.00000222
GPU_RATE=.001097+2*.0000131+32*.00000222
app=modal.App('gemmadecision-270m-v4-final')
volume=modal.Volume.from_name('gemma-270m-contrastive-experiment-20260927',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.14.0','transformers==5.17.0','pyarrow==25.0.1','numpy==2.4.6','ijson==3.4.0')
       .env({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1',
             'HF_HOME':'/experiment/work/huggingface','V4_ROOT':'/experiment/work/decision-v4',
             'GEMMA_MODEL_PATH':'/experiment/work/model'}))
for p in sorted(HERE.glob('*.py')):
    image=image.add_local_file(p,'/experiment/code-v4/'+p.name)
for name in ('common.py','clm_heads.py','clm_schema.py'):
    image=image.add_local_file(HERE.parent/'v2'/name,'/experiment/code-v4/'+name)
for name,target in (('data.py','data_v3.py'),('joint_deployment.py','joint_deployment.py')):
    image=image.add_local_file(HERE.parent/'v3'/name,'/experiment/code-v4/'+target)


def record(name,report,started,rate):
    import hashlib,time
    root=Path('/experiment/work/decision-v4')/name
    root.mkdir(parents=True,exist_ok=True)
    report.update(function_seconds=time.monotonic()-started)
    report['estimated_function_usd']=report['function_seconds']*rate
    (root/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/experiment/code-v4').glob('*.py')}
    (root/'source-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    files={}
    for p in root.rglob('*'):
        if not p.is_file() or p.stat().st_size>60_000_000:continue
        if any(part.startswith(('step-','latest','checkpoint','full-model','best')) for part in p.relative_to(root).parts[:-1]):continue
        if p.suffix not in {'.json','.py','.md','.txt','.safetensors'}:continue
        files[str(p.relative_to(root))]=p.read_bytes()
    return {'report':report,'files':files}


@app.function(image=image,cpu=(2,2),memory=(16384,16384),timeout=1200,startup_timeout=180,
              retries=0,max_containers=1,volumes={'/experiment/work':volume})
def cpu_job(stage,checkpoint):
    import sys,time
    sys.path.insert(0,'/experiment/code-v4');started=time.monotonic()
    try:
        if stage=='prepare':
            from data_v4 import prepare
            report=prepare()
        elif stage=='freeze':
            from evaluate_v4 import freeze
            report=freeze(checkpoint_dir=checkpoint,training_report=checkpoint.rsplit('/',1)[0]+'/metrics.json',
                          selection_rationale='Development-selected full-backbone joint Gemma270M, user-authorized final publication round.')
            import shutil
            target=Path('/experiment/work/decision-v4/freeze');target.mkdir(exist_ok=True)
            shutil.copyfile(target.parent/'frozen-recipe.json',target/'frozen-recipe.json')
            import hashlib
            identity_path=target.parent/'data/split-manifest.jsonl'
            identities=[json.loads(line) for line in identity_path.read_text().splitlines() if line.strip()]
            allowed={'id','group','workflow_group','source','family','split'}
            if any(set(row)!=allowed or not all(isinstance(value,str) for value in row.values()) for row in identities):
                raise RuntimeError('Unexpected fields in label-free split identities')
            audit=json.loads((target.parent/'data/audit.json').read_text())
            identity_sha=hashlib.sha256(identity_path.read_bytes()).hexdigest()
            if identity_sha!=audit['output_hashes']['split-manifest.jsonl']:raise RuntimeError('Split identities changed')
            (target/'split-identities.json').write_text(json.dumps({'source_jsonl_sha256':identity_sha,
                'contains_raw_text_or_labels':False,'rows':identities},indent=2)+'\n')
        elif stage=='package':
            from package_v4 import run
            report=run(seconds=500)
        elif stage=='verify_public':
            from verify_public_v4 import run
            report=run(revision=checkpoint,seconds=780)
        else:raise ValueError(stage)
        return record(stage,report,started,CPU_RATE)
    finally:volume.commit()


@app.function(image=image,gpu='H100',cpu=(2,2),memory=(32768,32768),timeout=2520,startup_timeout=180,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/experiment/work':volume})
def gpu_job(stage,seconds,seed,run_name):
    import sys,time
    sys.path.insert(0,'/experiment/code-v4');started=time.monotonic()
    import torch
    torch.set_num_threads(2)
    try:
        root=Path('/experiment/work/decision-v4')
        if stage in {'train','development'} and (root/'frozen-recipe.json').exists():
            raise RuntimeError('Selection is frozen; this round cannot resume training or development selection')
        if stage=='train':
            from train import run
            report=run(seconds=seconds,max_steps=6000,epochs=4,seed=seed,run_name=run_name)
            directory=run_name
        elif stage=='development':
            from evaluate_v4 import run_development
            report=run_development(seconds=seconds);directory='development-baselines'
        elif stage=='evaluate':
            from evaluate_v4 import run
            report=run(seconds=seconds);directory='final-evaluation'
        else:raise ValueError(stage)
        return record(directory,report,started,GPU_RATE)
    finally:volume.commit()


@app.local_entrypoint()
def main(stage:str='prepare',seed:int=2704206,run_name:str='train',checkpoint:str='train/best',hf_revision:str=''):
    import fcntl
    from datetime import datetime,timezone
    from modal import Workspace
    configs={'prepare':('data',1000),'train':('training',2520),'development':('evaluation',900),
             'freeze':('evaluation',300),'evaluate':('evaluation',1500),'package':('evaluation',600),
             'verify_public':('evaluation',900)}
    if stage not in configs:raise ValueError('Unknown v4 stage')
    if run_name not in {'train','train-repeat'}:raise ValueError('Unknown experiment namespace')
    if seed not in {2704206,2704207}:raise ValueError('Unplanned seed')
    if checkpoint not in {'train/best','train-repeat/best'}:raise ValueError('Unknown selected checkpoint')
    if stage=='verify_public':
        import re
        if not re.fullmatch('[0-9a-f]{40}',hf_revision):raise ValueError('Public verification requires a pinned commit')
    group,timeout=configs[stage];cpu=stage in {'prepare','freeze','package','verify_public'};rate=CPU_RATE if cpu else GPU_RATE
    reservation=(timeout+180)*rate+.15
    guard=(PROJECT/'work/decision-training.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=PROJECT/'work/decision-budget-ledger.json';state=json.loads(ledger.read_text())
    budget=state['v4'];used=float(Workspace.from_context().billing.summary().metered_cost)
    budget.update(publication_on_hold=False,github_account='rjn32s',huggingface_account='rajan2k',
                  publication_authorization='User requested final run and publication to GH rjn32s and HF.',
                  benchmark_submission_on_hold=True)
    previous=[j for j in state['jobs'] if j.get('version')==4]
    def cost(j):return j.get('estimated_function_usd',j['reserved_usd'])+(.15 if 'estimated_function_usd' in j else 0)
    estimate=sum(cost(j) for j in previous)
    category=sum(cost(j) for j in previous if j.get('group')==group)
    cap={'data':1,'training':12,'evaluation':3}[group]
    if category+reservation>cap:raise RuntimeError(f'V4 {group} would exceed its ${cap} allocation')
    if max(used-budget['baseline_metered_usd'],estimate)+reservation>20:raise RuntimeError('V4 would exceed $20')
    if max(used,budget['baseline_metered_usd']+estimate)+reservation>30:raise RuntimeError('Project would exceed $30')
    job={'version':4,'stage':stage,'group':group,'run_name':run_name,'seed':seed,'status':'reserved',
         'reserved_usd':reservation,'metered_before_usd':used,'started':datetime.now(timezone.utc).isoformat()}
    state['jobs'].append(job);ledger.write_text(json.dumps(state,indent=2)+'\n')
    print(f'V4 {stage} {run_name}: reserve ${reservation:.3f}, meter ${used:.2f}, round ceiling $20, project ceiling $30',flush=True)
    try:
        payload=(cpu_job.with_options(timeout=timeout).remote(stage,hf_revision if stage=='verify_public' else checkpoint) if cpu else
                 gpu_job.with_options(timeout=timeout).remote(stage,timeout-120,seed,run_name))
        report=payload['report'];directory=run_name if stage=='train' else stage
        output=HERE/'results'/directory;output.mkdir(parents=True,exist_ok=True)
        for name,content in payload['files'].items():
            destination=output/name
            if not destination.resolve().is_relative_to(output.resolve()):raise RuntimeError('Unsafe artifact path')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
        job.update(status='completed',result_status=report.get('status'),estimated_function_usd=report['estimated_function_usd'])
        print(json.dumps({k:report[k] for k in ('status','steps','best','function_seconds','estimated_function_usd') if k in report},indent=2),flush=True)
        print(f'Report: {output}/metrics.json',flush=True)
    except BaseException:
        job['status']='failed';raise
    finally:
        ledger.write_text(json.dumps(state,indent=2)+'\n');guard.close()
