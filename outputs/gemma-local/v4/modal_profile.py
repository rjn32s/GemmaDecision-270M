"""Bounded cloud full-backbone GPU comparison. No evaluation or publishing."""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path('/experiment/code-v4')
PROJECT = HERE.parents[2] if modal.is_local() else Path('/experiment')
OVERHEAD_RATE = 2 * .0000131 + 32 * .00000222
RATES = {'L40S': .000542 + OVERHEAD_RATE, 'H100!': .001097 + OVERHEAD_RATE,
         'A100-80GB': .000694 + OVERHEAD_RATE}
app = modal.App('gemmadecision-270m-v4-gpu-profile')
volume = modal.Volume.from_name('gemma-270m-contrastive-experiment-20260927', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.14.0','transformers==5.17.0','pyarrow==25.0.1','numpy==2.4.6','ijson==3.4.0')
    .env({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false',
          'HF_HOME':'/experiment/work/huggingface','HF_HUB_DISABLE_TELEMETRY':'1'}))
image = image.add_local_file(HERE/'gpu_profile.py','/experiment/code-v4/gpu_profile.py')
for name in ('common.py','clm_heads.py','clm_schema.py'):
    image = image.add_local_file(HERE.parent/'v2'/name,'/experiment/code-v4/'+name)


@app.function(image=image,gpu='L40S',cpu=(2,2),memory=(32768,32768),
              timeout=150,startup_timeout=120,retries=0,max_containers=1,
              scaledown_window=2,volumes={'/experiment/work':volume})
def profile(gpu):
    import hashlib,sys,time
    sys.path.insert(0,'/experiment/code-v4')
    started=time.monotonic()
    from gpu_profile import run
    try:
        report=run(seconds=110)
        report.update(requested_gpu=gpu,function_seconds=time.monotonic()-started)
        report['estimated_function_usd']=report['function_seconds']*RATES[gpu]
        report['estimated_hourly_usd']=RATES[gpu]*3600
        report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path('/experiment/code-v4').glob('*.py')}
        output=Path('/experiment/work/decision-v4/gpu-profile')/gpu.replace('!','')
        output.mkdir(parents=True,exist_ok=True)
        (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    finally:
        volume.commit()


@app.local_entrypoint()
def main(gpu:str='L40S'):
    import fcntl
    from datetime import datetime,timezone
    from modal import Workspace
    if gpu not in RATES:raise ValueError('GPU is outside the costed profile options')
    guard=(PROJECT/'work/decision-training.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=PROJECT/'work/decision-budget-ledger.json'
    state=json.loads(ledger.read_text())
    used=float(Workspace.from_context().billing.summary().metered_cost)
    budget=state.setdefault('v4',{'baseline_metered_usd':used,'new_budget_usd':20,
        'profile_ceiling_usd':1,'publication_on_hold':True,
        'note':'User permits faster GPUs; compare with bounded disposable training updates.'})
    previous=[j for j in state['jobs'] if j.get('version')==4]
    def cost(j):return j.get('estimated_function_usd',j['reserved_usd'])+(.08 if 'estimated_function_usd' in j else 0)
    estimate=sum(cost(j) for j in previous)
    reservation=(150+120)*RATES[gpu]+.08
    # Retain full reservations for this tiny fixed comparison: body timing omits
    # startup/commit/teardown and metering can lag. Do not recycle that margin.
    profiles=sum(max(j['reserved_usd'],cost(j)) for j in previous if j['stage']=='gpu_profile')
    if profiles+reservation>1:raise RuntimeError('The GPU comparison would exceed its $1 cap')
    if max(used-budget['baseline_metered_usd'],estimate)+reservation>20:
        raise RuntimeError('The next round would exceed its $20 cap')
    if max(used,budget['baseline_metered_usd']+estimate)+reservation>30:
        raise RuntimeError('The project would exceed its original $30 cap')
    job={'version':4,'stage':'gpu_profile','gpu':gpu,'reserved_usd':reservation,
         'status':'reserved','metered_before_usd':used,'started':datetime.now(timezone.utc).isoformat()}
    state['jobs'].append(job);ledger.write_text(json.dumps(state,indent=2)+'\n')
    print(f'GPU profile {gpu}: reservation ${reservation:.3f}; total comparison cap $1; no publication',flush=True)
    try:
        report=profile.with_options(gpu=gpu).remote(gpu)
        output=HERE/'results/gpu-profile'/gpu.replace('!','')
        output.mkdir(parents=True,exist_ok=True)
        (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        job.update(status='completed',result_status=report.get('status'),
                   estimated_function_usd=report['estimated_function_usd'])
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    except BaseException:
        job['status']='failed';raise
    finally:
        ledger.write_text(json.dumps(state,indent=2)+'\n');guard.close()
