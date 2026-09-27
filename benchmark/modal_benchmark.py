"""Budget-checked Modal execution of the fixed public JevBench protocol.

Project launcher: model inference runs remotely, not on the launcher's machine.
For portable evaluator commands, see README.md beside this file.
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path('/benchmark')
PROJECT = HERE.parents[3] if modal.is_local() else Path('/experiment')
RATE = .001097 + 2 * .0000131 + 32 * .00000222
TIMEOUT = 1200
RUN_ID = 'jevbench-v4-public-01'
app = modal.App('gemmadecision-v4-jevbench')
volume = modal.Volume.from_name('gemma-270m-contrastive-experiment-20260927', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.0', 'transformers==5.17.0', 'safetensors==0.8.0',
                      'numpy==2.4.6', 'huggingface_hub==1.33.0')
         .env({'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HOME': '/experiment/work/huggingface'}))
for name in ('server.py', 'run_public.py', 'PROTOCOL.json'):
    image = image.add_local_file(HERE / name, '/benchmark/' + name)
if modal.is_local():
    upstream = PROJECT / 'work/jevbench-submission'
else:
    upstream = Path('/jevbench')
image = (image.add_local_dir(upstream / 'jevbench', '/jevbench/jevbench', ignore=['__pycache__'])
         .add_local_dir(upstream / 'datasets/public', '/jevbench/datasets/public')
         .add_local_file(upstream / 'UPSTREAM_REVISION.json', '/jevbench/UPSTREAM_REVISION.json'))


@app.function(image=image, gpu='H100', cpu=(2, 2), memory=(32768, 32768),
              timeout=TIMEOUT, startup_timeout=180, retries=0, max_containers=1,
              scaledown_window=2, volumes={'/experiment/work': volume})
def benchmark():
    import hashlib
    import os
    import platform
    import subprocess
    import sys
    import time
    import urllib.request
    from datetime import datetime, timezone
    import torch
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    root = Path('/experiment/work/decision-v4') / RUN_ID
    root.mkdir(exist_ok=False)
    protocol = json.loads(Path('/benchmark/PROTOCOL.json').read_text())
    (root / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    proc = None
    status = 'failed'
    error = None
    try:
        for name, digest in protocol['benchmark_source_sha256'].items():
            if hashlib.sha256((Path('/benchmark') / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Benchmark source differs from preregistered protocol: ' + name)
        for name, digest in protocol['public_dataset_sha256'].items():
            if hashlib.sha256((Path('/jevbench/datasets/public') / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Public dataset hash mismatch: ' + name)
        model_path = Path(snapshot_download(protocol['model_id'], revision=protocol['model_revision'], token=False))
        manifest = json.loads((model_path / 'SHA256SUMS.json').read_text())
        for name, entry in manifest['files'].items():
            path = model_path / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise RuntimeError('Published model bundle hash mismatch: ' + name)
        ready_started = time.monotonic()
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        for secret_name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN'):
            env.pop(secret_name, None)
        with (root / 'server.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '/benchmark/server.py', '--model-dir', str(model_path),
                                     '--device', 'cuda', '--host', '127.0.0.1', '--port', '8000',
                                     '--torch-threads', '2'], stdout=log, stderr=subprocess.STDOUT, env=env)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Model server exited before readiness: ' + (root / 'server.log').read_text()[-3000:])
                if time.monotonic() - ready_started > 240:
                    raise TimeoutError('Model server did not become ready in 240 seconds')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                        health = json.load(response)
                    break
                except (OSError, ValueError):
                    time.sleep(.5)
            model_load_seconds = time.monotonic() - ready_started
            print('Pinned public model loaded; starting one serial 231-case public run.', flush=True)
            sys.path.insert(0, '/benchmark')
            from run_public import run
            report = run('/jevbench', root / 'public', 'http://127.0.0.1:8000')
            (root / 'environment.json').write_text(json.dumps({
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'cpu_reserved_cores': 2, 'memory_reserved_mib': 32768, 'python': platform.python_version(),
                'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'platform': platform.platform(),
                'precision': 'BF16 Gemma encoder, FP32 scalar head', 'torch_cpu_threads': 2,
                'transport': 'serial loopback HTTP with unmodified upstream TypeSafeAdapter and Runner',
                'model_load_to_ready_seconds': model_load_seconds, 'health': health,
                'cold_inference': 'No inference warmup; first benchmark call included in latency',
                'inference_network': 'Local model only; public snapshot downloaded before server startup',
                'model_revision': protocol['model_revision'],
            }, indent=2) + '\n')
            status = 'completed'
    except Exception as exc:
        error = str(exc)
        print('Benchmark failed: ' + error, flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        elapsed = time.monotonic() - started
        compute = {'status': status, 'error': error, 'function_seconds': elapsed,
                   'estimated_function_usd': elapsed * RATE, 'rate_usd_per_second': RATE,
                   'finished_at': datetime.now(timezone.utc).isoformat(),
                   'billing_scope': 'Function body GPU+reserved CPU/RAM estimate; excludes startup, storage, transfer and taxes. Not an inference-service tariff.'}
        (root / 'compute.json').write_text(json.dumps(compute, indent=2) + '\n')
        volume.commit()
    files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
             if p.is_file() and p.stat().st_size < 20_000_000}
    return {'report': compute, 'files': files}


@app.local_entrypoint()
def main():
    import fcntl
    from datetime import datetime, timezone
    from modal import Workspace
    reservation = (TIMEOUT + 180) * RATE + .15
    guard = (PROJECT / 'work/decision-training.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = PROJECT / 'work/decision-budget-ledger.json'
    state = json.loads(path.read_text())
    budget = state['v4']
    used = float(Workspace.from_context().billing.summary().metered_cost)
    previous = [j for j in state['jobs'] if j.get('version') == 4]
    def cost(j):
        return j.get('estimated_function_usd', j['reserved_usd']) + (.15 if 'estimated_function_usd' in j else 0)
    estimate = sum(cost(j) for j in previous)
    if any(j.get('run_name') == RUN_ID for j in previous):
        raise RuntimeError('This public run is already reserved or completed; do not silently rerun')
    if max(used - budget['baseline_metered_usd'], estimate) + reservation > 20:
        raise RuntimeError('Benchmark would exceed the authorized $20 round budget')
    if max(used, budget['baseline_metered_usd'] + estimate) + reservation > 30:
        raise RuntimeError('Benchmark would exceed the $30 project ceiling')
    budget.update(benchmark_submission_on_hold=False,
                  benchmark_authorization="User: ok let's prepoare for bechmark and it's submission")
    job = {'version': 4, 'stage': 'jevbench-public', 'group': 'benchmark', 'run_name': RUN_ID,
           'status': 'reserved', 'reserved_usd': reservation, 'metered_before_usd': used,
           'started': datetime.now(timezone.utc).isoformat()}
    state['jobs'].append(job)
    path.write_text(json.dumps(state, indent=2) + '\n')
    print(f'JevBench: reserving at most ${reservation:.3f}; current workspace meter ${used:.2f}.', flush=True)
    try:
        payload = benchmark.remote()
        output = HERE / 'results' / RUN_ID
        output.mkdir(parents=True, exist_ok=False)
        for name, content in payload['files'].items():
            destination = output / name
            if not destination.resolve().is_relative_to(output.resolve()):
                raise RuntimeError('Unsafe artifact path')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        job.update(payload['report'])
        print(json.dumps(payload['report'], indent=2), flush=True)
        if payload['report']['status'] != 'completed':
            raise RuntimeError('Remote benchmark did not complete; evidence preserved')
    except BaseException:
        job['status'] = 'failed'
        raise
    finally:
        path.write_text(json.dumps(state, indent=2) + '\n')
        guard.close()
