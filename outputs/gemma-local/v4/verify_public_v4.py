"""Anonymous fresh public download and offline CPU release verification on Modal."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from package_v4 import CPU_CHECK
from evaluate_v4 import sha, save, paths, validate_recipe


def run(revision,seconds=780):
    import modal
    if modal.is_local():raise RuntimeError('Public model verification runs on Modal CPU only')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('An immutable HF commit is required')
    started=time.monotonic();root,_,_=paths();recipe,identity=validate_recipe()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo='rajan2k/GemmaDecision-270M'
    with tempfile.TemporaryDirectory(prefix='gemmadecision-public-') as temporary:
        temporary=Path(temporary)
        api=HfApi(token=False)
        info=api.model_info(repo,revision=revision)
        if info.sha!=revision or info.private:raise RuntimeError('Wrong public release identity')
        manifest_path=hf_hub_download(repo,'SHA256SUMS.json',revision=revision,
                                      token=False,cache_dir=temporary/'manifest-cache')
        manifest=json.loads(Path(manifest_path).read_text());entries=manifest.get('files',manifest)
        bundle=temporary/'model'
        snapshot_download(repo,revision=revision,token=False,local_dir=bundle,
                          allow_patterns=[*entries,'SHA256SUMS.json'],max_workers=4)
        for name,item in entries.items():
            path=(bundle/name).resolve()
            expected=item['sha256'] if isinstance(item,dict) else item
            if not path.is_relative_to(bundle.resolve()) or sha(path)!=expected:
                raise RuntimeError('Public release checksum mismatch: '+name)
        if sha(bundle/'model.safetensors')!=recipe['selected']['files']['model.safetensors']:
            raise RuntimeError('Published backbone differs from the selected model')
        if sha(bundle/'joint_head.safetensors')!=recipe['selected']['files'][recipe['selected']['head_file']]:
            raise RuntimeError('Published scalar head differs from the selected model')
        for name,digest in recipe['selected']['files'].items():
            if name=='joint_config.json':continue
            if sha(bundle/name)!=digest:raise RuntimeError('Published checkpoint file changed: '+name)
        for name in ('joint_deployment.py','common.py','clm_schema.py','clm_heads.py'):
            if sha(bundle/name)!=recipe['sources_sha256'][name]:raise RuntimeError('Published serving source changed: '+name)
        if sha(bundle/'reproduction/initial-v3-joint-head.safetensors')!=recipe['v3_head_sha256']:
            raise RuntimeError('Published initialization head changed')
        config=json.loads((bundle/'joint_config.json').read_text())
        if config['head_config']!=recipe['selected']['head_config'] or config['head_file']!='joint_head.safetensors' or config['max_state_tokens']!=recipe['limits']['state'] or config['max_action_tokens']!=recipe['limits']['candidate']:
            raise RuntimeError('Published serving contract changed')
        if json.loads((bundle/'evaluation.json').read_text())['identity']!=identity:
            raise RuntimeError('Published evaluation belongs to another recipe')
        if sha(bundle/'frozen-recipe.json')!=identity['recipe_sha256']:
            raise RuntimeError('Published recipe changed')
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
             'HF_HOME':str(temporary/'empty-offline-cache'),'CUDA_VISIBLE_DEVICES':'',
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1'}
        remaining=int(seconds-(time.monotonic()-started)-15)
        if remaining<60:raise RuntimeError('Insufficient time for public offline verification')
        result=subprocess.run([sys.executable,'-I','-c',CPU_CHECK,str(bundle)],cwd=temporary,
                              env=env,capture_output=True,text=True,timeout=remaining)
        if result.returncode:raise RuntimeError('Public offline check failed: '+result.stderr[-2500:])
        check=json.loads(result.stdout.strip().splitlines()[-1])
        if check.get('status')!='passed':raise RuntimeError('Public offline check did not pass')
    report={'status':'passed','repository':repo,'revision':revision,'anonymous_download':True,
            'fresh_download':True,'all_manifest_hashes_verified':True,
            'published_weights_equal_frozen_selection':True,'recipe_identity':identity,
            'offline_cpu_verification':check,'seconds':time.monotonic()-started}
    save(root/'verify_public/metrics.json',report)
    return report
