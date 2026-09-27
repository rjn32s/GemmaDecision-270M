"""Cloud-CPU assembly and offline verification of a selected joint ranker.

Creates a complete local-use package without changing the published v2 model,
training code, official Gemma base weights, or any evaluation partition. Passing
software checks is kept separate from the correctness of illustrative examples.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from package_local import _save, _sha, _copy, EXAMPLES

HELPERS = ("joint_deployment.py", "common.py", "clm_heads.py", "clm_schema.py")
EXPECTED_DEMO_LABELS = {"support_routing": "payments", "evidence_relation": "contradicted", "tool_choice": "weather"}

CPU_CHECK = r'''
import gc, json, math, socket, sys, time
from pathlib import Path
bundle = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(bundle))
attempted_connections = []
def deny_network(*args, **kwargs):
    attempted_connections.append(True)
    raise RuntimeError('Network access is disabled during joint-package verification')
socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from joint_deployment import GemmaJointRanker
examples = json.loads((bundle/'verification-examples.json').read_text())
started = time.monotonic()
model = GemmaJointRanker(bundle, device='cpu')
assert model.device.type == 'cpu'
assert next(model.encoder.parameters()).dtype == torch.float32
assert all(not p.requires_grad for p in model.encoder.parameters())
outputs = []
max_permutation_error = 0.
max_extra_candidate_error = 0.
for example in examples:
    tick = time.monotonic()
    ranking = model.rank(example['state'], example['candidates'], example['question'])
    seconds = time.monotonic()-tick
    scores = {item['candidate']:item['score'] for item in ranking}
    assert set(scores) == set(example['candidates'])
    assert all(math.isfinite(value) for value in scores.values())
    assert all(ranking[i]['score'] >= ranking[i+1]['score'] for i in range(len(ranking)-1))
    reverse = model.rank(example['state'], dict(reversed(list(example['candidates'].items()))), example['question'])
    reversed_scores = {item['candidate']:item['score'] for item in reverse}
    permutation_error = max(abs(scores[key]-reversed_scores[key]) for key in scores)
    assert permutation_error < 1e-5, ('Candidate reversal changed scalar scores', permutation_error)
    expanded = dict(example['candidates'])
    expanded['archive_without_processing'] = 'Archive this request without processing it.'
    additional = model.rank(example['state'], expanded, example['question'])
    additional_scores = {item['candidate']:item['score'] for item in additional}
    expansion_error = max(abs(scores[key]-additional_scores[key]) for key in scores)
    assert expansion_error < 1e-5, ('An extra candidate changed original scalar scores', expansion_error)
    max_permutation_error = max(max_permutation_error, permutation_error)
    max_extra_candidate_error = max(max_extra_candidate_error, expansion_error)
    # These labels are illustrative task expectations, never pass/fail assertions.
    outputs.append({'name':example['name'], 'ranking':ranking, 'cpu_seconds':seconds,
        'illustrative_expected_top':example['illustrative_expected_top'],
        'illustrative_top_matches_expectation':ranking[0]['candidate']==example['illustrative_expected_top'],
        'candidate_order_max_error':permutation_error, 'extra_candidate_max_error':expansion_error})

for mode in ('state','candidate'):
    try:
        if mode == 'state':
            model.rank('word '*(model.max_state_tokens+32), {'a':'Accept.','b':'Reject.'})
        else:
            model.rank('A short state.', {'a':'word '*(model.max_action_tokens+32),'b':'Reject.'})
    except ValueError as error:
        assert 'limit' in str(error)
    else:
        raise AssertionError('Overlength '+mode+' was silently accepted')
semantics = model.score_semantics
del model
gc.collect()
reloaded = GemmaJointRanker(bundle, device='cpu')
max_reload_error = 0.
for example, previous in zip(examples, outputs):
    ranking = reloaded.rank(example['state'],example['candidates'],example['question'])
    original = {item['candidate']:item['score'] for item in previous['ranking']}
    current = {item['candidate']:item['score'] for item in ranking}
    max_reload_error = max(max_reload_error,max(abs(original[key]-current[key]) for key in original))
assert max_reload_error < 1e-5, ('Fresh object reload changed scores',max_reload_error)
assert not attempted_connections, 'A package component attempted network access'
print(json.dumps({'status':'passed','software_checks_passed':True,
    'device':'cpu','encoder_dtype':'float32','head_dtype':'float32',
    'network_attempts':len(attempted_connections),'fresh_empty_hf_cache':True,
    'candidate_permutation_max_error':max_permutation_error,
    'additional_candidate_max_error':max_extra_candidate_error,
    'independent_reload_max_error':max_reload_error,
    'overlength_state_rejected':True,'overlength_candidate_rejected':True,
    'score_semantics':semantics,'examples':outputs,
    'illustrative_examples_matching_expectation':sum(item['illustrative_top_matches_expectation'] for item in outputs),
    'illustrative_example_count':len(outputs),
    'verification_seconds':time.monotonic()-started,
    'quality_claim':'Software assertions do not require correct illustrative decisions. The three demonstration outcomes are reported separately; they are not a benchmark or general-accuracy estimate.',
    'device_limits':'CPU serving is verified here. MPS behavior and CUDA-to-CPU ranking agreement are not established by this check.'},allow_nan=False))
'''


def run(seconds=450):
    """Modal CPU entry point; only metadata and small artifacts are returned."""
    import modal
    if modal.is_local(): raise RuntimeError("Joint package verification must run on Modal CPU, not the user's Mac")
    if not 120 <= seconds <= 1200: raise ValueError("Package allocation must be120–1200 seconds")
    from common import MODEL_REVISION
    started = time.monotonic(); deadline = started+seconds
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    base = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
    output = root / "package_joint"; output.mkdir(parents=True,exist_ok=True)
    bundle = root / "local-package-models/joint"; bundle.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).resolve().parent
    training_path = root / "joint/metrics.json"
    training = json.loads(training_path.read_text())
    if training.get("status") != "complete": raise RuntimeError("Joint training has not completed its bounded run")
    manifest = training["manifest"]
    if manifest["format"] != "gemmadecision-joint-v1" or not manifest.get("encoder_frozen"):
        raise RuntimeError("Unexpected joint training architecture")
    if manifest["head_config"] != {"hidden":640,"width":512}:
        raise RuntimeError("The recorded scalar head configuration differs from the serving contract")
    if manifest["separator"] != "\n\nCandidate action:\n": raise RuntimeError("Joint input-format mismatch")
    if manifest["base_revision"] != MODEL_REVISION: raise RuntimeError("Joint backbone revision mismatch")
    head = root / "joint/best/head.safetensors"
    if _sha(head) != training["selected_head_sha256"]: raise RuntimeError("The selected scalar checkpoint changed since training verification")
    files = {}
    base_files = sorted(path for path in base.iterdir() if path.is_file() and path.suffix in {".json",".safetensors",".model"})
    if not {"config.json","model.safetensors","tokenizer.json","tokenizer_config.json"}.issubset({path.name for path in base_files}):
        raise RuntimeError("Official Gemma base files are incomplete")
    for path in base_files:
        files[path.name] = {**_copy(path,bundle/path.name),"origin":"official_gemma_base"}
    for name in HELPERS:
        files[name] = {**_copy(source/name,bundle/name),"origin":"local_joint_inference_helper"}
        _copy(bundle/name,output/name)
    files["joint_head.safetensors"] = {**_copy(head,bundle/"joint_head.safetensors"),"origin":"selected_joint_scalar_head"}
    _copy(bundle/"joint_head.safetensors",output/"joint_head.safetensors")
    config = {"model_name":"GemmaDecision-270M-Joint","format":"gemmadecision-joint-v1","experimental":True,
              "base_model":"google/gemma-3-270m","base_revision":MODEL_REVISION,
              "head_config":manifest["head_config"],"head_file":"joint_head.safetensors",
              "max_state_tokens":2048,"max_action_tokens":768,
              "input_format":"schema_state(state,question) + '\\n\\nCandidate action:\\n' + candidate",
              "encoder_frozen":True,"base_weights_modified":False,"lora":False,
              "score_semantics":"Uncalibrated scalar ranking score; higher is preferred. Not a probability or confidence estimate.",
              "training_encoder_dtype":"bfloat16","training_embedding_cache_dtype":"float32",
              "cpu_mps_serving_dtype":"float32","selected_training_step":training["best"]["step"],
              "publication_on_hold":True}
    provenance = {"base_model":config["base_model"],"base_revision":MODEL_REVISION,
        "base_files":{name:item for name,item in files.items() if item["origin"]=="official_gemma_base"},
        "selected_arm":"joint","selected_head_sha256":_sha(head),"training_report_sha256":_sha(training_path),
        "training_manifest_sha256":_sha(root/"joint/manifest.json"),
        "selected_training_step":training["best"]["step"],"training_updates":training["steps"],
        "head_parameters":training["checks"]["head_parameters"],
        "base_weights_modified":False,"adapter_merged":False,"training_data_copied":False,
        "final_or_calibration_data_opened":False,"published_v2_unchanged":True,
        "selection_scope":"Development-selected local ranking prototype; final evaluation has not established general decision accuracy.",
        "terms":"Gemma weights retain Gemma terms. Include the existing Gemma terms, prohibited-use policy and upstream CLM attribution in local distribution."}
    examples = [{**example,"illustrative_expected_top":EXPECTED_DEMO_LABELS[example["name"]]} for example in EXAMPLES]
    for name,value in [("joint_config.json",config),("base-provenance.json",provenance),
                       ("verification-examples.json",examples),
                       ("example_request.json",{key:value for key,value in EXAMPLES[0].items() if key!="name"})]:
        _save(bundle/name,value);_copy(bundle/name,output/name)
        files[name]={"sha256":_sha(bundle/name),"bytes":(bundle/name).stat().st_size,"origin":"package_metadata"}
    payload = {"format":"local-joint-ranker-sha256-v1","selected_arm":"joint","files":files,
               "verification_reports_excluded_from_payload_hashes":True,
               "assembly":"Copy the small package artifacts and matching official Gemma files into one directory; verify every payload hash."}
    _save(bundle/"SHA256SUMS.json",payload);_copy(bundle/"SHA256SUMS.json",output/"SHA256SUMS.json")
    remaining = int(deadline-time.monotonic()-15)
    if remaining<60: raise RuntimeError("Insufficient time remains for independent CPU verification")
    print("Verifying joint ranker in a fresh offline CPU process; illustrative accuracy is separate from software checks",flush=True)
    with tempfile.TemporaryDirectory(prefix="joint-ranker-offline-") as empty:
        env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_DATASETS_OFFLINE":"1",
             "HF_HOME":str(Path(empty)/"empty-hf-cache"),"CUDA_VISIBLE_DEVICES":"",
             "OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
             "TOKENIZERS_PARALLELISM":"false","PYTHONNOUSERSITE":"1"}
        result=subprocess.run([sys.executable,"-I","-c",CPU_CHECK,str(bundle)],cwd=empty,env=env,
                              capture_output=True,text=True,timeout=remaining)
    if result.returncode: raise RuntimeError("Offline joint CPU verification failed: "+result.stderr[-2500:])
    try: verification=json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError): raise RuntimeError("Joint CPU verification returned no valid report") from None
    if verification.get("status")!="passed":raise RuntimeError("Offline joint software verification failed")
    _save(output/"offline-check.json",verification)
    for name,expected in files.items():
        if _sha(bundle/name)!=expected["sha256"]:raise RuntimeError("Inference modified a packaged artifact")
    report={"status":"complete","selected_arm":"joint","offline_cpu_verification":"passed",
            "full_cloud_bundle":str(bundle),"small_transfer_directory":str(output),
            "payload_bytes":sum(item["bytes"] for item in files.values()),
            "base_files_reusable_locally":{name:item for name,item in files.items() if item["origin"]=="official_gemma_base"},
            "verification":verification,"function_seconds":time.monotonic()-started,
            "publication_on_hold":True,"published_v2_unchanged":True,
            "quality_claim":"A functioning offline ranker; software checks and illustrative decision correctness are reported separately."}
    _save(output/"metrics.json",report)
    return report
