"""Package the frozen fully tuned v4 export and verify it on cloud CPU.

No publication is performed. Original Gemma, v3 and published v2 are untouched.
The selected encoder is modified; original base weights cannot substitute for it.
Only metric reports, never evaluation labels, may enter the local-use package.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from evaluate_v4 import save, sha, paths, inside, validate_recipe, HEAD_CONFIG

HELPERS = ("joint_deployment.py", "common.py", "clm_schema.py", "clm_heads.py")
EXAMPLES = [
    {"name": "support_routing",
     "state": "A customer says: The same card payment appears twice on my statement. Please help me investigate the duplicate charge.",
     "question": "Which support team should handle this request?",
     "candidates": {"payments": "Payments support: investigate charges, transfers and refunds.",
                    "technical": "Technical support: investigate app crashes and login errors.",
                    "delivery": "Delivery support: track shipments and missing parcels."},
     "illustrative_expected_top": "payments"},
    {"name": "evidence_relation",
     "state": "Evidence: The notice says the shop closes at 18:00 today. Claim: The shop closes at 20:00 today.",
     "question": "Using only the evidence, how does the claim relate to the evidence?",
     "candidates": {"supported": "The evidence supports the claim.",
                    "unresolved": "The evidence leaves the claim unresolved.",
                    "contradicted": "The evidence contradicts the claim."},
     "illustrative_expected_top": "contradicted"},
    {"name": "tool_choice",
     "state": "The user asks for the current weather in London. No current weather observations have been retrieved. Available tools can fetch live weather or perform arithmetic.",
     "question": "Which next action best addresses the user's request?",
     "candidates": {"weather": "Use the weather tool to retrieve current London weather.",
                    "calculator": "Use the calculator to multiply 17 by 23.",
                    "guess": "Answer with an assumed weather condition without retrieving current information."},
     "illustrative_expected_top": "weather"},
]


def _copy(source, destination):
    source, destination = Path(source), Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha(source)
    if not destination.exists() or sha(destination) != expected:
        temporary = destination.with_suffix(destination.suffix + ".tmp"); shutil.copyfile(source, temporary)
        if sha(temporary) != expected: raise RuntimeError("Copy checksum mismatch")
        temporary.replace(destination)
    return {"sha256": expected, "bytes": source.stat().st_size}


CPU_CHECK = r'''
import gc, json, math, socket, sys, time
from pathlib import Path
bundle = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(bundle))
attempts = []
def deny_network(*args, **kwargs):
    attempts.append(True)
    raise RuntimeError('Network disabled during offline verification')
socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
import torch
torch.set_num_threads(2); torch.set_num_interop_threads(1)
from joint_deployment import GemmaJointRanker
examples = json.loads((bundle/'verification-examples.json').read_text())
started = time.monotonic(); model = GemmaJointRanker(bundle, device='cpu')
load_seconds = time.monotonic()-started
assert next(model.encoder.parameters()).dtype == torch.float32
assert all(not p.requires_grad for p in model.encoder.parameters())
assert all(p.dtype == torch.float32 for p in model.head.parameters())
outputs = []; max_order = 0.; max_extra = 0.
for example in examples:
    tick = time.monotonic()
    ranking = model.rank(example['state'], example['candidates'], example['question'])
    latency = time.monotonic()-tick
    values = {x['candidate']: x['score'] for x in ranking}
    assert set(values) == set(example['candidates']) and all(math.isfinite(x) for x in values.values())
    assert all(ranking[i]['score'] >= ranking[i+1]['score'] for i in range(len(ranking)-1))
    reverse = model.rank(example['state'], dict(reversed(list(example['candidates'].items()))), example['question'])
    reversed_values = {x['candidate']: x['score'] for x in reverse}
    order_error = max(abs(values[k]-reversed_values[k]) for k in values)
    expanded = dict(example['candidates']); expanded['archive'] = 'Archive this request without processing it.'
    extra_values = {x['candidate']:x['score'] for x in model.rank(example['state'], expanded, example['question'])}
    extra_error = max(abs(values[k]-extra_values[k]) for k in values)
    assert order_error < 1e-5 and extra_error < 1e-5
    max_order = max(max_order, order_error); max_extra = max(max_extra, extra_error)
    outputs.append({'name':example['name'], 'ranking':ranking, 'cpu_seconds':latency,
        'illustrative_expected_top':example['illustrative_expected_top'],
        'illustrative_top_matches_expectation':ranking[0]['candidate']==example['illustrative_expected_top']})
for mode in ('state','candidate'):
    try:
        if mode=='state': model.rank('word '*(model.max_state_tokens+32), {'a':'Accept.','b':'Reject.'})
        else: model.rank('A short state.', {'a':'word '*(model.max_action_tokens+32),'b':'Reject.'})
    except ValueError as error:
        assert 'limit' in str(error)
    else: raise AssertionError('Overlength '+mode+' silently accepted')
semantics = model.score_semantics
del model; gc.collect()
reloaded = GemmaJointRanker(bundle, device='cpu'); reload_error = 0.
for example, previous in zip(examples, outputs):
    values = {x['candidate']:x['score'] for x in reloaded.rank(example['state'],example['candidates'],example['question'])}
    old = {x['candidate']:x['score'] for x in previous['ranking']}
    reload_error = max(reload_error, max(abs(values[k]-old[k]) for k in old))
assert reload_error < 1e-5 and not attempts
print(json.dumps({'status':'passed','software_checks_passed':True,'device':'cpu',
    'encoder_dtype':'float32','head_dtype':'float32','cpu_threads':2,
    'network_attempts':len(attempts),'fresh_empty_hf_cache':True,
    'candidate_permutation_max_error':max_order,'additional_candidate_max_error':max_extra,
    'independent_reload_max_error':reload_error,'overlength_state_rejected':True,'overlength_candidate_rejected':True,
    'score_semantics':semantics,'cold_model_load_seconds':load_seconds,
    'examples':outputs,'illustrative_correct':sum(x['illustrative_top_matches_expectation'] for x in outputs),
    'illustrative_count':len(outputs),'verification_seconds':time.monotonic()-started,
    'latency_scope':'Observed Modal CPU, two threads, these exact short examples; not a Mac or general latency estimate',
    'quality_claim':'Software checks do not assert correct decisions. These exposed illustrative examples are not a fresh benchmark.',
    'device_limits':'CPU verified only; MPS and CUDA-to-CPU ranking agreement are not established by this check.'},allow_nan=False))
'''


def run(seconds=450):
    import modal
    if modal.is_local(): raise RuntimeError("Full-model package verification must run on Modal CPU")
    if not 120 <= seconds <= 1200: raise ValueError("Package allocation must be 120–1200 seconds")
    started = time.monotonic(); deadline = started + seconds
    root, base, _ = paths(); recipe, identity = validate_recipe()
    checkpoint = inside(root, recipe["selected"]["checkpoint_dir"])
    output = root / "package"; bundle = output / "full-model"
    output.mkdir(parents=True, exist_ok=True); bundle.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent; files = {}
    # The checkpoint is a complete BF16 encoder export plus tokenizer and FP32 head.
    # Copy only the required model payload, never optimizer, labels, or training rows.
    for name, digest in recipe["selected"]["files"].items():
        if name in {"joint_config.json", "head.safetensors", "joint_head.safetensors"}: continue
        path = inside(checkpoint, name)
        if sha(path) != digest: raise RuntimeError("Frozen full-model payload changed")
        if name not in {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                        "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json"}: continue
        files[name] = {**_copy(path, bundle / name), "origin": "selected_full_tuning_export"}
    if files["model.safetensors"]["sha256"] == sha(base / "model.safetensors"):
        raise RuntimeError("Package unexpectedly contains unchanged original encoder weights")
    head = checkpoint / recipe["selected"]["head_file"]
    files["joint_head.safetensors"] = {**_copy(head, bundle / "joint_head.safetensors"), "origin": "selected_full_tuning_head"}
    for name in HELPERS:
        files[name] = {**_copy(source / name, bundle / name), "origin": "shared_offline_serving_source"}
    evaluation_path = root / "final-evaluation/metrics.json"; evaluation = None
    if evaluation_path.exists():
        candidate = json.loads(evaluation_path.read_text())
        if candidate.get("status") == "complete":
            if candidate.get("identity") != identity: raise RuntimeError("Evaluation belongs to another selected checkpoint")
            evaluation = candidate
            files["evaluation.json"] = {**_copy(evaluation_path, bundle / "evaluation.json"), "origin": "aggregate_fresh_evaluation_report"}
    config = {"model_name": "GemmaDecision-270M", "format": "gemmadecision-full-joint-v4",
        "experimental": evaluation is None or evaluation.get("release_status") != "targets_met",
        "base_model": recipe["base_model"], "base_revision": recipe["base_revision"],
        "head_config": HEAD_CONFIG, "head_file": "joint_head.safetensors",
        "max_state_tokens": recipe["limits"]["state"], "max_action_tokens": recipe["limits"]["candidate"],
        "input_format": "schema_state(state,question) + '\\n\\nCandidate action:\\n' + candidate",
        "encoder_frozen_during_training": False, "base_weights_modified": True, "base_weights_unchanged": False,
        "adapter": None, "export_encoder_dtype": "bfloat16", "cpu_mps_serving_dtype": "float32",
        "score_semantics": "Uncalibrated scalar ranking score; higher is preferred. Not a probability or confidence estimate.",
        "qualified_capabilities": evaluation.get("qualified_capabilities", []) if evaluation else [],
        "evaluation_status": evaluation.get("release_status") if evaluation else "pending",
        "publication_authorized": True, "benchmark_issue_submission_authorized": False}
    provenance = {"base_model": recipe["base_model"], "initial_revision": recipe["base_revision"],
        "initial_base_files": recipe["initial_base_files"], "selected_checkpoint_files": recipe["selected"]["files"],
        "frozen_recipe_sha256": identity["recipe_sha256"], "data_sha256": recipe["data_sha256"],
        "training_report_sha256": recipe["training_report_sha256"], "sources_sha256": recipe["sources_sha256"],
        "selection_rationale": recipe["selection_rationale"], "seeds": recipe["seeds"],
        "base_weights_modified": True, "base_weights_unchanged": False,
        "training_data_copied": False, "evaluation_labels_copied": False,
        "published_v2_and_v3_unchanged": True,
        "terms": "Gemma weights retain Gemma terms and prohibited-use policy; include upstream attribution before distribution."}
    for name, value in (("joint_config.json", config), ("base-provenance.json", provenance),
                        ("verification-examples.json", EXAMPLES),
                        ("example_request.json", {k: v for k, v in EXAMPLES[0].items() if k not in {"name", "illustrative_expected_top"}})):
        save(bundle / name, value); files[name] = {"sha256": sha(bundle / name), "bytes": (bundle / name).stat().st_size, "origin": "package_metadata"}
    readme = """# GemmaDecision-270M\n\nExperimental local candidate ranker using a fully tuned Gemma-270M encoder and scalar decision head. It ranks supplied choices; it does not generate answers.\n\nRun after installing PyTorch, Transformers and safetensors:\n\n```sh\npython joint_deployment.py --model . --input example_request.json --device cpu\n```\n\nUse `GemmaDecisionRanker(path, device='cpu').rank(state, candidates, question)` from `joint_deployment.py`. Candidates are a list of strings or a label-to-description mapping. Higher raw scores rank first. Scores are not calibrated confidence and cannot be compared across requests.\n\nThe complete modified `model.safetensors` is required; original Gemma weights cannot replace it. Read `evaluation.json` when present for fresh task results and `offline-check.json` for actual software checks and illustrative failures. Input limits are 2048 state tokens and 768 tokens per candidate; overlength input raises an error. CPU verification does not certify Mac MPS behavior.\n\nThis package makes no JEV-parity or general decision-reliability claim. See `base-provenance.json` for source hashes and initial model identity. Gemma terms and applicable attribution accompany distribution.\n"""
    (bundle / "README.md").write_text(readme)
    files["README.md"] = {"sha256": sha(bundle / "README.md"), "bytes": (bundle / "README.md").stat().st_size, "origin": "usage_documentation"}
    remaining = int(deadline - time.monotonic() - 15)
    if remaining < 60: raise RuntimeError("Insufficient remaining time for CPU verification")
    print("Verifying complete modified encoder in a fresh offline CPU process", flush=True)
    with tempfile.TemporaryDirectory(prefix="v4-offline-") as empty:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(Path(empty) / "empty-hf-cache"), "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}
        result = subprocess.run([sys.executable, "-I", "-c", CPU_CHECK, str(bundle)], cwd=empty, env=env,
                                capture_output=True, text=True, timeout=remaining)
    if result.returncode: raise RuntimeError("Offline CPU verification failed: " + result.stderr[-2500:])
    try: verification = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError): raise RuntimeError("Offline verification returned no report") from None
    if verification.get("status") != "passed": raise RuntimeError("Offline software checks did not pass")
    save(bundle / "offline-check.json", verification)
    files["offline-check.json"] = {"sha256": sha(bundle / "offline-check.json"), "bytes": (bundle / "offline-check.json").stat().st_size, "origin": "cpu_verification_report"}
    for name, item in files.items():
        if sha(bundle / name) != item["sha256"]: raise RuntimeError("Inference changed packaged payload")
    manifest = {"format": "gemmadecision-v4-sha256-v1", "files": files,
                "base_weights_modified": True, "full_modified_encoder_required": True,
                "assembly": "Download every listed file, including the new model.safetensors, and verify hashes."}
    save(bundle / "SHA256SUMS.json", manifest)
    for path in bundle.iterdir():
        if path.is_file() and path.name != "model.safetensors": _copy(path, output / path.name)
    report = {"status": "complete", "full_cloud_bundle": str(bundle), "small_transfer_directory": str(output),
        "model_cloud_path": str(bundle / "model.safetensors"), "model_sha256": files["model.safetensors"]["sha256"],
        "payload_bytes": sum(item["bytes"] for item in files.values()), "base_weights_modified": True,
        "base_weights_unchanged": False, "offline_cpu_verification": "passed", "verification": verification,
        "evaluation_status": config["evaluation_status"], "qualified_capabilities": config["qualified_capabilities"],
        "publication_authorized": True, "benchmark_issue_submission_authorized": False,
        "seconds": time.monotonic() - started}
    save(output / "metrics.json", report); return report
