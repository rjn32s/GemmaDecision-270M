"""Build and verify a practical offline ranker on a Modal CPU worker.

No execution, file access, or model loading occurs at import. The complete cloud
bundle is written under local-package-models/<arm>; package/ holds only the
small transfer artifacts, hashes and verification report. Local assembly can
reuse matching official Gemma files already available to the user.

This module never publishes, changes v2, merges adapters, or opens evaluation
data. Three examples check the interface and repeatability, not decision quality.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


HELPERS = ("deployment.py", "common.py", "clm_heads.py", "clm_schema.py")
EXAMPLES = [
    {"name": "support_routing",
     "state": "A customer says: The same card payment appears twice on my statement. Please help me investigate the duplicate charge.",
     "question": "Which support team should handle this request?",
     "candidates": {"payments": "Payments support: investigate charges, transfers and refunds.",
                    "technical": "Technical support: investigate app crashes and login errors.",
                    "delivery": "Delivery support: track shipments and missing parcels."}},
    {"name": "evidence_relation",
     "state": "Evidence: The notice says the shop closes at 18:00 today. Claim: The shop closes at 20:00 today.",
     "question": "Using only the evidence, how does the claim relate to the evidence?",
     "candidates": {"supported": "The evidence supports the claim.",
                    "unresolved": "The evidence leaves the claim unresolved.",
                    "contradicted": "The evidence contradicts the claim."}},
    {"name": "tool_choice",
     "state": "The user asks for the current weather in London. No current weather observations have been retrieved. Available tools can fetch live weather or perform arithmetic.",
     "question": "Which next action best addresses the user's request?",
     "candidates": {"weather": "Use the weather tool to retrieve current London weather.",
                    "calculator": "Use the calculator to multiply 17 by 23.",
                    "guess": "Answer with an assumed weather condition without retrieving current information."}},
]


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def _copy(source, destination):
    source = Path(source); destination = Path(destination)
    expected = _sha(source)
    if not destination.exists() or _sha(destination) != expected:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if _sha(temporary) != expected: raise RuntimeError(f"Copy checksum mismatch: {source.name}")
        temporary.replace(destination)
    return {"sha256": expected, "bytes": source.stat().st_size}


CPU_CHECK = r'''
import gc, json, math, os, socket, sys, time
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(bundle))
attempted_connections = []
def deny_network(*args, **kwargs):
    attempted_connections.append(True)
    raise RuntimeError('Network access is disabled during offline package verification')
socket.create_connection = deny_network
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from deployment import GemmaDecisionRanker
examples = json.loads((bundle/'verification-examples.json').read_text())
started = time.monotonic()
model = GemmaDecisionRanker(bundle, device='cpu', candidate_cache_size=0)
assert model.device.type == 'cpu'
assert next(model.encoder.parameters()).dtype == torch.float32
outputs = []
max_permutation_error = 0.
for example in examples:
    tick = time.monotonic()
    ranking = model.rank(example['state'], example['candidates'], example['question'])
    seconds = time.monotonic()-tick
    scores = {item['candidate']: item['score'] for item in ranking}
    assert set(scores) == set(example['candidates'])
    assert all(math.isfinite(value) for value in scores.values())
    assert all(ranking[i]['score'] >= ranking[i+1]['score'] for i in range(len(ranking)-1))
    # Disable the candidate cache above so this also checks actual re-encoding.
    reversed_candidates = dict(reversed(list(example['candidates'].items())))
    reverse = model.rank(example['state'], reversed_candidates, example['question'])
    reverse_scores = {item['candidate']: item['score'] for item in reverse}
    error = max(abs(scores[key]-reverse_scores[key]) for key in scores)
    assert error < 1e-4, ('candidate permutation changed scores', error)
    max_permutation_error = max(max_permutation_error, error)
    outputs.append({'name': example['name'], 'ranking': ranking,
                    'cpu_seconds': seconds, 'candidate_order_max_error': error})

for mode in ('state', 'candidate'):
    try:
        if mode == 'state':
            model.rank('word '*(model.max_state_tokens+32), {'a':'Accept.','b':'Reject.'})
        else:
            model.rank('A short state.', {'a':'word '*(model.max_action_tokens+32),'b':'Reject.'})
    except ValueError as exc:
        assert 'limit' in str(exc)
    else: raise AssertionError('Overlength '+mode+' was silently accepted')

semantics = model.score_semantics
del model
gc.collect()
# A fresh object loads only the saved standalone folder. Enable the normal
# serving cache here to verify that it agrees with the uncached first object.
reloaded = GemmaDecisionRanker(bundle, device='cpu', candidate_cache_size=128)
max_reload_error = 0.
max_cache_error = 0.
for example, previous in zip(examples, outputs):
    current = reloaded.rank(example['state'], example['candidates'], example['question'])
    cached = reloaded.rank(example['state'], example['candidates'], example['question'])
    before = {x['candidate']:x['score'] for x in previous['ranking']}
    after = {x['candidate']:x['score'] for x in current}
    after_cached = {x['candidate']:x['score'] for x in cached}
    max_reload_error = max(max_reload_error, max(abs(before[key]-after[key]) for key in before))
    max_cache_error = max(max_cache_error, max(abs(after[key]-after_cached[key]) for key in before))
assert max_reload_error < 1e-4, ('reload mismatch', max_reload_error)
assert max_cache_error < 1e-4, ('cache mismatch', max_cache_error)
assert not attempted_connections, 'A component attempted network access'
print(json.dumps({'status':'passed','device':'cpu','dtype':'float32',
    'network_attempts':len(attempted_connections),'offline_environment':True,
    'fresh_empty_hf_cache':True,'candidate_permutation_max_error':max_permutation_error,
    'independent_reload_max_error':max_reload_error,'candidate_cache_max_error':max_cache_error,
    'overlength_state_rejected':True,'overlength_candidate_rejected':True,
    'score_semantics':semantics,'examples':outputs,'verification_seconds':time.monotonic()-started,
    'quality_claim':'No accuracy assertion: examples verify local loading, ranking outputs and repeatability.',
    'device_limits':'CPU serving verified here; MPS and CUDA-to-CPU numerical agreement are not certified by this check.'}, allow_nan=False))
'''


def run(arm="frozen", seconds=300):
    """Modal CPU entry point. The return value contains only small metadata."""
    import modal
    if modal.is_local(): raise RuntimeError("Package verification must run on a Modal CPU worker, not the user's Mac")
    if arm not in {"frozen", "frozen_extension", "lora", "v2_unchanged", "v2_hard"}:
        raise ValueError("Select frozen, frozen_extension, lora, v2_unchanged, or v2_hard")
    if not 120 <= seconds <= 1200: raise ValueError("Package time budget must be between 120 and 1200 seconds")
    from common import HEAD_CONFIG, MODEL_REVISION
    started = time.monotonic(); deadline = started + seconds
    root = Path(os.environ.get("V3_ROOT", "/experiment/work/decision-v3"))
    base = Path(os.environ.get("GEMMA_MODEL_PATH", "/experiment/work/model"))
    v2 = Path(os.environ.get("V2_ROOT", "/experiment/work/decision-v2"))
    output = root / "package"; output.mkdir(parents=True, exist_ok=True)
    bundle = root / "local-package-models" / arm; bundle.mkdir(parents=True, exist_ok=True)
    checkpoint = v2 / "train" if arm in {"v2_unchanged", "v2_hard"} else root / arm / "best"
    head_name = {"v2_unchanged": "agent.safetensors", "v2_hard": "hard.safetensors"}.get(arm, "heads.safetensors")
    head = checkpoint / head_name
    if not head.is_file(): raise RuntimeError("Chosen head checkpoint does not exist")
    source = Path(__file__).resolve().parent

    files = {}
    base_files = sorted(p for p in base.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".model"})
    if not {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}.issubset({p.name for p in base_files}):
        raise RuntimeError("The original standalone Gemma base files are incomplete")
    for path in base_files:
        files[path.name] = {**_copy(path, bundle / path.name), "origin": "official_gemma_base"}
    for name in HELPERS:
        metadata = _copy(source / name, bundle / name)
        _copy(bundle / name, output / name)
        files[name] = {**metadata, "origin": "local_inference_helper"}
    files["decision_heads.safetensors"] = {**_copy(head, bundle / "decision_heads.safetensors"), "origin": arm + "_head_checkpoint"}
    _copy(bundle / "decision_heads.safetensors", output / "decision_heads.safetensors")
    adapter = None
    if arm == "lora":
        adapter = checkpoint / "adapter.safetensors"
        files["adapter.safetensors"] = {**_copy(adapter, bundle / "adapter.safetensors"), "origin": "unmerged_runtime_lora"}
        _copy(bundle / "adapter.safetensors", output / "adapter.safetensors")
    elif (output / "adapter.safetensors").exists():
        # Remove only this packager's obsolete transfer artifact when selecting
        # a different arm. No model or training checkpoint is changed.
        (output / "adapter.safetensors").unlink()
    config = {"model_name": "GemmaDecision-270M", "format": "local-clm-ranker-v1", "experimental": True,
              "base_model": "google/gemma-3-270m", "base_revision": MODEL_REVISION,
              "selected_arm": arm, "head_config": HEAD_CONFIG, "head_file": "decision_heads.safetensors",
              "adapter_mode": "runtime" if adapter is not None else "none",
              "max_state_tokens": 2048, "max_action_tokens": 768,
              "score_semantics": "Uncalibrated scaled cosine ranking score; higher is preferred. Not a probability or confidence estimate.",
              "trained_dtype": "bfloat16", "cpu_mps_serving_dtype": "float32",
              "publication_on_hold": True}
    if adapter is not None:
        # Read the saved training configuration so packaging cannot silently
        # invent an adapter architecture. No training module is bundled/imported.
        training_report = json.loads((root / "lora/metrics.json").read_text())
        config["lora_config"] = training_report["lora_config"]
        if not config["lora_config"]: raise RuntimeError("Saved LoRA configuration is missing")
        config["adapter_file"] = "adapter.safetensors"
    provenance = {"base_model": config["base_model"], "base_revision": MODEL_REVISION,
                  "base_files": {name: item for name, item in files.items() if item["origin"] == "official_gemma_base"},
                  "selected_arm": arm, "head_sha256": files["decision_heads.safetensors"]["sha256"],
                  "adapter_sha256": files.get("adapter.safetensors", {}).get("sha256"),
                  "adapter_merged": False, "base_weights_modified": False,
                  "training_data_copied": False, "final_or_calibration_data_opened": False,
                  "selection_scope": "Practical local ranking prototype; checkpoint choice belongs to the reviewed development comparison.",
                  "terms": "Gemma weights remain subject to Gemma terms. Keep the existing Gemma terms, prohibited-use policy and CLM attribution with local distribution.",
                  "published_v2_unchanged": True}
    for name, value in (("decision_config.json", config), ("base-provenance.json", provenance),
                        ("verification-examples.json", EXAMPLES),
                        ("example_request.json", {k: v for k, v in EXAMPLES[0].items() if k != "name"})):
        _save(bundle / name, value); _copy(bundle / name, output / name)
        files[name] = {"sha256": _sha(bundle / name), "bytes": (bundle / name).stat().st_size, "origin": "package_metadata"}
    manifest = {"format": "local-clm-ranker-sha256-v1", "selected_arm": arm, "files": files,
                "verification_reports_excluded_from_payload_hashes": True,
                "assembly": "Copy the small package artifacts and matching official Gemma files into one directory. Every payload file must match this manifest."}
    _save(bundle / "SHA256SUMS.json", manifest); _copy(bundle / "SHA256SUMS.json", output / "SHA256SUMS.json")

    remaining = int(deadline - time.monotonic() - 15)
    if remaining < 60: raise RuntimeError("Insufficient time left for offline CPU verification")
    print(f"Verifying {arm} as an offline standalone CPU ranker", flush=True)
    with tempfile.TemporaryDirectory(prefix="ranker-offline-") as empty:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HOME": str(Path(empty) / "empty-hf-cache"), "CUDA_VISIBLE_DEVICES": "",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1"}
        # Isolated Python + empty working directory excludes research-module and
        # cached-model fallbacks. The subprocess inserts only the bundle itself.
        result = subprocess.run([sys.executable, "-I", "-c", CPU_CHECK, str(bundle)], cwd=empty, env=env,
                                capture_output=True, text=True, timeout=remaining)
    if result.returncode:
        raise RuntimeError("Offline CPU package verification failed: " + result.stderr[-2500:])
    try: verification = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError): raise RuntimeError("CPU verification returned no valid report") from None
    if verification.get("status") != "passed": raise RuntimeError("Offline verification did not pass")
    _save(output / "offline-check.json", verification)
    # Recheck copied artifacts after inference; a loader must not mutate them.
    for name, expected in files.items():
        if _sha(bundle / name) != expected["sha256"]: raise RuntimeError("Inference modified a packaged artifact")
    report = {"status": "complete", "selected_arm": arm, "offline_cpu_verification": "passed",
              "full_cloud_bundle": str(bundle), "small_transfer_directory": str(output),
              "payload_bytes": sum(item["bytes"] for item in files.values()),
              "base_files_reusable_locally": {name: item for name, item in files.items() if item["origin"] == "official_gemma_base"},
              "verification": verification, "function_seconds": time.monotonic() - started,
              "publication_on_hold": True, "published_v2_unchanged": True,
              "quality_claim": "This verifies a usable local ranking interface, not expert performance or JEV parity."}
    _save(output / "metrics.json", report)
    return report
