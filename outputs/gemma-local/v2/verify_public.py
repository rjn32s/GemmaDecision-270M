"""Verify a pinned public HF release in a fresh directory, without HF credentials."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
from huggingface_hub import snapshot_download


def verify(revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("An exact published commit is required")
    with tempfile.TemporaryDirectory(prefix="gemmadecision-public-") as directory:
        snapshot_download("rajan2k/GemmaDecision-270M", revision=revision,
                          local_dir=directory, token=False, max_workers=3)
        root = Path(directory)
        manifest = json.loads((root / "SHA256SUMS.json").read_text())
        for name, expected in manifest.items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid manifest path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Public artifact hash mismatch: " + name)
        # Confirm the downloaded executable helpers are exactly our reviewed code.
        for name in ["decision_model.py", "common.py", "clm_schema.py", "clm_heads.py"]:
            if (root / name).read_bytes() != (Path(__file__).parent / name).read_bytes():
                raise ValueError("Unexpected inference source: " + name)
        program = '''
import json, math
from decision_model import GemmaDecision
m=GemmaDecision('.',device='cuda')
p=m.rank('My invoice was charged twice.','Choose the responsible department.',
         {'billing':'Charges and refunds','technical':'Software errors'})
assert all(math.isfinite(x) for x in p.values()) and abs(sum(p.values())-1)<1e-5
print(json.dumps({'status':'passed','probabilities':p}))
'''
        result = subprocess.run(["python", "-c", program], cwd=root, text=True,
                                capture_output=True, timeout=300,
                                env={**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PYTHONPATH":str(root)})
        if result.returncode:
            raise RuntimeError("Published model load failed: " + result.stderr[-2000:])
        return {"status":"passed", "repo_id":"rajan2k/GemmaDecision-270M", "revision":revision,
                "files_verified":len(manifest), "fresh_anonymous_download":True,
                "offline_inference":json.loads(result.stdout.strip().splitlines()[-1])}
