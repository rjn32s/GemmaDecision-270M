"""Freeze one reviewed development choice before opening calibration/final labels."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def run(arm):
    from common import MODEL_REVISION, save
    if arm not in {'joint', 'frozen_extension'}:
        raise ValueError('Select the reviewed joint or separate-encoder checkpoint explicitly')
    root = Path(os.environ.get('V3_ROOT', '/experiment/work/decision-v3'))
    base = Path(os.environ.get('GEMMA_MODEL_PATH', '/experiment/work/model'))
    v2 = Path('/experiment/work/decision-v2')
    if (root / 'final-evaluation/final-labels-opened.json').exists():
        raise RuntimeError('Final labels are already open; cannot change selection')
    report_path = root / arm / 'metrics.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete' or report['steps'] != 3000:
        raise RuntimeError('The selected experiment must finish its matched exposure')
    comparison_path = root / 'extension_compare/metrics.json'
    comparison = json.loads(comparison_path.read_text())
    if comparison['status'] != 'complete':
        raise RuntimeError('Full-development comparison is incomplete')
    manifest = report['manifest']
    if manifest['development_data_sha256'] != comparison['development_data_sha256']:
        raise RuntimeError('Development populations differ')
    checkpoint = root / arm / 'best'
    head_file = 'head.safetensors' if arm == 'joint' else 'heads.safetensors'
    base_provenance = json.loads((root / 'package/base-provenance.json').read_text())
    base_files = {name: item['sha256'] for name, item in base_provenance['base_files'].items()}
    for name, expected in base_files.items():
        if sha(base / name) != expected:
            raise RuntimeError('The pinned backbone or tokenizer changed')
    audit = json.loads((root / 'audit/metrics.json').read_text())
    if not audit['passed_structural'] or audit['near_duplicate_review_required']:
        raise RuntimeError('Dataset audit has not passed')
    data_sha = {name: audit['split_file_sha256'][name] for name in ('calibration.jsonl', 'final.jsonl')}
    # Binary hashes only: no decoding of labels before the recipe is frozen.
    for name, expected in data_sha.items():
        if sha(root / 'data' / name) != expected:
            raise RuntimeError('Evaluation data changed after the audit')
    selected = {'arm': 'joint' if arm == 'joint' else 'frozen',
                'checkpoint_dir': f'{arm}/best', 'heads_sha256': sha(checkpoint / head_file)}
    if arm == 'joint':
        selected['head_config'] = {'hidden': 640, 'width': 512}
    recipe = {
        'schema_version': 1, 'selected': selected, 'data_sha256': data_sha,
        'base_revision': MODEL_REVISION, 'base_files': base_files,
        'v2_agent_heads_sha256': sha(v2 / 'train/agent.safetensors'),
        'limits': {'state': 2048, 'candidate': 768},
        'development_repeatability': {'passed': False, 'seeds': [2704203],
            'evidence_path': f'{arm}/metrics.json', 'evidence_sha256': sha(report_path),
            'note': 'One seed only; no claim of replication across seeds.'},
        'selection_rationale': f'Reviewed full-development comparison selected {arm} for a practical 270M local ranker. This is a prototype choice, not a claim of general superiority or JEV parity. The final result is diagnostic and will be preserved even if unfavorable.',
        'development_evidence': {'selected_metrics_sha256': sha(report_path),
            'separate_comparison_sha256': sha(comparison_path),
            'selected_step': report['best']['step'], 'selected_development_macro': report['best']['score']},
        'scope': 'User prioritizes a useful locally deployable 270M ranker; the earlier universal-improvement gate is historical.',
        'publication_on_hold': True,
    }
    destination = root / 'frozen-recipe.json'
    if destination.exists() and json.loads(destination.read_text()) != recipe:
        raise RuntimeError('A different choice is already frozen; refuse overwrite')
    save(destination, recipe)
    save(root / 'freeze/frozen-recipe.json', recipe)
    return {'status': 'complete', 'selected_arm': arm, 'recipe_sha256': sha(destination),
            'calibration_labels_opened': False, 'final_labels_opened': False,
            'publication_on_hold': True}
