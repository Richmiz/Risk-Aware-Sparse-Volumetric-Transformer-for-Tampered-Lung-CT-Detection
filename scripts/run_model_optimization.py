"""Separate Colab training job. Existing audit/review notebooks are unchanged."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rsvt.data.acquisition import sha256_file
from rsvt.training.phase4 import run_phase4

PROJECT = Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')
F03_RELATIVE = 'outputs/partitions/f03_20261110T013617_031412Z'


def verify_run(directory):
    """Read back reports and existing checkpoint receipts before success."""
    directory = Path(directory)
    records = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.csv', '.md', '.pt'}:
            continue
        if path.name == 'persistence_receipt.json': continue
        digest = sha256_file(path)
        if path.suffix == '.json': json.loads(path.read_text(encoding='utf-8'))
        if path.suffix == '.pt':
            checkpoint_receipt = json.loads(path.with_name(path.name+'.json').read_text(encoding='utf-8'))
            if checkpoint_receipt['sha256'] != digest or checkpoint_receipt['bytes'] != path.stat().st_size:
                raise ValueError('Checkpoint readback mismatch: '+path.name)
        records[path.relative_to(directory).as_posix()] = {'sha256': digest, 'bytes': path.stat().st_size}
    if 'summary.json' not in records: raise ValueError('Missing run summary')
    receipt = {'schema_version': 'model_optimization_persistence_v1', 'files': records,
               'created_utc': datetime.now(timezone.utc).isoformat()}
    target = directory/'persistence_receipt.json'
    target.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    if json.loads(target.read_text(encoding='utf-8')) != receipt: raise OSError('Receipt readback failed')
    # Re-read derived reports against the receipt. Checkpoints were already
    # compared to the writer's own stored checksums above.
    for name, record in records.items():
        if not name.endswith('.pt') and sha256_file(directory/name) != record['sha256']:
            raise OSError('Report readback mismatch: '+name)
    return receipt


def run(stage='control'):
    import torch
    if stage not in {'control', 'full'}: raise ValueError('Choose control or full')
    if not PROJECT.is_dir(): raise RuntimeError('Mount Google Drive in Colab first; no local-output fallback.')
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU Colab kernel; this job does not fall back to CPU.')
    f03 = PROJECT/F03_RELATIVE
    if not (f03/'manifest.draft.json').is_file(): raise FileNotFoundError('Pinned F03 manifest not found in Drive')
    profile = 'baseline_optimized_control' if stage == 'control' else 'baseline_optimized'
    output = PROJECT/'outputs/model_optimization'/(
        stage+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    print('Output Drive:', output, flush=True)
    result = run_phase4(f03, ROOT/f'configs/training/{profile}.toml', output,
                        progress=lambda event: print(json.dumps(event), flush=True))
    receipt = verify_run(output)
    if not result.get('ok'): raise RuntimeError('Training incomplete; inspect saved summary: '+str(output))
    print('SAVED AND VERIFIED:', output, '| files:', len(receipt['files']), flush=True)
    if stage == 'control':
        passed = result['train_fitting_all_models_passed_at_last_epoch']
        print('Fitting control passed for both models:', passed, flush=True)
        print('Full run is a separate invocation; no automatic launch.', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('control', 'full'), default='control')
    run(parser.parse_args().stage)
