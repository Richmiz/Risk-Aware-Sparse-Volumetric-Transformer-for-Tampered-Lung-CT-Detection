"""Run actual RSVT training in Colab using the CT data already in project Drive."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from rsvt.training.proposed_run import run_proposed
from scripts.run_model_optimization import verify_run

PROJECT = Path('/content/drive/MyDrive/Research/LUNG-CT-SCAN')


def run(resume_run=None):
    import torch
    if not PROJECT.is_dir(): raise RuntimeError('Mount project Drive; outputs must not fall back to local storage')
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU Colab kernel in VS Code')
    parent = (PROJECT/'outputs/rsvt_training').resolve()
    output = (Path(resume_run).resolve() if resume_run else
              parent/('rsvt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    if not output.is_relative_to(parent) or output == parent: raise ValueError('Keep RSVT outputs inside its Drive training folder')
    print('RSVT OUTPUT DRIVE:', output, flush=True)
    result = run_proposed(PROJECT/'outputs/partitions/f03_20261110T013617_031412Z',
        ROOT/'configs/training/rsvt_real_ct.toml', output, resume=bool(resume_run),
        progress=lambda row: print(json.dumps(row), flush=True))
    receipt = verify_run(output)
    print('RSVT SAVED AND VERIFIED:', output, '| artifacts:', len(receipt['files']), flush=True)
    print('Validation:', json.dumps(result['validation_metrics']), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-run', default=None)
    run(parser.parse_args().resume_run)
