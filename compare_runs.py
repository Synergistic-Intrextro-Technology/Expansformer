#!/usr/bin/env python3
"""Inspect comparable Expansformer v0.3 ablation runs without exaggerating evidence.

Only compares held-out cross-entropy when corpus split and tokenizer bytes match.
This tool does not prove statistical significance, speedup, reasoning or factuality.
"""
import argparse
import json
from pathlib import Path
import torch


def summarize(directory: Path):
    ckpt = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=True)
    if ckpt.get('format') != 'expansformer-from-scratch-v0.3':
        raise ValueError(f'{directory}: expected v0.3 checkpoint')
    rows = [json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines() if x.strip()]
    if not rows or ckpt['step'] != ckpt['settings']['steps'] or rows[-1].get('step') != ckpt['step']:
        raise ValueError(f'{directory}: experiment has not completed its planned steps')
    evaluations = [row for row in rows if row.get('val_loss') is not None]
    if not evaluations:
        raise ValueError(f'{directory}: no measured validation loss')
    best = min(evaluations, key=lambda r:r['val_loss'])
    return {'path': str(directory), 'fingerprint': ckpt['fingerprint'],
            'tokenizer_sha256': ckpt['tokenizer_sha256'],
            'steps': ckpt['settings']['steps'], 'settings': ckpt['settings'],
            'parameters': ckpt.get('parameter_count'), 'eval_windows': ckpt['settings']['eval_windows'],
            'seed': ckpt['settings']['seed'], 'config': ckpt['config'],
            'research_config': ckpt['research_config'], 'best_step': best['step'],
            'val_loss': best['val_loss'], 'val_bits_per_byte': best['val_bits_per_byte'],
            'evaluated_tokens': best['evaluated_tokens']}


def compare(a: Path, b: Path):
    before, after = summarize(a), summarize(b)
    for field in ('fingerprint', 'tokenizer_sha256', 'eval_windows', 'evaluated_tokens', 'seed'):
        if before[field] != after[field]:
            raise ValueError(f'Not comparable: {field} differs; preserve same held-out data and tokenizer')
    if before['config']['context_length'] != after['config']['context_length']:
        raise ValueError('Not directly comparable: context lengths differ')
    if before['steps'] != after['steps']:
        raise ValueError('Not comparable: planned training schedules differ')
    controlled = ('batch_size','grad_accum','lr','min_lr','warmup_steps', 'weight_decay',
                  'grad_clip', 'attention', 'eval_every', 'tokenizer','tokenizer_sample_bytes',
                  'dropout','seed','val_fraction','steps')
    for field in controlled:
        if before['settings'][field] != after['settings'][field]:
            raise ValueError(f'Not comparable: training setting {field} differs')
    if before['config'] != after['config']:
        raise ValueError('Not comparable: base model architecture differs')
    delta = after['val_loss'] - before['val_loss']
    return {'baseline': before, 'candidate': after,
            'candidate_minus_baseline_loss': delta,
            'lower_loss': ('candidate' if delta < 0 else 'baseline' if delta > 0 else 'tie'),
            'caveat': 'Controls matched, but MoE changes parameter count/compute. One seed cannot establish a reliable advantage.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('candidate')
    args = p.parse_args()
    print(json.dumps(compare(Path(args.baseline), Path(args.candidate)), indent=2))
