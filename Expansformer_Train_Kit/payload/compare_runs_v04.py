"""Compare *completed* v0.4 runs with the same data/tokenizer/training controls."""
from compare_runs import compare as compare_v03
# Match v0.3 strict comparison behavior; only accept v0.4 checkpoint schema.
import argparse
import json
from pathlib import Path
from compare_runs import summarize as old_summarize

# The shared v0.3 utility is preserved untouched for reproducing old experiments.
def summarize(directory: Path):
    import torch
    checkpoint = torch.load(directory / 'latest.pt', weights_only=True, map_location='cpu')
    if checkpoint.get('format') != 'expansformer-from-scratch-v0.4':
        raise ValueError('Expected a v0.4 checkpoint')
    rows = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines() if x.strip()]
    if not rows or rows[-1]['step'] != checkpoint['step'] or checkpoint['step'] != checkpoint['settings']['steps']:
        raise ValueError('Comparison requires completed, logged runs')
    evaluations = [x for x in rows if x.get('val_loss') is not None]
    if not evaluations:
        raise ValueError('No measured evaluation')
    best = min(evaluations, key=lambda x: x['val_loss'])
    return {'path': str(directory), 'fingerprint': checkpoint['fingerprint'],
            'tokenizer_sha256': checkpoint['tokenizer_sha256'],
            'steps': checkpoint['settings']['steps'], 'settings': checkpoint['settings'],
            'config': checkpoint['config'], 'parameters': checkpoint['parameter_count'],
            'best_step': best['step'], 'val_loss': best['val_loss'],
            'val_bits_per_byte': best['val_bits_per_byte'], 'evaluated_tokens': best['evaluated_tokens']}


def compare(a: Path, b: Path):
    old, new = summarize(a), summarize(b)
    for k in ['fingerprint', 'tokenizer_sha256', 'steps', 'config', 'evaluated_tokens']:
        if old[k] != new[k]:
            raise ValueError(f'Not comparable: {k} differs')
    controlled = ['seed', 'tokenizer', 'tokenizer_sample_bytes', 'val_fraction', 'batch_size',
                  'grad_accum', 'lr', 'min_lr', 'warmup_steps', 'weight_decay', 'grad_clip',
                  'eval_every', 'eval_windows', 'attention', 'dropout', 'amp']
    for k in controlled:
        if old['settings'][k] != new['settings'][k]:
            raise ValueError(f'Not comparable: {k} differs')
    difference = new['val_loss'] - old['val_loss']
    return {'baseline': old, 'candidate': new, 'candidate_minus_baseline_loss': difference,
            'warning': 'Matched corpus, tokenizer, seed, and training settings; differences in parameter count/compute remain. Single short-run result is not proof of improvement.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    args = parser.parse_args()
    print(json.dumps(compare(Path(args.baseline), Path(args.candidate)), indent=2))
