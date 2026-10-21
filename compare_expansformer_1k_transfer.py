#!/usr/bin/env python3
"""Read-only, matched v0.4 checkpoint evaluation.

Compares original and transfer checkpoints on identical old/new held-out
DOCUMENTS using the same tokenizer, sliding-free sampled evaluation windows.
Note: both validation cohorts have been used for checkpoint selection, so they
are not independent final test sets.
"""
import argparse
import json
from pathlib import Path

import torch
from corpus_builder import read_documents
from train_expansformer_v01 import OwnTokenizer, load_model_source
from train_expansformer_v04_transfer import (
    FORMAT, evaluate, get_amp, get_device, serialize_stream,
    sha256_file, use_sdpa,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=Path('runs/v04_rtx5050_enhanced/best.pt'))
    parser.add_argument('--transfer', type=Path, default=Path('runs/v04_transfer_431/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('runs/v04_staged_research_data_499'))
    parser.add_argument('--windows', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.windows < 1 or args.batch_size < 1:
        parser.error('windows and batch-size must be positive')

    paths = [args.original, args.transfer]
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    if any(c.get('format') != FORMAT for c in checkpoints):
        raise ValueError('Expected two v0.4 checkpoints')
    a, b = checkpoints
    if a['config'] != b['config'] or a['research_config'] != b['research_config']:
        raise ValueError('Different model architectures; this comparison requires identical architecture')
    sha = a['tokenizer_sha256']
    if sha != b['tokenizer_sha256']:
        raise ValueError('Different tokenizer IDs: cannot compare with a shared token stream')
    for p in paths:
        if sha256_file(p.parent / 'tokenizer.json') != sha:
            raise ValueError(f'Tokenizer file mismatch near {p}')

    original_prov = json.loads((args.original.parent / 'provenance.json').read_text())
    transfer_prov = json.loads((args.transfer.parent / 'provenance.json').read_text())
    old_train = {row['sha256'] for row in original_prov['train']}
    old_val_order = [row['sha256'] for row in original_prov['val']]
    new_train = {row['sha256'] for row in transfer_prov['train']}
    transfer_val_order = [row['sha256'] for row in transfer_prov['val']]
    old_val = set(old_val_order)
    transfer_val = set(transfer_val_order)
    if old_train & transfer_val or not old_val <= transfer_val:
        raise ValueError('Transfer splits did not preserve original training/validation assignments')
    if old_val & new_train or transfer_val & new_train:
        raise ValueError('Validation and train splits overlap')
    new_val_order = [h for h in transfer_val_order if h not in old_val]
    if not new_val_order:
        raise ValueError('No newly held-out documents found in transfer provenance')
    documents = read_documents(args.data)
    docs_by_sha = {d.sha256: d for d in documents}
    required = old_val | set(new_val_order)
    if len(docs_by_sha) != len(documents) or required - set(docs_by_sha):
        raise ValueError('Corpus missing or duplicating expected validation documents')
    groups = {
        'Original 39 held-out documents': [docs_by_sha[h] for h in old_val_order],
        'New 26 held-out documents': [docs_by_sha[h] for h in new_val_order],
    }
    print(f'Matched tokenizer SHA256: {sha}')
    print(f'Original validation documents: {len(old_val_order)}; newly held-out: {len(new_val_order)}')
    print(f'Checkpoint steps: original={a["step"]}, transfer={b["step"]}')
    print('Note: checkpoints have been selected using validation scores; neither cohort is an untouched final test set.\n')

    load_model_source(None)
    from model import ModelConfig
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    cfg = ModelConfig(**a['config'])
    tok = OwnTokenizer.load(args.original.parent / 'tokenizer.json')
    device = get_device(args.device)
    amp, _, _ = get_amp(device, a['settings']['amp'])
    streams = {name: serialize_stream(docs, tok) for name, docs in groups.items()}
    all_scores = {}
    for label, p, checkpoint in [('Original 1K', args.original, a), ('Transfer', args.transfer, b)]:
        model = NeuralTransformerLM(cfg, NeuralConfig(**checkpoint['research_config'])).to(device)
        if checkpoint['settings']['attention'] == 'sdpa':
            use_sdpa(model)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        all_scores[label] = {}
        for cohort, stream in streams.items():
            scores = evaluate(model, stream, tok, cfg.context_length, args.batch_size,
                              args.windows, device, amp)
            all_scores[label][cohort] = scores
            print(f'{label:12} | {cohort:31} | loss={scores["val_loss"]:.4f} | '
                  f'acc={100*scores["val_token_accuracy"]:.2f}% | '
                  f'bpb={scores["val_bits_per_byte"]:.4f} | '
                  f'tokens={scores["evaluated_tokens"]} | '
                  f'coverage={100*scores["val_token_coverage"]:.1f}%')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('\nTransfer − original (same evaluated documents; lower loss/bpb and higher accuracy are better):')
    for cohort in groups:
        prior = all_scores['Original 1K'][cohort]
        after = all_scores['Transfer'][cohort]
        print(f'  {cohort}: loss {after["val_loss"]-prior["val_loss"]:+.4f}; '
              f'accuracy {100*(after["val_token_accuracy"]-prior["val_token_accuracy"]):+.2f} pp; '
              f'bpb {after["val_bits_per_byte"]-prior["val_bits_per_byte"]:+.4f}')


if __name__ == '__main__':
    main()
