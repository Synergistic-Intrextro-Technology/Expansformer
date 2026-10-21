#!/usr/bin/env python3
"""Audit experimental learned depth selection on genuinely held-out corpus.

Measures next-token top-1 accuracy and full-depth prediction agreement, NOT
semantic truth, claim support, calibrated probability, or safety.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from corpus_builder import read_documents, split_documents, corpus_fingerprint
from train_expansformer_v01 import OwnTokenizer, load_model_source
from train_expansformer_v04 import sha256_file, serialize_stream, use_sdpa, get_device, FORMAT


def audit(checkpoint: Path, data: Path, *, threshold: float = 0.95,
          windows: int = 32, val_data: Path | None = None, device_name: str = 'cpu'):
    if windows < 1 or not 0 < threshold <= 1:
        raise ValueError('windows >=1 and 0 < threshold <=1 are required')
    load_model_source(None)
    from model import ModelConfig
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('format') != FORMAT or not ckpt['research_config']['next_token_heads']:
        raise ValueError('Requires a v0.4 checkpoint trained with --next-token-heads')
    settings = ckpt['settings']
    docs = read_documents(data, settings['data_license'])
    if val_data:
        train_docs, val_docs = docs, read_documents(val_data, settings['data_license'])
    else:
        train_docs, val_docs = split_documents(docs, settings['seed'], settings['val_fraction'])
    if corpus_fingerprint(train_docs, val_docs) != ckpt['fingerprint']:
        raise ValueError('Data/provenance changed; cannot audit checkpoint against different holdout')
    tokfile = checkpoint.parent / 'tokenizer.json'
    if sha256_file(tokfile) != ckpt['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum differs')
    tokenizer = OwnTokenizer.load(tokfile)
    device = get_device(device_name)
    cfg = ModelConfig(**ckpt['config'])
    model = NeuralTransformerLM(cfg, NeuralConfig(**ckpt['research_config'])).to(device).eval()
    if settings['attention'] == 'sdpa':
        use_sdpa(model)
    model.load_state_dict(ckpt['model'])
    stream = serialize_stream(val_docs, tokenizer)
    total = min(windows, (len(stream)-1)//cfg.context_length)
    if not total:
        raise ValueError('Not enough held-out tokens')
    selected_depths, selected_matches, full_matches, agreement, early = [], 0, 0, 0, 0
    with torch.no_grad():
        for window in range(total):
            start = (window * ((len(stream)-1)//cfg.context_length-1) // max(1,total-1)) * cfg.context_length
            x = stream[start:start+cfg.context_length][None,:].to(device)
            next_token = int(stream[start+cfg.context_length])
            chosen, depth, _ = model.adaptive_logits(x,threshold)
            full = model.logits_at_depth(x,cfg.n_layers)
            cp = int(chosen[0,-1].argmax())
            fp = int(full[0,-1].argmax())
            selected_depths.append(depth)
            early += int(depth < cfg.n_layers)
            selected_matches += int(cp == next_token)
            full_matches += int(fp == next_token)
            agreement += int(cp == fp)
    return {'held_out_windows': total, 'threshold': threshold,
            'early_fraction': early/total,
            'mean_selected_depth': sum(selected_depths)/total, 'full_depth': cfg.n_layers,
            'selected_top1_next_token_accuracy': selected_matches/total,
            'full_top1_next_token_accuracy': full_matches/total,
            'agreement_with_full_top1': agreement/total,
            'warning': 'Uncalibrated, small-sample next-token test only. Does not imply factual correctness, independent verification, lower latency, or knowledge reliability.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--val-data')
    p.add_argument('--threshold', type=float, default=.95)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--device', choices=['cpu','cuda','auto'], default='cpu')
    a = p.parse_args()
    print(json.dumps(audit(Path(a.checkpoint), Path(a.data), threshold=a.threshold, windows=a.windows,
                           val_data=Path(a.val_data) if a.val_data else None, device_name=a.device), indent=2))
