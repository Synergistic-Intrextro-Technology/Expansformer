#!/usr/bin/env python3
"""Measure all trained exit depths against the same held-out windows.

Chooses shallowest depth within user-defined CE regression tolerance. This is
not per-query adaptive routing or calibrated evidence of correct answers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from corpus_builder import read_documents, split_documents, corpus_fingerprint
from train_expansformer_v01 import OwnTokenizer, load_model_source
from train_expansformer_v03 import (FORMAT, evaluate, serialize_stream, use_sdpa,
                                     get_amp, get_device, sha256_file)


class ExitView(nn.Module):
    def __init__(self, model, depth):
        super().__init__()
        self.model = model
        self.depth = depth

    def forward(self, ids, targets):
        logits = self.model.logits_at_depth(ids, self.depth)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def measure(checkpoint: Path, data: Path, val_data: Path | None = None,
            eval_windows: int | None = None, max_loss_increase: float = 0.03, device_name='auto'):
    if max_loss_increase < 0:
        raise ValueError('max_loss_increase must be nonnegative')
    load_model_source(None)
    from model import ModelConfig
    from expansformer_v03_model import ResearchConfig, ResearchTransformerLM
    ckpt=torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('format') != FORMAT:
        raise ValueError('Expected v0.3 checkpoint')
    settings = ckpt['settings']
    if not ckpt['research_config']['auxiliary_exits']:
        raise ValueError('No intermediate exits were trained')
    train_docs=read_documents(data, settings['data_license'])
    if val_data:
        val_docs=read_documents(val_data, settings['data_license'])
    else:
        train_docs, val_docs=split_documents(train_docs,settings['seed'],settings['val_fraction'])
    if corpus_fingerprint(train_docs,val_docs)!=ckpt['fingerprint']:
        raise ValueError('Held-out corpus or provenance changed; cannot compare exit depths')
    tok_path=checkpoint.parent/'tokenizer.json'
    if sha256_file(tok_path)!=ckpt['tokenizer_sha256']:
        raise ValueError('Tokenizer hash differs from checkpoint')
    tok=OwnTokenizer.load(tok_path)
    device=get_device(device_name)
    cfg=ModelConfig(**ckpt['config'])
    model=ResearchTransformerLM(cfg,ResearchConfig(**ckpt['research_config'])).to(device)
    if settings['attention']=='sdpa':
        use_sdpa(model)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ids=serialize_stream(val_docs,tok)
    if eval_windows is None:
        eval_windows=settings['eval_windows']
    if eval_windows < 1:
        raise ValueError('eval_windows must be positive')
    amp_ctx,_,_=get_amp(device,settings['amp'])
    output=[]
    for depth in range(1,cfg.n_layers+1):
        result=evaluate(ExitView(model,depth),ids,tok,cfg.context_length,
                        settings['batch_size'],eval_windows,device,amp_ctx)
        output.append({'depth':depth,**result})
    full=output[-1]['val_loss']
    acceptable=[r for r in output if r['val_loss'] <= full + max_loss_increase]
    chosen=min(acceptable,key=lambda r:r['depth'])
    return {'checkpoint':str(checkpoint), 'max_loss_increase':max_loss_increase,
            'full_depth_val_loss':full, 'selected_depth':chosen['depth'],
            'depths':output, 'warning':'Shallow depth accepted on bounded held-out CE only; no accuracy, latency, or factuality guarantee.'}


def main():
    p=argparse.ArgumentParser(description='Evaluate trained intermediate exits against held-out CE')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--val-data')
    p.add_argument('--eval-windows',type=int)
    p.add_argument('--max-loss-increase',type=float,default=0.03)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    print(json.dumps(measure(Path(a.checkpoint),Path(a.data),Path(a.val_data) if a.val_data else None,
                             a.eval_windows,a.max_loss_increase,a.device),indent=2))

if __name__=='__main__':
    main()
