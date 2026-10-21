#!/usr/bin/env python3
"""Expansformer v0.4: independently trainable, ablation-ready language-model lab.

Original model.py and all v0.2 scripts remain unmodified. No pretrained weights or hidden APIs.
Optional MoE and intermediate prediction exits require fresh random initialization.
Uses the v0.1 verified tokenizer implementation for stable IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from corpus_builder import Document, corpus_fingerprint, read_documents, split_documents
from train_expansformer_v01 import OwnTokenizer, SPECIAL, load_model_source, lr_for_step

FORMAT = 'expansformer-from-scratch-v0.4'
BYTE_OFFSET = len(SPECIAL)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed: int, device: torch.device):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    if selected == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable: verify nvidia-smi and PyTorch CUDA build.')
    return torch.device(selected)


def get_amp(device: torch.device, mode: str):
    if mode not in {'auto', 'off', 'bf16', 'fp16'}:
        raise ValueError('Unsupported mixed-precision mode')
    if device.type != 'cuda' or mode == 'off':
        if mode in {'bf16', 'fp16'} and device.type != 'cuda':
            raise ValueError('Requested CUDA mixed precision without CUDA device')
        return nullcontext(), torch.amp.GradScaler('cuda', enabled=False), 'float32'
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if mode == 'auto' else (torch.bfloat16 if mode == 'bf16' else torch.float16)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 requested but GPU reports no bf16 support')
    return torch.autocast('cuda', dtype=dtype), torch.amp.GradScaler('cuda', enabled=dtype == torch.float16), str(dtype)


def use_sdpa(model):
    """Swap the attention calculation only; preserve original model parameters and state dict."""
    from types import MethodType
    from model import apply_rope

    def forward(self, x, rope_freqs=None):
        batch, seq, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope and rope_freqs is not None:
            q, k = apply_rope(q, rope_freqs), apply_rope(k, rope_freqs)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.attn_dropout.p if self.training else 0.0)
        result = result.transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.resid_dropout(self.out_proj(result))

    for block in model.blocks:
        block.attn.forward = MethodType(forward, block.attn)
    return model


def serialize_stream(docs: list[Document], tok: OwnTokenizer) -> torch.Tensor:
    ids = []
    for doc in docs:
        ids.append(SPECIAL['<BOS>'])
        ids.extend(tok.encode(doc.text))
        ids.append(SPECIAL['<EOS>'])
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(ids: torch.Tensor, context: int, batch_size: int, rng: torch.Generator, device):
    max_start = len(ids) - context - 1
    if max_start < 0:
        raise ValueError('Too few training tokens for context length')
    positions = torch.randint(max_start + 1, (batch_size,), generator=rng)[:, None] + torch.arange(context)[None, :]
    return ids[positions].to(device), ids[positions + 1].to(device)


@torch.no_grad()
def evaluate(model, ids: torch.Tensor, tok: OwnTokenizer, context: int, batch_size: int,
             max_windows: int, device, amp_ctx) -> dict[str, float]:
    """Non-overlapping target chunks, token-weighted CE and tokenizer-comparable bits/byte.

    First target starts at index 1; each next window starts `context` tokens later.
    Evaluate deterministic bounded prefix. This is a sample, not full-corpus CE
    unless coverage reaches 1.0.
    """
    was_training = model.training
    model.eval()
    blocks = (len(ids) - 1) // context
    n_windows = min(blocks, max_windows)
    if n_windows < 1:
        raise ValueError('Validation corpus too short to evaluate one context window')
    loss_sum, token_sum, correct, byte_sum = 0.0, 0, 0, 0
    byte_len = torch.tensor([len(tok.vocab.get(i, b'')) for i in range(tok.vocab_size)], dtype=torch.long)
    try:
        # Evenly spread disjoint evaluation blocks over the entire held-out stream.
        # A fixed prefix would bias estimates towards the first validation document.
        chosen = ([blocks // 2] if n_windows == 1 else
                  [(i * (blocks - 1)) // (n_windows - 1) for i in range(n_windows)])
        assert len(chosen) == len(set(chosen))
        for first in range(0, n_windows, batch_size):
            starts = torch.tensor(chosen[first:first + batch_size]) * context
            offsets = torch.arange(context)
            pos = starts[:, None] + offsets[None, :]
            x, y = ids[pos].to(device), ids[pos + 1].to(device)
            with amp_ctx:
                logits, loss = model(x, y)
            count = y.numel()
            loss_sum += float(loss.item()) * count
            token_sum += count
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            byte_sum += int(byte_len[ids[pos + 1]].sum().item())
    finally:
        model.train(was_training)
    average = loss_sum / token_sum
    return {'val_loss': average, 'val_perplexity': math.exp(min(average, 50)),
            'val_token_accuracy': correct / token_sum,
            'val_bits_per_byte': (loss_sum / (byte_sum * math.log(2))) if byte_sum else float('nan'),
            'val_token_coverage': token_sum / (len(ids) - 1),
            'evaluated_tokens': token_sum}


def optimizer_for(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and 'embed' not in name else no_decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                              {'params': no_decay, 'weight_decay': 0.0}],
                             lr=args.lr, betas=(.9, .95), eps=1e-8)


def save_atomic(blob: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(blob, temp)
    os.replace(temp, path)


def write_json_atomic(data: dict, path: Path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def check_args(a):
    from expansformer_v04_model import NeuralConfig
    research = NeuralConfig(a.moe_experts, a.moe_top_k, a.moe_balance_weight,
                              a.auxiliary_exits, a.exit_loss_weight,
                            a.memory_slots, a.memory_every,
                            a.next_token_heads, a.next_token_head_weight)
    research.validate(a.n_layers)
    for name in ('steps', 'batch_size', 'grad_accum', 'eval_every', 'save_every', 'eval_windows', 'n_layers', 'n_heads', 'embed_dim'):
        if getattr(a, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if a.context_length < 2 or a.embed_dim % a.n_heads or (a.embed_dim // a.n_heads) % 2:
        raise ValueError('Require context >= 2; embed_dim divisible by heads with even head dimension')
    if not (0 <= a.dropout < 1):
        raise ValueError('dropout must be between 0 and 1')
    if not (a.lr > 0 and 0 <= a.min_lr <= a.lr and a.warmup_steps >= 0):
        raise ValueError('Invalid learning-rate settings')
    if a.stop_after is not None and not 1 <= a.stop_after <= a.steps:
        raise ValueError('stop-after must be 1..steps')
    if not (a.grad_clip > 0 and a.weight_decay >= 0 and a.min_delta >= 0):
        raise ValueError('Invalid clipping/decay/min-delta')
    if a.tokenizer == 'bpe' and a.vocab_size < 260:
        raise ValueError('BPE vocabulary must contain at least 260 token IDs')
    if a.early_stop_patience < 0:
        raise ValueError('early-stop-patience cannot be negative')


def prepare_data(a):
    documents = read_documents(Path(a.data), a.data_license, a.max_document_bytes)
    if a.val_data:
        train_docs = documents
        val_docs = read_documents(Path(a.val_data), a.data_license, a.max_document_bytes)
    else:
        train_docs, val_docs = split_documents(documents, a.seed, a.val_fraction)
    train_ids = {d.sha256 for d in train_docs}
    val_ids = {d.sha256 for d in val_docs}
    if not train_ids.isdisjoint(val_ids):
        raise ValueError('Exact content duplication across training and validation splits')
    if a.strict_provenance and any(d.license == 'UNVERIFIED' for d in train_docs + val_docs):
        raise ValueError('Strict provenance requested but some documents lack license metadata')
    fingerprint = corpus_fingerprint(train_docs, val_docs)
    return train_docs, val_docs, fingerprint


def prepare_transfer_data(a, parent_dir: Path):
    """Preserve all original train/validation assignments; allocate ONLY new docs.

    A formerly trained document must never enter validation. Parent validation
    documents remain held out (though previously used for checkpoint selection).
    """
    if a.val_data:
        raise ValueError('--val-data is not supported with --init-from; inherited holdouts are mandatory')
    provenance_path = parent_dir / 'provenance.json'
    if not provenance_path.is_file():
        raise ValueError(f'Missing source split provenance: {provenance_path}')
    parent = json.loads(provenance_path.read_text(encoding='utf-8'))
    old_train = {d['sha256'] for d in parent['train']}
    old_val = {d['sha256'] for d in parent['val']}
    if not old_train or not old_val or not old_train.isdisjoint(old_val):
        raise ValueError('Invalid or overlapping parent train/validation provenance')
    documents = read_documents(Path(a.data), a.data_license, a.max_document_bytes)
    by_hash = {d.sha256: d for d in documents}
    if len(by_hash) != len(documents):
        raise ValueError('Expanded corpus contains exact content-hash duplicates')
    missing = (old_train | old_val) - set(by_hash)
    if missing:
        raise ValueError(f'Expanded corpus is missing {len(missing)} source documents; refuse transfer')
    new_docs = [d for d in documents if d.sha256 not in old_train | old_val]
    if len(new_docs) < 2:
        raise ValueError(f'Only {len(new_docs)} unseen new documents; require >=2 for held-out transfer evaluation')
    new_train, new_val = split_documents(new_docs, a.seed, a.val_fraction)
    train_docs = [d for d in documents if d.sha256 in old_train] + new_train
    val_docs = [d for d in documents if d.sha256 in old_val] + new_val
    if not {d.sha256 for d in train_docs}.isdisjoint({d.sha256 for d in val_docs}):
        raise ValueError('Training/validation overlap after transfer split')
    if a.strict_provenance and any(d.license == 'UNVERIFIED' for d in train_docs + val_docs):
        raise ValueError('Strict provenance requested but a document lacks license declaration')
    counts = {'old_training': len(old_train), 'old_validation': len(old_val),
              'new_training': len(new_train), 'new_validation': len(new_val)}
    return train_docs, val_docs, corpus_fingerprint(train_docs, val_docs), counts


def prepare_saved_transfer_data(a, out: Path):
    """Rebuild the exact inherited split on resume, rather than reshuffling."""
    prov = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    if not prov.get('transfer_lineage'):
        return prepare_data(a)
    documents = read_documents(Path(a.data), a.data_license, a.max_document_bytes)
    by_hash = {d.sha256: d for d in documents}
    if len(by_hash) != len(documents):
        raise ValueError('Changed or duplicated transfer input')
    train_hashes = [row['sha256'] for row in prov['train']]
    val_hashes = [row['sha256'] for row in prov['val']]
    if set(train_hashes) & set(val_hashes):
        raise ValueError('Saved transfer split overlaps')
    if (set(train_hashes) | set(val_hashes)) != set(by_hash):
        raise ValueError('Expanded corpus changed since warm-start; cannot resume')
    train_docs = [by_hash[h] for h in train_hashes]
    val_docs = [by_hash[h] for h in val_hashes]
    if a.strict_provenance and any(d.license == 'UNVERIFIED' for d in train_docs + val_docs):
        raise ValueError('Strict provenance requires every document license declaration')
    return train_docs, val_docs, corpus_fingerprint(train_docs, val_docs)


def tokenize_data(tok, train_docs, val_docs, context):
    train_ids, val_ids = serialize_stream(train_docs, tok), serialize_stream(val_docs, tok)
    if len(train_ids) < context + 2 or len(val_ids) < context + 1:
        raise ValueError(f'Too few tokens: train={len(train_ids)}, validation={len(val_ids)}, '
                         f'context={context}. Reduce --context-length or acquire more documents.')
    return train_ids, val_ids


def settings_for(a):
    keys = ('tokenizer', 'vocab_size', 'tokenizer_sample_bytes', 'context_length', 'n_layers', 'n_heads',
            'embed_dim', 'dropout', 'batch_size', 'grad_accum', 'steps', 'lr', 'min_lr',
            'warmup_steps', 'weight_decay', 'grad_clip', 'eval_every', 'eval_windows',
            'save_every', 'seed', 'amp', 'attention', 'val_fraction', 'data_license',
            'strict_provenance', 'min_delta', 'early_stop_patience',
            'moe_experts', 'moe_top_k', 'moe_balance_weight',
            'auxiliary_exits', 'exit_loss_weight', 'memory_slots', 'memory_every',
            'next_token_heads', 'next_token_head_weight')
    return {name: getattr(a, name) for name in keys}


def trim_metrics(path: Path, resume_step: int):
    if not path.exists():
        if resume_step > 0:
            raise ValueError('Metrics file missing for a nonempty checkpoint')
        return
    lines = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Only a torn final append is recoverable. A gap earlier is not.
            break
        if record.get('step', -1) <= resume_step:
            lines.append(record)
    if [r['step'] for r in lines] != list(range(1, len(lines) + 1)):
        raise ValueError('Metrics step sequence is corrupt')
    if len(lines) == resume_step - 1:
        # Atomic checkpoint saved, but the process exited before the last log
        # append. Never invent a loss or gradient norm for that lost log row.
        lines.append({'step': resume_step, 'checkpoint_recovered': True,
                      'train_loss': None, 'val_loss': None,
                      'note': 'Checkpoint survived an interrupted metrics append; losses unavailable'})
    if len(lines) != resume_step:
        raise ValueError('Existing metrics are missing or inconsistent with checkpoint; '
                         'preserve the old run and start a new output directory.')
    path.write_text(''.join(json.dumps(r) + '\n' for r in lines), encoding='utf-8')


def train(a):
    check_args(a)
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    config_type, _ = load_model_source(a.source)
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    research = NeuralConfig(a.moe_experts, a.moe_top_k, a.moe_balance_weight,
                              a.auxiliary_exits, a.exit_loss_weight,
                            a.memory_slots, a.memory_every,
                            a.next_token_heads, a.next_token_head_weight)
    device = get_device(a.device)
    if a.init_from and a.resume:
        raise ValueError('Use exactly one of --init-from or --resume')
    init_ckpt, init_meta = None, None
    if a.init_from:
        source_path = Path(a.init_from).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f'Missing source checkpoint: {source_path}')
        if Path(a.out).resolve() == source_path.parent:
            raise ValueError('Transfer output must be a different directory from parent checkpoint')
        init_ckpt = torch.load(source_path, map_location='cpu', weights_only=True)
        if init_ckpt.get('format') != FORMAT:
            raise ValueError('Parent checkpoint format incompatible with v0.4')
        source_tokenizer = source_path.parent / 'tokenizer.json'
        if not source_tokenizer.is_file() or sha256_file(source_tokenizer) != init_ckpt.get('tokenizer_sha256'):
            raise ValueError('Source tokenizer missing or checksum mismatch')
        if asdict(research) != init_ckpt.get('research_config'):
            raise ValueError('Neural architecture/settings mismatch with parent')
        old_cfg = init_ckpt['config']
        for attr in ('n_layers', 'n_heads', 'embed_dim', 'context_length', 'dropout'):
            if old_cfg.get(attr) != getattr(a, attr):
                raise ValueError(f'Parent architecture mismatch: {attr}')
        if init_ckpt['settings']['tokenizer'] != a.tokenizer or old_cfg['vocab_size'] != a.vocab_size:
            raise ValueError('Parent tokenization settings incompatible with new experiment')
        train_docs, val_docs, fingerprint, split_counts = prepare_transfer_data(a, source_path.parent)
        init_meta = {'checkpoint': str(source_path), 'checkpoint_sha256': sha256_file(source_path),
                     'source_step': int(init_ckpt['step']), 'source_fingerprint': init_ckpt['fingerprint'],
                     'source_tokenizer_sha256': init_ckpt['tokenizer_sha256'],
                     'split_counts': split_counts,
                     'optimizer': 'REINITIALIZED', 'lr_schedule': 'RESTARTED',
                     'tokenizer': 'PRESERVED_IDENTICAL',
                     'warning': 'Old validation data was used to select the parent checkpoint; new validation documents are the only newly held-out examples.'}
    else:
        out_candidate = Path(a.out)
        if a.resume and (out_candidate / 'provenance.json').is_file():
            train_docs, val_docs, fingerprint = prepare_saved_transfer_data(a, out_candidate)
        else:
            train_docs, val_docs, fingerprint = prepare_data(a)
    out = Path(a.out)
    if a.resume:
        ckpt_path = Path(a.resume)
        if out.resolve() != ckpt_path.resolve().parent:
            raise ValueError('Resume checkpoint must be in --out directory')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if ckpt.get('format') != FORMAT:
            raise ValueError('Incompatible checkpoint schema; v0.2 checkpoints cannot be resumed as v0.4')
        if ckpt.get('fingerprint') != fingerprint:
            raise ValueError('Dataset content or provenance changed; refusing resume')
        settings = settings_for(a)
        for key, val in settings.items():
            if ckpt['settings'][key] != val:
                raise ValueError(f'Resume setting changed: {key}')
        cfg = config_type(**ckpt['config'])
        tok = OwnTokenizer.load(out / 'tokenizer.json')
        if sha256_file(out / 'tokenizer.json') != ckpt['tokenizer_sha256']:
            raise ValueError('Tokenizer checksum changed; refusing resume')
        if cfg.vocab_size != tok.vocab_size:
            raise ValueError('Tokenizer/model vocabulary disagreement')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Output folder is not empty ({out}); choose a new --out or use --resume')
        out.mkdir(parents=True, exist_ok=True)
        if init_ckpt is not None:
            parent_tokenizer = Path(a.init_from).resolve().parent / 'tokenizer.json'
            shutil.copyfile(parent_tokenizer, out / 'tokenizer.json')
            tok = OwnTokenizer.load(out / 'tokenizer.json')
            if sha256_file(out / 'tokenizer.json') != init_ckpt['tokenizer_sha256']:
                raise ValueError('Transferred tokenizer checksum mismatch')
            cfg = config_type(**init_ckpt['config'])
            if cfg.vocab_size != tok.vocab_size:
                raise ValueError('Transferred vocabulary mismatch')
        else:
            tok = OwnTokenizer()
            if a.tokenizer == 'bpe':
                text = '\n\n'.join(doc.text for doc in train_docs)
                tok.train_bpe(text, a.vocab_size, a.tokenizer_sample_bytes)
            tok.save(out / 'tokenizer.json')
            cfg = config_type(vocab_size=tok.vocab_size, context_length=a.context_length,
                              n_layers=a.n_layers, n_heads=a.n_heads, embed_dim=a.embed_dim,
                              dropout=a.dropout)
        write_json_atomic(asdict(cfg), out / 'config.json')
        settings = settings_for(a)
        write_json_atomic({'fingerprint': fingerprint, 'train': [asdict(d) | {'text': None} for d in train_docs],
                           'val': [asdict(d) | {'text': None} for d in val_docs],
                           'license_notice': 'Metadata is descriptive, not proof of licensing or correctness.',
                           'model_source': ('Transferred neural weights from source checkpoint' if init_ckpt is not None
                                            else 'Original Expansformer backbone, optional v0.4 trainable experiments; new random weights.'),
                           'transfer_lineage': init_meta,
                           'research_config': asdict(research)},
                          out / 'provenance.json')
        ckpt = None
    train_ids, val_ids = tokenize_data(tok, train_docs, val_docs, cfg.context_length)
    seed_everything(a.seed, device)
    model = NeuralTransformerLM(cfg, research).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt['model'], strict=True)
        del init_ckpt   # Release parent state before GPU optimizer allocation.
    if a.attention == 'sdpa':
        use_sdpa(model)
    optimizer = optimizer_for(model, a)
    amp_ctx, scaler, dtype_label = get_amp(device, a.amp)
    sampler = torch.Generator(device='cpu').manual_seed(a.seed + 1729)
    step0, best, bad_evals, stopped = 0, float('inf'), 0, False
    if ckpt:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scaler.is_enabled() and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        sampler.set_state(ckpt['sampler_rng'])
        torch.set_rng_state(ckpt['cpu_rng'])
        if device.type == 'cuda' and ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        step0, best = int(ckpt['step']), float(ckpt['best_val_loss'])
        bad_evals = int(ckpt['bad_evals'])
        stopped = bool(ckpt.get('early_stopped', False))
        trim_metrics(out / 'metrics.jsonl', step0)
    if stopped:
        print(f'Checkpoint stopped early at step {step0}. Train a new run to change the experiment.')
        return
    final_step = min(a.steps, a.stop_after) if a.stop_after else a.steps
    if step0 >= final_step:
        print(f'Already trained to step {step0}; no changes.')
        return
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'Expansformer v0.4 | {device} ({dtype_label}) | attention={a.attention} | moe={research.moe_experts} | exits={research.auxiliary_exits} | '
          f'memory={research.memory_slots} | next_token_heads={research.next_token_heads} | {n_parameters:,} parameters | train={len(train_ids):,} / val={len(val_ids):,} tokens | '
          f'documents={len(train_docs)}/{len(val_docs)}', flush=True)
    def checkpoint(completed: int, early=False):
        return {'format': FORMAT, 'step': completed, 'config': asdict(cfg),
                'research_config': asdict(research), 'parameter_count': n_parameters,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                'fingerprint': fingerprint, 'tokenizer_sha256': sha256_file(out / 'tokenizer.json'),
                'transfer_lineage': (init_meta if init_meta is not None else
                    json.loads((out / 'provenance.json').read_text(encoding='utf-8')).get('transfer_lineage')),
                'settings': settings, 'best_val_loss': best, 'bad_evals': bad_evals,
                'early_stopped': early, 'sampler_rng': sampler.get_state(),
                'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}

    metrics = out / 'metrics.jsonl'
    tic = time.monotonic()
    for step in range(step0, final_step):
        model.train()
        lr = lr_for_step(step, a.steps, a.warmup_steps, a.lr, a.min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_lm_loss = 0.0
        train_exit_loss = 0.0
        train_balance_loss = 0.0
        train_predict_loss = 0.0
        for micro in range(a.grad_accum):
            x, y = sample_batch(train_ids, cfg.context_length, a.batch_size, sampler, device)
            with amp_ctx:
                loss, details = model.training_loss(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at optimizer step {step+1}, microstep {micro+1}')
            scaler.scale(loss / a.grad_accum).backward()
            train_loss += float(loss.detach()) / a.grad_accum
            train_lm_loss += float(details['lm_loss']) / a.grad_accum
            train_exit_loss += float(details['exit_loss']) / a.grad_accum
            train_balance_loss += float(details['balance_loss']) / a.grad_accum
            train_predict_loss += float(details.get('next_token_head_loss', 0.0)) / a.grad_accum
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f'Non-finite gradients at optimizer step {step+1}')
        old_scale = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        if old_scale is not None and scaler.get_scale() < old_scale:
            raise FloatingPointError('AMP skipped an optimizer update; refusing to miscount steps')
        completed = step + 1
        do_eval = completed % a.eval_every == 0 or completed == final_step
        score = None
        is_best = False
        early = False
        if do_eval:
            score = evaluate(model, val_ids, tok, cfg.context_length, a.batch_size,
                             a.eval_windows, device, amp_ctx)
            if not math.isfinite(score['val_loss']):
                raise FloatingPointError('Non-finite validation loss')
            if score['val_loss'] < best - a.min_delta:
                best = score['val_loss']
                bad_evals = 0
                is_best = True
            else:
                bad_evals += 1
            early = a.early_stop_patience > 0 and bad_evals >= a.early_stop_patience
        record = {'step': completed, 'train_loss': train_loss, 'train_lm_loss': train_lm_loss,
                  'train_exit_loss': train_exit_loss, 'train_router_balance_loss': train_balance_loss,
                  'train_next_token_head_loss': train_predict_loss,
                  'grad_norm': grad_norm,
                  'lr': lr, 'tokens_seen': completed*a.batch_size*a.grad_accum*cfg.context_length,
                  'seconds_since_start': round(time.monotonic()-tic, 3), **(score or {})}
        # Commit order: write latest checkpoint before best, metrics last;
        # metrics beyond checkpoint are discarded on resume (no double accounting).
        if do_eval or completed % a.save_every == 0 or completed == final_step:
            state = checkpoint(completed, early)
            save_atomic(state, out / 'latest.pt')
            if is_best:
                save_atomic(state, out / 'best.pt')
        with metrics.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        if completed == 1 or do_eval or completed % a.save_every == 0:
            message = f'update {completed}/{a.steps} | train {train_loss:.4f} | lr {lr:.3g}'
            if score:
                message += f' | val {score["val_loss"]:.4f} | bpb {score["val_bits_per_byte"]:.3f} | best {best:.4f}'
            print(message, flush=True)
        if early:
            print(f'Early stopping at step {completed} after {bad_evals} non-improving evaluations.')
            break
    print(f'Complete; checkpoints in {out}. Native PyTorch generation only; Ollama export not yet implemented.')


@torch.no_grad()
def generate(a):
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    _, _ = load_model_source(a.source)
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    path = Path(a.checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('format') != FORMAT:
        raise ValueError('Expected a v0.4 checkpoint; v0.2 models use train_expansformer_v02.py')
    tok_path = path.parent / 'tokenizer.json'
    if sha256_file(tok_path) != ckpt['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum does not match checkpoint')
    tok = OwnTokenizer.load(tok_path)
    from model import ModelConfig
    cfg = ModelConfig(**ckpt['config'])
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError('Checkpoint/tokenizer vocabulary differs')
    device = get_device(a.device)
    model = NeuralTransformerLM(cfg, NeuralConfig(**ckpt['research_config'])).to(device)
    if ckpt['settings']['attention'] == 'sdpa':
        use_sdpa(model)
    model.load_state_dict(ckpt['model'])
    model.eval()
    torch.manual_seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
    ids = tok.encode(a.prompt) or [SPECIAL['<BOS>']]
    current = torch.tensor([ids], dtype=torch.long, device=device)
    if a.experimental_adaptive_exit_threshold is not None:
        print('WARNING: experimental adaptive exits are not calibrated for correctness or factual reliability.')
        if a.inference_depth is not None:
            raise ValueError('Do not combine --inference-depth with adaptive selection')
    used_depths = []
    # Exclude untrained control tokens from ordinary text generation. EOS terminates.
    for _ in range(a.max_new_tokens):
        if a.experimental_adaptive_exit_threshold is not None:
            logits, depth, _ = model.adaptive_logits(current[:, -cfg.context_length:],
                                                      a.experimental_adaptive_exit_threshold)
            used_depths.append(depth)
        else:
            logits = model.logits_at_depth(current[:, -cfg.context_length:], a.inference_depth or cfg.n_layers)
        scores = logits[:, -1, :].float()
        scores[:, :3] = -float('inf')
        if a.temperature == 0:
            token = scores.argmax(dim=-1, keepdim=True)
        else:
            scores /= a.temperature
            if a.top_k > 0:
                threshold = torch.topk(scores, min(a.top_k, scores.shape[-1])).values[:, -1:]
                scores[scores < threshold] = -float('inf')
            if a.top_p < 1:
                values, indices = torch.sort(scores, descending=True)
                probs = values.softmax(dim=-1)
                mask = probs.cumsum(dim=-1) - probs >= a.top_p
                values[mask] = -float('inf')
                scores = torch.full_like(scores, -float('inf')).scatter(1, indices, values)
            token = torch.multinomial(scores.softmax(dim=-1), 1)
        if int(token[0, 0]) == SPECIAL['<EOS>']:
            break
        current = torch.cat([current, token], dim=-1)
    print(tok.decode(current[0].tolist()))
    if used_depths:
        print(f'Experimental depths: mean={sum(used_depths)/len(used_depths):.2f}; full={cfg.n_layers}; tokens={len(used_depths)}')


def make_parser():
    p = argparse.ArgumentParser(description='Expansformer from-scratch research trainer v0.4 (not Ollama)')
    sub = p.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train', help='Train from random weights or resume same experiment')
    tr.add_argument('--data', required=True)
    tr.add_argument('--val-data')
    tr.add_argument('--out', default='./runs/expansformer_v04')
    tr.add_argument('--source', help='Original expansformer_model folder or ZIP')
    tr.add_argument('--data-license', default='UNVERIFIED', help='License claim for local files lacking manifest; NOT validated automatically')
    tr.add_argument('--strict-provenance', action='store_true')
    tr.add_argument('--val-fraction', type=float, default=.15)
    tr.add_argument('--max-document-bytes', type=int, default=16_000_000)
    tr.add_argument('--tokenizer', choices=['byte', 'bpe'], default='byte')
    tr.add_argument('--vocab-size', type=int, default=512)
    tr.add_argument('--tokenizer-sample-bytes', type=int, default=131_072)
    tr.add_argument('--n-layers', type=int, default=4)
    tr.add_argument('--n-heads', type=int, default=4)
    tr.add_argument('--embed-dim', type=int, default=256)
    tr.add_argument('--context-length', type=int, default=128)
    tr.add_argument('--dropout', type=float, default=.1)
    tr.add_argument('--batch-size', type=int, default=8)
    tr.add_argument('--grad-accum', type=int, default=2)
    tr.add_argument('--steps', type=int, default=1000)
    tr.add_argument('--stop-after', type=int)
    tr.add_argument('--lr', type=float, default=3e-4)
    tr.add_argument('--min-lr', type=float, default=1e-5)
    tr.add_argument('--warmup-steps', type=int, default=30)
    tr.add_argument('--weight-decay', type=float, default=.1)
    tr.add_argument('--grad-clip', type=float, default=1.0)
    tr.add_argument('--eval-every', type=int, default=100)
    tr.add_argument('--eval-windows', type=int, default=64)
    tr.add_argument('--save-every', type=int, default=100)
    tr.add_argument('--min-delta', type=float, default=0.0)
    tr.add_argument('--early-stop-patience', type=int, default=0)
    tr.add_argument('--moe-experts', type=int, default=0, help='0 = original dense FFN. Otherwise sparse top-k experts per block')
    tr.add_argument('--moe-top-k', type=int, default=2)
    tr.add_argument('--moe-balance-weight', type=float, default=0.01)
    tr.add_argument('--auxiliary-exits', action='store_true', help='Train prediction heads at intermediate block depths')
    tr.add_argument('--exit-loss-weight', type=float, default=0.15)
    tr.add_argument('--memory-slots', type=int, default=0, help='0=disabled; learned parametric slots, not factual memory')
    tr.add_argument('--memory-every', type=int, default=2, help='Apply slot read after every N blocks')
    tr.add_argument('--next-token-heads', action='store_true', help='Train intermediate next-token accuracy predictors, NOT truth detectors; requires exits')
    tr.add_argument('--next-token-head-weight', type=float, default=0.1)
    tr.add_argument('--attention', choices=['classic', 'sdpa'], default='sdpa')
    tr.add_argument('--amp', choices=['auto', 'off', 'bf16', 'fp16'], default='auto')
    tr.add_argument('--resume')
    tr.add_argument('--init-from', help='Initialize new experiment from v0.4 weights/tokenizer, retaining old train/val split; resets optimizer and step count')
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    tr.add_argument('--cpu-threads', type=int, default=2)
    ge = sub.add_parser('generate', help='Generate with trained PyTorch model')
    ge.add_argument('--checkpoint', required=True)
    ge.add_argument('--prompt', default='Question: What do we know?\nAnswer:')
    ge.add_argument('--max-new-tokens', type=int, default=80)
    ge.add_argument('--temperature', type=float, default=.8)
    ge.add_argument('--top-k', type=int, default=40)
    ge.add_argument('--top-p', type=float, default=.9)
    ge.add_argument('--experimental-adaptive-exit-threshold', type=float, help='UNCALIBRATED early-exit threshold; experimental only')
    ge.add_argument('--inference-depth', type=int, help='Use a trained intermediate exit, or full model by default')
    ge.add_argument('--seed', type=int, default=42)
    ge.add_argument('--source')
    ge.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ge.add_argument('--cpu-threads', type=int, default=2)
    sub.add_parser('smoke', help='Run deterministic CPU/manifest/attention/regression tests (offline)')
    return p


def main():
    p = make_parser()
    a = p.parse_args()
    if a.command == 'train':
        train(a)
    elif a.command == 'generate':
        generate(a)
    else:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        if not outcome.wasSuccessful():
            raise SystemExit(1)


if __name__ == '__main__':
    main()
