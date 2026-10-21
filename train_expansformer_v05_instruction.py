#!/usr/bin/env python3
"""Expansformer v0.5: isolated answer-only supervised fine-tuning research pilot.

Requires the user's existing v0.4 project files on PYTHONPATH / in current folder.
No writes to parent checkpoints, tokenizer, source dataset, or audit artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from train_expansformer_v01 import OwnTokenizer, SPECIAL, load_model_source, lr_for_step
from train_expansformer_v04 import (
    get_amp, get_device, optimizer_for, save_atomic, seed_everything,
    sha256_file, use_sdpa, evaluate as lm_evaluate, serialize_stream,
)

FORMAT = 'expansformer-instruction-pilot-v0.5'
PARENT_FORMAT = 'expansformer-from-scratch-v0.4'


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_dataset(root: Path):
    """Only read explicitly assigned splits. Fail closed on leakage and ambiguity."""
    splits, hashes = {}, {}
    seen_prompts, seen_ids, seen_pairs = {}, set(), set()
    for split in ('train', 'dev', 'test'):
        p = root / (split + '.jsonl')
        if not p.is_file():
            raise FileNotFoundError(f'Missing required split file: {p}')
        hashes[split] = sha256_file(p)
        rows = []
        for line_num, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{p}:{line_num}: JSON parse error: {e}') from e
            for key in ('id', 'prompt', 'response', 'category', 'origin'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{p}:{line_num}: missing/non-string {key}')
            if row['id'] in seen_ids:
                raise ValueError(f'Duplicate example id: {row["id"]}')
            seen_ids.add(row['id'])
            prompt_key = ' '.join(row['prompt'].lower().split())
            pair_key = (prompt_key, ' '.join(row['response'].lower().split()))
            if prompt_key in seen_prompts:
                raise ValueError(f'Prompt leakage/duplicate between {seen_prompts[prompt_key]} and {split}: {row["id"]}')
            if pair_key in seen_pairs:
                raise ValueError(f'Duplicate example: {row["id"]}')
            seen_prompts[prompt_key] = split
            seen_pairs.add(pair_key)
            if not row['prompt'].endswith(('Answer: ', 'Response: ')):
                raise ValueError(f'{p}:{line_num}: prompt must end with Answer: or Response: followed by space')
            if '\x00' in row['response'] or '\x00' in row['prompt']:
                raise ValueError('NUL in training example')
            rows.append(row)
        if not rows:
            raise ValueError(f'Empty data split: {p}')
        splits[split] = rows
    return splits, hashes


def tokenize_examples(rows, tok, context):
    out = []
    for r in rows:
        prefix = [SPECIAL['<BOS>']] + tok.encode(r['prompt'])
        answer = tok.encode(r['response']) + [SPECIAL['<EOS>']]
        full = prefix + answer
        if len(full) < 3 or len(full) - 1 > context:
            raise ValueError(f'Example {r["id"]} has {len(full)-1} input tokens; context limit is {context}. '
                             'Do not silently truncate questions or answers.')
        # x[i] -> y[i] predicts full[i+1]; only targets in the answer span are learned.
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        y[:len(prefix)-1] = -1
        assert (y != -1).sum().item() == len(answer)
        out.append((r, x, y))
    return out


def collate(examples, device):
    length = max(x.numel() for _, x, _ in examples)
    xs = torch.full((len(examples), length), SPECIAL['<PAD>'], dtype=torch.long)
    ys = torch.full((len(examples), length), -1, dtype=torch.long)
    for i, (_, x, y) in enumerate(examples):
        xs[i, :len(x)] = x
        ys[i, :len(y)] = y
    return xs.to(device), ys.to(device)


def get_model(ckpt, device):
    if ckpt.get('format') not in (PARENT_FORMAT, FORMAT):
        raise ValueError('Unsupported checkpoint format')
    load_model_source(None)
    from model import ModelConfig
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    cfg = ModelConfig(**ckpt['config'])
    research = NeuralConfig(**ckpt['research_config'])
    model = NeuralTransformerLM(cfg, research).to(device)
    # The v0.5 runner preserves the pretrained attention implementation.
    if ckpt.get('settings', {}).get('attention') == 'sdpa':
        use_sdpa(model)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, cfg


@torch.no_grad()
def teacher_forced(model, examples, device, amp_ctx, batch_size=4):
    was_training = model.training
    model.eval()
    total, good, total_nll = 0, 0, 0.0
    for start in range(0, len(examples), batch_size):
        x, y = collate(examples[start:start + batch_size], device)
        with amp_ctx:
            logits, _ = model(x)
            logits = logits.float()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  ignore_index=-1, reduction='sum')
        valid = y.ne(-1)
        total += int(valid.sum())
        good += int(((logits.argmax(-1) == y) & valid).sum())
        total_nll += float(nll)
    model.train(was_training)
    return {'masked_answer_ce': total_nll / total,
            'masked_answer_token_accuracy': good / total,
            'answer_tokens': total,
            'examples': len(examples)}


def settings_for(a):
    return {name: getattr(a, name) for name in (
        'steps', 'batch_size', 'grad_accum', 'eval_every', 'save_every', 'seed', 'lr',
        'min_lr', 'warmup_steps', 'weight_decay', 'grad_clip', 'amp'
    )}


def trim_metrics(path, step):
    if not path.exists():
        if step:
            raise ValueError('Missing metrics for resume')
        return
    lines = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    kept = [x for x in lines if x['step'] <= step]
    if [r['step'] for r in kept] != list(range(1, len(kept)+1)):
        raise ValueError('Metrics sequence inconsistent')
    if len(kept) == step-1:
        kept.append({'step': step, 'checkpoint_recovered': True})
    if len(kept) != step:
        raise ValueError('Metrics do not match checkpoint')
    path.write_text(''.join(json.dumps(x) + '\n' for x in kept), encoding='utf-8')


def train(a):
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    if not (a.steps >= 1 and 1 <= a.stop_after <= a.steps and a.batch_size >= 1 and a.grad_accum >= 1):
        raise ValueError('Invalid steps, stop-after, batch-size or grad-accum')
    if not (0 < a.lr and 0 <= a.min_lr <= a.lr and 0 <= a.warmup_steps < a.steps):
        raise ValueError('Invalid learning rate/warmup')
    if a.grad_clip <= 0 or a.eval_every < 1 or a.save_every < 1:
        raise ValueError('Invalid eval/save/clip settings')
    data, hashes = read_dataset(Path(a.data))
    out = Path(a.out)
    if a.resume and a.init_from:
        raise ValueError('Use --init-from on the first run OR --resume later')
    if not a.resume and not a.init_from:
        raise ValueError('First run requires --init-from; subsequent runs require --resume')
    if a.resume:
        path = Path(a.resume)
        if path.resolve().parent != out.resolve():
            raise ValueError('--resume checkpoint must be located in --out')
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if ckpt.get('format') != FORMAT:
            raise ValueError('Not an Expansformer v0.5 checkpoint')
        if hashes != ckpt.get('dataset_sha256') or settings_for(a) != ckpt.get('settings_v05'):
            raise ValueError('Data split or training settings changed; refusing resume')
        if sha256_file(out / 'tokenizer.json') != ckpt['tokenizer_sha256']:
            raise ValueError('Modified/missing tokenizer, cannot resume')
        step0 = int(ckpt['step'])
        trim_metrics(out / 'metrics.jsonl', step0)
    else:
        path = Path(a.init_from)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'{out} is not empty. Use --resume or another output directory.')
        parent_ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if parent_ckpt.get('format') != PARENT_FORMAT:
            raise ValueError('Expected frozen v0.4 transfer/baseline checkpoint')
        tok_path = path.parent / 'tokenizer.json'
        if sha256_file(tok_path) != parent_ckpt['tokenizer_sha256']:
            raise ValueError('Parent tokenizer checksum disagrees with checkpoint')
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tok_path, out / 'tokenizer.json')
        parent_hash = sha256_file(path)
        ckpt = parent_ckpt
        step0 = 0
        dump_json(out / 'provenance.json', {
            'status': 'experimental_unreviewed_instruction_data',
            'parent_path': str(path), 'parent_step': parent_ckpt['step'],
            'parent_format': PARENT_FORMAT, 'parent_sha256': parent_hash,
            'tokenizer_sha256': parent_ckpt['tokenizer_sha256'],
            'dataset_sha256': hashes, 'counts': {k: len(v) for k,v in data.items()},
            'test_reserved': 'test split never read for selection during train',
            'note': 'Bootstrap examples are synthetic and not a substitute for a reviewed instruction corpus.'})
    tok = OwnTokenizer.load(out / 'tokenizer.json')
    if tok.vocab_size != ckpt['config']['vocab_size']:
        raise ValueError('Vocab size mismatch')
    device = get_device(a.device)
    seed_everything(a.seed, device)
    model, cfg = get_model(ckpt, device)
    examples = {k: tokenize_examples(v, tok, cfg.context_length) for k, v in data.items()}
    optimizer = optimizer_for(model, a)
    amp_ctx, scaler, dtype_name = get_amp(device, a.amp)
    sampler = torch.Generator(device='cpu').manual_seed(a.seed + 3127)
    best = float('inf')
    if a.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        if scaler.is_enabled() and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        sampler.set_state(ckpt['sampler_rng'])
        torch.set_rng_state(ckpt['cpu_rng'])
        if device.type == 'cuda' and ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        best = float(ckpt['best_dev_ce'])
    else:
        baseline = teacher_forced(model, examples['dev'], device, amp_ctx, a.batch_size)
        dump_json(out / 'initial_dev.json', baseline)
        print(f'Untrained instruction baseline | dev CE {baseline["masked_answer_ce"]:.4f} | '
              f'dev token accuracy {baseline["masked_answer_token_accuracy"]*100:.2f}%', flush=True)
    if step0 >= a.stop_after:
        print(f'Already at step {step0}; no training performed.')
        return
    parent_meta = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    print(f'Expansformer v0.5 | {device} ({dtype_name}) | '
          f'{sum(p.numel() for p in model.parameters()):,} parameters | '
          f'parent step={parent_meta["parent_step"]} | '
          f'train/dev/test={len(data["train"])}/{len(data["dev"])}/{len(data["test"])} | '
          f'context={cfg.context_length}', flush=True)

    def checkpoint(step):
        return {
            'format': FORMAT, 'step': step, 'config': asdict(cfg),
            'research_config': ckpt['research_config'],
            'settings': ckpt['settings'], # existing attention path and config
            'settings_v05': settings_for(a),
            'dataset_sha256': hashes,
            'tokenizer_sha256': sha256_file(out / 'tokenizer.json'),
            'parent_checkpoint_sha256': parent_meta['parent_sha256'],
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'sampler_rng': sampler.get_state(), 'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'best_dev_ce': best,
        }
    metrics = out / 'metrics.jsonl'
    tic = time.monotonic()
    for step in range(step0, a.stop_after):
        model.train()
        lr = lr_for_step(step, a.steps, a.warmup_steps, a.lr, a.min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        num_tokens = 0
        for _ in range(a.grad_accum):
            choices = torch.randint(len(examples['train']), (a.batch_size,), generator=sampler)
            batch = [examples['train'][int(j)] for j in choices]
            x, y = collate(batch, device)
            with amp_ctx:
                loss, detail = model.training_loss(x, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss / a.grad_accum).backward()
            accumulated += float(loss.detach()) / a.grad_accum
            num_tokens += int(y.ne(-1).sum())
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip))
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        scale0 = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        if scale0 is not None and scaler.get_scale() < scale0:
            raise FloatingPointError('Optimizer update skipped by AMP scaler')
        completed = step + 1
        do_eval = (completed % a.eval_every == 0 or completed == a.stop_after)
        dev = teacher_forced(model, examples['dev'], device, amp_ctx, a.batch_size) if do_eval else None
        is_best = False
        if dev and dev['masked_answer_ce'] < best:
            best = dev['masked_answer_ce']
            is_best = True
        record = {'step': completed, 'train_total_loss': accumulated,
                  'trained_answer_tokens_in_step': num_tokens,
                  'grad_norm': norm, 'lr': lr,
                  'seconds_since_run_start': round(time.monotonic() - tic, 2),
                  **({'dev': dev} if dev else {})}
        if do_eval or completed % a.save_every == 0:
            state = checkpoint(completed)
            save_atomic(state, out / 'latest.pt')
            if is_best:
                save_atomic(state, out / 'best.pt')
        with metrics.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        if completed == 1 or do_eval:
            print(f'update {completed}/{a.steps} | train {accumulated:.4f} | '
                  f'lr {lr:.2g}' + (f' | dev CE {dev["masked_answer_ce"]:.4f} | '
                  f'dev acc {dev["masked_answer_token_accuracy"]*100:.2f}%' if dev else ''), flush=True)
    print('Pilot complete for requested stop-after; original parent checkpoint unchanged.', flush=True)


@torch.no_grad()
def greedy(model, tok, prompt, context, max_new_tokens, device):
    prefix = [SPECIAL['<BOS>']] + tok.encode(prompt)
    if len(prefix) > context:
        raise ValueError('Prompt exceeds context window')
    state = torch.tensor([prefix], dtype=torch.long, device=device)
    generated = []
    model.eval()
    for _ in range(max_new_tokens):
        logits = model.logits_at_depth(state[:, -context:], model.config.n_layers)
        scores = logits[:, -1, :].float()
        scores[:, :3] = -float('inf')
        ix = int(scores.argmax(dim=-1).item())
        if ix == SPECIAL['<EOS>']:
            break
        generated.append(ix)
        state = torch.cat((state, torch.tensor([[ix]], device=device)), dim=1)
    return tok.decode(generated)


def normalized(s):
    return ' '.join(s.casefold().strip().split())


def inference(a):
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    cp = Path(a.checkpoint)
    ckpt = torch.load(cp, map_location='cpu', weights_only=True)
    if ckpt.get('format') not in (FORMAT, PARENT_FORMAT):
        raise ValueError('Expected v0.4 or v0.5 checkpoint')
    tok_path = cp.parent / 'tokenizer.json'
    if sha256_file(tok_path) != ckpt['tokenizer_sha256']:
        raise ValueError('Tokenizer mismatch')
    tok = OwnTokenizer.load(tok_path)
    device = get_device(a.device)
    model, cfg = get_model(ckpt, device)
    model.eval()
    if a.command == 'generate':
        print(greedy(model, tok, a.prompt, cfg.context_length, a.max_new_tokens, device))
        return
    if a.split == 'test' and not a.final_test:
        raise ValueError('The test split is reserved until model selection is finished; pass --final-test explicitly once.')
    data, hashes = read_dataset(Path(a.data))
    if ckpt['format'] == FORMAT and hashes != ckpt['dataset_sha256']:
        raise ValueError('Dataset changed relative to checkpoint')
    rows = data[a.split]
    if a.max_examples and a.max_examples < len(rows):
        # Deterministic category-stratified subset rather than only first examples.
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r['category'], []).append(r)
        subset = []
        while len(subset) < a.max_examples and any(by_cat.values()):
            for cat in sorted(by_cat):
                if by_cat[cat] and len(subset) < a.max_examples:
                    subset.append(by_cat[cat].pop(0))
    else:
        subset = rows
    examples = tokenize_examples(subset, tok, cfg.context_length)
    amp_ctx, _, _ = get_amp(device, a.amp)
    tf = teacher_forced(model, examples, device, amp_ctx, a.batch_size)
    outputs = []
    for row in subset:
        output = greedy(model, tok, row['prompt'], cfg.context_length, a.max_new_tokens, device)
        outputs.append({'id': row['id'], 'category': row['category'],
                        'expected': row['response'], 'generated': output,
                        'exact_match': normalized(output) == normalized(row['response'])})
    result = {'checkpoint': str(cp), 'checkpoint_step': ckpt['step'], 'split': a.split,
              'reviewed_source_claim': False, 'teacher_forced': tf,
              'generated_exact_match': sum(x['exact_match'] for x in outputs)/len(outputs),
              'examples': outputs}
    path = Path(a.out)
    dump_json(path, result)
    print(f'{a.split}: masked CE={tf["masked_answer_ce"]:.4f} | '
          f'token acc={tf["masked_answer_token_accuracy"]*100:.2f}% | '
          f'exact generated answers={100*result["generated_exact_match"]:.1f}% '
          f'({len(outputs)} examples) | {path}')


def lm_check(a):
    """Same old transfer validation texts for baseline and v0.5; dev-only regression signal."""
    from corpus_builder import read_documents
    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
    parent_path = Path(a.parent)
    child_path = Path(a.checkpoint)
    base = torch.load(parent_path, map_location='cpu', weights_only=True)
    child = torch.load(child_path, map_location='cpu', weights_only=True)
    if base.get('format') != PARENT_FORMAT or child.get('format') != FORMAT:
        raise ValueError('Expected v0.4 parent and v0.5 instruction child')
    if base['tokenizer_sha256'] != child['tokenizer_sha256'] or base['config'] != child['config'] or base['research_config'] != child['research_config']:
        raise ValueError('Models differ in tokenizer or architecture')
    if sha256_file(parent_path.parent / 'tokenizer.json') != base['tokenizer_sha256']:
        raise ValueError('Parent tokenizer hash mismatch')
    provenance = json.loads((parent_path.parent / 'provenance.json').read_text(encoding='utf-8'))
    old_val = [r['sha256'] for r in provenance['val']]
    docs = read_documents(Path(a.corpus))
    by_sha = {d.sha256: d for d in docs}
    if len(by_sha) != len(docs) or any(h not in by_sha for h in old_val):
        raise ValueError('Missing/duplicated old validation documents')
    tok = OwnTokenizer.load(parent_path.parent / 'tokenizer.json')
    stream = serialize_stream([by_sha[h] for h in old_val], tok)
    device = get_device(a.device)
    amp_ctx, _, _ = get_amp(device, a.amp)
    scores = {}
    for label, cp in (('parent', base), ('instruction', child)):
        model, cfg = get_model(cp, device)
        model.eval()
        scores[label] = lm_evaluate(model, stream, tok, cfg.context_length,
                                    a.batch_size, a.windows, device, amp_ctx)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = {'note': 'Development-held-out documents; NOT an untouched final test.',
              'parent_checkpoint': str(parent_path), 'instruction_checkpoint': str(child_path),
              'scores': scores,
              'delta_instruction_minus_parent': {
                  name: scores['instruction'][name] - scores['parent'][name]
                  for name in ('val_loss', 'val_token_accuracy', 'val_bits_per_byte')}}
    dump_json(Path(a.out), report)
    print(json.dumps(report['delta_instruction_minus_parent'], indent=2))
    print('Saved:', a.out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='Fine-tune answer tokens only; no test evaluation')
    t.add_argument('--data', required=True)
    t.add_argument('--out', default='./runs/v05_instruction_pilot')
    group = t.add_mutually_exclusive_group(required=True)
    group.add_argument('--init-from')
    group.add_argument('--resume')
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--stop-after', type=int, default=25)
    t.add_argument('--batch-size', type=int, default=4)
    t.add_argument('--grad-accum', type=int, default=4)
    t.add_argument('--lr', type=float, default=2e-5)
    t.add_argument('--min-lr', type=float, default=2e-6)
    t.add_argument('--warmup-steps', type=int, default=15)
    t.add_argument('--weight-decay', type=float, default=0.01)
    t.add_argument('--grad-clip', type=float, default=1.0)
    t.add_argument('--eval-every', type=int, default=25)
    t.add_argument('--save-every', type=int, default=25)
    t.add_argument('--amp', choices=['auto','bf16','fp16','off'], default='bf16')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--cpu-threads', type=int, default=2)
    t.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')

    for mode in ('evaluate', 'generate'):
        x = sub.add_parser(mode, help='Use checkpoint for generation or scored evaluation')
        x.add_argument('--checkpoint', required=True)
        x.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
        x.add_argument('--cpu-threads', type=int, default=2)
        x.add_argument('--max-new-tokens', type=int, default=48)
        if mode == 'generate':
            x.add_argument('--prompt', default='Question: What is three plus four?\nAnswer: ')
        else:
            x.add_argument('--data', required=True)
            x.add_argument('--split', choices=['dev', 'test'], default='dev')
            x.add_argument('--out', default='./runs/v05_instruction_pilot/evaluation.json')
            x.add_argument('--max-examples', type=int, default=12)
            x.add_argument('--final-test', action='store_true', help='Explicitly unlock reserved test split after model selection')
            x.add_argument('--batch-size', type=int, default=4)
            x.add_argument('--amp', choices=['auto','bf16','fp16','off'], default='bf16')

    r = sub.add_parser('lm-check', help='Measure language-model regression on inherited validation documents')
    r.add_argument('--parent', required=True)
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--corpus', required=True)
    r.add_argument('--out', default='./runs/v05_instruction_pilot/lm_regression.json')
    r.add_argument('--windows', type=int, default=64)
    r.add_argument('--batch-size', type=int, default=4)
    r.add_argument('--amp', choices=['auto','bf16','fp16','off'], default='bf16')
    r.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    r.add_argument('--cpu-threads', type=int, default=2)
    return p.parse_args()


def main():
    a = parse_args()
    if a.command == 'train':
        train(a)
    elif a.command in ('generate', 'evaluate'):
        inference(a)
    elif a.command == 'lm-check':
        lm_check(a)


if __name__ == '__main__':
    main()
