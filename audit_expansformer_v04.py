#!/usr/bin/env python3
"""Read-only capability/architecture audit for trained Expansformer v0.4.

Designed for the Expansformer_v03_Research project, the 1K original checkpoint,
and the 800-step 431-document transfer checkpoint.  Never trains or overwrites
checkpoint weights.  Includes matched held-out next-token evaluation, trained
intermediate-depth analysis, MoE routing, head calibration, memory-gate
perturbation, latency estimates and deterministic qualitative prompts.

IMPORTANT: An inference-only perturbation is NOT a matched training ablation.
The inherited/new validation sets are not untouched final test sets.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F

DEFAULT_PROBES = [
    {"id": "definition", "prompt": "Question: What is artificial intelligence?\nAnswer:"},
    {"id": "comparison", "prompt": "Question: What is one difference between human and artificial intelligence?\nAnswer:"},
    {"id": "arithmetic", "prompt": "Question: What is 12 plus 7?\nAnswer:"},
    {"id": "logic", "prompt": "Question: If all cats are mammals and Luna is a cat, is Luna a mammal?\nAnswer:"},
    {"id": "instruction", "prompt": "Instruction: Give exactly one short sentence about memory.\nResponse:"},
]


def hash_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.tmp_', delete=False) as out:
        tmp = Path(out.name)
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')
    os.replace(tmp, path)


def freeze_one(src: Path, target: Path, src_hash: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hash_file(target) != src_hash:
            raise ValueError(f'Refusing to overwrite different frozen checkpoint: {target}')
        return 'previous copy verified'
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.checkpoint_', delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copyfile(src, temporary)
        if hash_file(temporary) != src_hash:
            raise IOError(f'Copied checkpoint hash mismatch: {src}')
        # No replace if target appeared concurrently; do not clobber preexisting copy.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if hash_file(target) != src_hash:
                raise ValueError(f'Unexpected different frozen checkpoint: {target}')
        return 'copy created and hash verified'
    finally:
        temporary.unlink(missing_ok=True)


def get_group_docs(original_file: Path, transfer_file: Path, data_dir: Path, read_documents):
    old_p = json.loads((original_file.parent / 'provenance.json').read_text())
    new_p = json.loads((transfer_file.parent / 'provenance.json').read_text())
    old_train = {x['sha256'] for x in old_p['train']}
    old_val_list = [x['sha256'] for x in old_p['val']]
    old_val = set(old_val_list)
    new_train = {x['sha256'] for x in new_p['train']}
    new_val = [x['sha256'] for x in new_p['val']]
    new_val_set = set(new_val)
    if len(old_val_list) != len(old_val) or len(new_val_set) != len(new_val):
        raise ValueError('Duplicate validation SHA-256 records in provenance')
    if old_train & new_val_set or not old_val <= new_val_set or old_val & new_train or new_train & new_val_set:
        raise ValueError('Invalid inherited train/validation split; audit aborted')
    original_train_set = old_train | old_val
    if not original_train_set <= new_train | new_val_set:
        raise ValueError('Original dataset was not fully preserved in the transfer split')
    added_val = [s for s in new_val if s not in old_val]
    if not added_val:
        raise ValueError('No newly held-out documents in transfer provenance')
    docs = read_documents(data_dir)
    lookup = {d.sha256: d for d in docs}
    if len(lookup) != len(docs):
        raise ValueError('Repeated content hashes in prepared corpus')
    if set(old_val_list + added_val) - lookup.keys():
        raise ValueError('Prepared corpus missing validation text hashes')
    return {
        'inherited_validation': [lookup[s] for s in old_val_list],
        'new_validation': [lookup[s] for s in added_val],
    }, {'original_train_documents': len(old_train), 'original_validation_documents': len(old_val),
        'transfer_train_documents': len(new_train), 'transfer_validation_documents': len(new_val),
        'new_validation_documents': len(added_val), 'prepared_documents': len(docs)}


def load_ckpt(path: Path, *, expected_step: int | None = None):
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('format') != 'expansformer-from-scratch-v0.4':
        raise ValueError(f'Not an Expansformer v0.4 checkpoint: {path}')
    if expected_step is not None and ckpt['step'] != expected_step:
        raise ValueError(f'{path}: expected step {expected_step}, got {ckpt["step"]}')
    return ckpt


def make_model(ckpt, cfg_class, model_class, neural_class, use_sdpa, device):
    cfg = cfg_class(**ckpt['config'])
    model = model_class(cfg, neural_class(**ckpt['research_config']))
    if ckpt['settings']['attention'] == 'sdpa':
        use_sdpa(model)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()
    return model


def gpu_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def window_positions(length: int, context: int, max_windows: int):
    blocks = (length - 1) // context
    n = min(blocks, max_windows)
    if n < 1:
        raise ValueError(f'Not enough tokens to evaluate (len={length}, ctx={context})')
    positions = ([blocks // 2] if n == 1 else
                 [(i * (blocks - 1)) // (n - 1) for i in range(n)])
    if len(set(positions)) != len(positions):
        raise AssertionError('Evaluation windows must not overlap')
    return positions


def matched_score(model, stream, tok, cfg, evaluate, device, amp, windows: int, batch_size: int):
    with torch.inference_mode():
        return evaluate(model, stream, tok, cfg.context_length, batch_size, windows, device, amp)


def calibration_update(bins, pred_probs, correct):
    ps, ys = pred_probs.float().flatten(), correct.float().flatten()
    ps = ps.clamp(0., 1.)
    slots = (ps * len(bins)).long().clamp(max=len(bins)-1)
    for i in range(len(bins)):
        mask = slots == i
        if mask.any():
            bins[i]['n'] += int(mask.sum().item())
            bins[i]['prob_sum'] += float(ps[mask].sum().item())
            bins[i]['correct_sum'] += float(ys[mask].sum().item())
            bins[i]['brier_sum'] += float(((ps[mask]-ys[mask])**2).sum().item())


def depth_and_routing(model, stream, tok, ctx, device, amp, windows, thresholds):
    """One true forward traversal: per-depth logits, router counts, head estimates."""
    chosen = window_positions(len(stream), ctx, windows)
    n_layers = model.config.n_layers
    loss = [0.] * n_layers
    accurate = [0] * n_layers
    agreement = [0] * n_layers
    total = 0
    totals_by_layer = [[0] * model.research_config.moe_experts for _ in range(n_layers)]
    last_token_scores = {str(t): {'correct': 0, 'full_correct': 0,
                                  'agree': 0, 'early_count': 0, 'depth_sum': 0} for t in thresholds}
    head_bins = [[{'n': 0, 'prob_sum': 0., 'correct_sum': 0., 'brier_sum': 0.}
                  for _ in range(10)] for _ in range(n_layers - 1)]
    last_token_windows = 0
    from model import ModelConfig   # model source already registered by loader
    del ModelConfig
    with torch.inference_mode():
        for block_idx in chosen:
            start = block_idx * ctx
            x_ids = stream[start:start+ctx].unsqueeze(0).to(device)
            y = stream[start+1:start+ctx+1].unsqueeze(0).to(device)
            with amp:
                x = model.token_embed(x_ids)
                if not model.config.rope:
                    x = x + model.pos_embed(torch.arange(ctx, device=device))
                x = model.drop(x)
                rope = model.rope_freqs if model.config.rope else None
                depth_predictions = []
                head_confidences = []
                for layer_idx, block in enumerate(model.blocks):
                    x = block(x, rope)
                    if model.slot_memory is not None:
                        x = model.slot_memory(x, layer_idx)
                    if getattr(block.ffn, 'last_routing_counts', None) is not None:
                        counts = block.ffn.last_routing_counts.detach().cpu().tolist()
                        for i, c in enumerate(counts):
                            totals_by_layer[layer_idx][i] += int(c)
                    norm = model.norm if layer_idx == n_layers-1 else model.exit_norms[layer_idx]
                    logits = model.lm_head(norm(x)).float()
                    depth_predictions.append(logits)
                    if layer_idx < n_layers - 1 and model.next_token_heads:
                        head_confidences.append(torch.sigmoid(model.next_token_heads[layer_idx](x).float().squeeze(-1)))
            full_pred = depth_predictions[-1].argmax(dim=-1)
            for layer_idx, logits in enumerate(depth_predictions):
                per_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                          y.reshape(-1), reduction='sum')
                loss[layer_idx] += float(per_tok.item())
                prediction = logits.argmax(dim=-1)
                accurate[layer_idx] += int((prediction == y).sum().item())
                agreement[layer_idx] += int((prediction == full_pred).sum().item())
                if layer_idx < n_layers - 1 and head_confidences:
                    calibration_update(head_bins[layer_idx], head_confidences[layer_idx], prediction == y)
            total += y.numel()
            last_token_windows += 1
            for t in thresholds:
                picked_layer = n_layers - 1
                for l, confidences in enumerate(head_confidences):
                    if float(confidences[0, -1].item()) >= t:
                        picked_layer = l
                        break
                pred = int(depth_predictions[picked_layer][0, -1].argmax().item())
                full = int(full_pred[0, -1].item())
                gold = int(y[0, -1].item())
                slot = last_token_scores[str(t)]
                slot['correct'] += int(pred == gold)
                slot['full_correct'] += int(full == gold)
                slot['agree'] += int(pred == full)
                slot['early_count'] += int(picked_layer < n_layers - 1)
                slot['depth_sum'] += picked_layer + 1
    depth_rows = []
    for idx in range(n_layers):
        depth_rows.append({'depth': idx+1, 'cross_entropy': loss[idx]/total,
                           'exact_next_token_accuracy': accurate[idx]/total,
                           'agreement_with_full': agreement[idx]/total,
                           'evaluated_tokens': total})
    router_rows = []
    for idx, counts in enumerate(totals_by_layer):
        count_sum = sum(counts)
        shares = [c/count_sum for c in counts] if count_sum else []
        router_rows.append({'layer': idx+1, 'topk_selection_counts': counts,
                            'topk_selection_shares': shares,
                            'active_experts': sum(1 for x in counts if x > 0),
                            'total_assignments': count_sum})
    calib_rows = []
    for idx, bins in enumerate(head_bins):
        n = sum(b['n'] for b in bins)
        calib_rows.append({'depth': idx+1, 'examples': n,
                           'mean_predicted_correctness': sum(b['prob_sum'] for b in bins)/n if n else None,
                           'observed_early_accuracy': sum(b['correct_sum'] for b in bins)/n if n else None,
                           'brier': sum(b['brier_sum'] for b in bins)/n if n else None,
                           'ece_10_bins': sum(abs(b['prob_sum']-b['correct_sum']) for b in bins)/n if n else None})
    auto = {}
    for t, vals in last_token_scores.items():
        auto[t] = {'windows': last_token_windows,
                   'mean_selected_depth': vals['depth_sum']/last_token_windows,
                   'early_exit_fraction': vals['early_count']/last_token_windows,
                   'selected_last_token_accuracy': vals['correct']/last_token_windows,
                   'full_last_token_accuracy': vals['full_correct']/last_token_windows,
                   'selected_vs_full_agreement': vals['agree']/last_token_windows}
    return {'depths': depth_rows, 'moe_routing': router_rows, 'prediction_head_calibration': calib_rows,
            'adaptive_thresholds_last_token_only': auto, 'samples': len(chosen)}


@contextmanager
def suppressed_memory(model):
    memory = model.slot_memory
    if memory is None:
        yield
        return
    with torch.no_grad():
        saved = memory.gates.detach().clone()
        memory.gates.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            memory.gates.copy_(saved)


def time_depths(model, stream, context, device, amp, repeats=5):
    # Measures whole-model time, not MACs or actual platform energy savings.
    starts = window_positions(len(stream), context, 1)
    x = stream[starts[0]*context:starts[0]*context+context].unsqueeze(0).to(device)
    times = []
    with torch.inference_mode():
        for depth in range(1, model.config.n_layers+1):
            with amp:
                for _ in range(2):
                    model.logits_at_depth(x, depth)
                gpu_sync(device)
                samples = []
                for _ in range(repeats):
                    gpu_sync(device)
                    start = time.perf_counter()
                    model.logits_at_depth(x, depth)
                    gpu_sync(device)
                    samples.append(1000*(time.perf_counter()-start))
            samples.sort()
            times.append({'depth': depth, 'milliseconds_median': samples[len(samples)//2],
                          'sequence_length': context, 'batch_size': 1, 'runs': repeats})
    return times


def generate_greedy(model, tok, prompt, device, max_new_tokens):
    from train_expansformer_v01 import SPECIAL
    ids = tok.encode(prompt) or [SPECIAL['<BOS>']]
    current = torch.tensor([ids], dtype=torch.long, device=device)
    output = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model.logits_at_depth(current[:, -model.config.context_length:], model.config.n_layers)
            scores = logits[0,-1].float().clone()
            scores[:3] = -float('inf')
            next_id = int(scores.argmax().item())
            if next_id == SPECIAL['<EOS>']:
                break
            output.append(next_id)
            current = torch.cat([current, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(output)


def write_summary(path: Path, results):
    original = results['checkpoints']['original']
    transfer = results['checkpoints']['transfer']
    lines = [
        '# Expansformer v0.4 — automated capability and architecture audit', '',
        f"Generated: {results['timestamp_utc']}", '',
        f"Original step: {original['step']} | Transfer step: {transfer['step']}",
        f"Original SHA-256: `{original['sha256']}`",
        f"Transfer SHA-256: `{transfer['sha256']}`", '',
        '## Matched next-token evaluation (development validation)', '',
        '| Cohort | Original accuracy | Transfer accuracy | Delta (pp) | Original loss | Transfer loss |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for cohort, row in results.get('matched', {}).items():
        a, b = row['original'], row['transfer']
        lines.append(f"| {cohort} | {a['val_token_accuracy']*100:.2f}% | {b['val_token_accuracy']*100:.2f}% | "
                     f"{(b['val_token_accuracy']-a['val_token_accuracy'])*100:+.2f} | "
                     f"{a['val_loss']:.4f} | {b['val_loss']:.4f} |")
    if 'depth' in results:
        lines.extend(['', '## Transfer model: intermediate depth', '',
                      '| Depth | Next-token accuracy | Cross-entropy | Agreement with full |',
                      '|---|---:|---:|---:|'])
        for row in results['depth']['depths']:
            lines.append(f"| {row['depth']} | {100*row['exact_next_token_accuracy']:.2f}% | "
                         f"{row['cross_entropy']:.4f} | {100*row['agreement_with_full']:.2f}% |")
        lines.extend(['', '### Expert routing', '',
                      'Selection frequencies are measured. They do not demonstrate beneficial specialization.'])
        for r in results['depth']['moe_routing']:
            lines.append(f"- Layer {r['layer']}: {r['active_experts']} active experts; selected shares "
                         + ', '.join(f'{p:.1%}' for p in r['topk_selection_shares']))
        lines.extend(['', '### Confidence heads', '',
                      '| Exit | Predicted correctness | Observed correctness | Brier | ECE |',
                      '|---|---:|---:|---:|---:|'])
        for r in results['depth']['prediction_head_calibration']:
            if r['examples']:
                lines.append(f"| {r['depth']} | {r['mean_predicted_correctness']:.2%} | "
                             f"{r['observed_early_accuracy']:.2%} | {r['brier']:.4f} | "
                             f"{r['ece_10_bins']:.4f} |")
    if 'memory' in results:
        m = results['memory']
        lines.extend(['', '## Memory gate sensitivity (NOT a retrained ablation)', '',
                      f"Normal CE: {m['normal']['val_loss']:.4f}; with gates zeroed: "
                      f"{m['gates_zeroed']['val_loss']:.4f}; "
                      f"CE difference: {m['ce_delta_when_zeroed']:+.4f}.",
                      'Only an inference-time intervention; cannot measure what a separately trained no-memory model would learn.'])
    if 'timing' in results:
        lines.extend(['', '## Inference timing, one batch of one context window', ''])
        for row in results['timing']:
            lines.append(f"- Depth {row['depth']}: median {row['milliseconds_median']:.2f} ms")
    if 'generation' in results:
        lines.extend(['', '## Qualitative generation probes (ungraded)', '',
                      'These are deterministic continuations, NOT validated reasoning scores.'])
        for row in results['generation']:
            lines.append(f"\n**{row['id']}** — {row['prompt']}\n")
            for key in ('original', 'transfer'):
                text = row.get(key, '').replace('\r', '')
                lines.append(f"- {key}: `{text[:450].replace('`', '′')}`")
    lines.extend(['', '## Evidence limits', '',
                  '- Neither development validation cohort is an untouched final test set; checkpoints were selected using validation scores.',
                  '- Small-window evaluation is not necessarily full-corpus evaluation; inspect coverage in `audit.json`.',
                  '- Inference-time gate perturbation is not an ablation trained from scratch with matched compute.',
                  '- MoE selection distribution is not proof that experts have useful specialization.',
                  '- The memory slots are trained parametric slots, not an episodic or retrieval memory system.',
                  '- Confidence heads predict next-token argmax correctness, not factual truth or certainty.',
                  '- Generated responses are qualitative samples, not automatically scored on factual/reasoning quality.',
                  '- To demonstrate module advantage, train matched controlled models with and without each component.',
                  '- A new instruction/QA dataset must have separate train/dev/test partitions and a provenance review before fine-tuning.',
                  '', '## Next research phase', '',
                  '1. Assemble a separate provenance-tracked question-answer/instruction dataset; do not train on the qualitative probes.',
                  '2. Hold out genuinely untouched tasks before any instruction fine-tuning.',
                  '3. Train controlled, compute-matched no-memory, dense-FFN, no-exit variants to establish architectural contributions.',
                  '4. Record accuracy, quality, speed, VRAM, and seeds across runs.'])
    if results.get('errors'):
        lines.extend(['', '## Nonfatal audit errors', ''])
        lines.extend(f"- {e['stage']}: {e['error']}" for e in results['errors'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--original', type=Path, default=Path('runs/v04_rtx5050_enhanced/best.pt'))
    ap.add_argument('--transfer', type=Path, default=Path('runs/v04_transfer_431/best.pt'))
    ap.add_argument('--data', type=Path, default=Path('runs/v04_staged_research_data_499'))
    ap.add_argument('--out', type=Path, default=Path('runs/v04_capability_audit'))
    ap.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    ap.add_argument('--windows', type=int, default=128, help='Non-overlapping windows PER cohort for matched comparison')
    ap.add_argument('--depth-windows', type=int, default=12, help='Windows for depth, routing, confidence and threshold probes')
    ap.add_argument('--memory-windows', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--new-tokens', type=int, default=64, help='Greedy completion length for each qualitative prompt')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[.5, .75, .9, .95])
    ap.add_argument('--no-freeze', action='store_true', help='Skip verified frozen checkpoint copies (~600MB disk total)')
    ap.add_argument('--no-generation', action='store_true')
    ap.add_argument('--no-timing', action='store_true')
    ap.add_argument('--expect-original', type=int, default=1000)
    ap.add_argument('--expect-transfer', type=int, default=800)
    args = ap.parse_args(argv)
    if any(x < 1 for x in (args.windows,args.depth_windows,args.memory_windows,args.batch_size,args.new_tokens)):
        ap.error('window counts, batch size and new tokens must be positive')
    if any(not (0 < t <= 1) for t in args.thresholds):
        ap.error('thresholds must lie in (0,1]')
    for f in (args.original,args.transfer,args.data):
        if not f.exists():
            ap.error(f'Missing input: {f}')
    args.out.mkdir(parents=True, exist_ok=True)
    from corpus_builder import read_documents
    from train_expansformer_v01 import OwnTokenizer, load_model_source
    from train_expansformer_v04_transfer import FORMAT, get_device, get_amp, serialize_stream, evaluate, use_sdpa
    del FORMAT
    load_model_source(None)
    from model import ModelConfig
    from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
    device = get_device(args.device)
    c1 = load_ckpt(args.original, expected_step=args.expect_original)
    c2 = load_ckpt(args.transfer, expected_step=args.expect_transfer)
    if c1['config'] != c2['config'] or c1['research_config'] != c2['research_config']:
        raise ValueError('Refusing mismatched architectures')
    tok_sha = c1['tokenizer_sha256']
    if tok_sha != c2['tokenizer_sha256']:
        raise ValueError('Tokenizer IDs differ; cannot run matched evaluation')
    for p in (args.original,args.transfer):
        if hash_file(p.parent / 'tokenizer.json') != tok_sha:
            raise ValueError(f'Tokenizer SHA-256 mismatch: {p.parent}')
    tokenizer = OwnTokenizer.load(args.original.parent / 'tokenizer.json')
    cfg = ModelConfig(**c1['config'])
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError('Model/tokenizer vocabulary mismatch')
    cohorts, splits = get_group_docs(args.original,args.transfer,args.data,read_documents)
    streams = {k: serialize_stream(v,tokenizer) for k,v in cohorts.items()}
    start_time = datetime.now(timezone.utc).isoformat()
    results = {'timestamp_utc': start_time,
               'torch_version': torch.__version__, 'device': str(device),
               'gpu': torch.cuda.get_device_name(device) if device.type=='cuda' else None,
               'checkpoints': {
                   'original': {'path': str(args.original), 'step': c1['step'], 'sha256': hash_file(args.original)},
                   'transfer': {'path': str(args.transfer), 'step': c2['step'], 'sha256': hash_file(args.transfer)}},
               'tokenizer_sha256': tok_sha, 'cohort_document_counts': splits,
               'evaluation_settings': {'matched_windows_per_cohort': args.windows,
                    'depth_windows': args.depth_windows,'memory_windows':args.memory_windows,
                    'batch_size': args.batch_size,'thresholds':args.thresholds}, 'errors': []}
    if not args.no_freeze:
        print('Freezing checkpoint copies (without overwriting existing files)...', flush=True)
        results['frozen'] = {}
        for role, source in (('original', args.original), ('transfer', args.transfer)):
            frozen = args.out / 'frozen' / f'{role}_{results["checkpoints"][role]["step"]}.pt'
            status = freeze_one(source,frozen,results['checkpoints'][role]['sha256'])
            results['frozen'][role] = {'path':str(frozen),'status':status}
            print(f'  {role}: {status}', flush=True)
    print(f"Auditing {device}; cohort sizes "
          f"{splits['original_validation_documents']} inherited / {splits['new_validation_documents']} new.", flush=True)
    results['matched'] = {group: {} for group in streams}
    original_model = None
    transfer_model = None
    try:
        for role, checkpoint in (('original',c1),('transfer',c2)):
            model = make_model(checkpoint,ModelConfig,NeuralTransformerLM,NeuralConfig,use_sdpa,device)
            amp, _, _ = get_amp(device, checkpoint['settings']['amp'])
            for name, stream in streams.items():
                score = matched_score(model,stream,tokenizer,cfg,evaluate,device,amp,args.windows,args.batch_size)
                results['matched'][name][role] = score
                print(f"  {role:8} {name:24} loss={score['val_loss']:.4f}, accuracy={score['val_token_accuracy']*100:.2f}%",flush=True)
            if role == 'original' and not args.no_generation:
                original_model = model
            elif role == 'transfer':
                transfer_model = model
            else:
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        if transfer_model is None:
            raise RuntimeError('Internal error: transfer model not loaded')
        amp,_,_ = get_amp(device,c2['settings']['amp'])
        print('Auditing intermediate exits, expert routing and prediction-head calibration...',flush=True)
        results['depth'] = depth_and_routing(transfer_model,streams['inherited_validation'],tokenizer,
                   cfg.context_length,device,amp,args.depth_windows,args.thresholds)
        if transfer_model.slot_memory is not None:
            print('Testing temporary memory-gate suppression (inference sensitivity only)...',flush=True)
            original_score = matched_score(transfer_model,streams['inherited_validation'],tokenizer,
                         cfg,evaluate,device,amp,args.memory_windows,args.batch_size)
            with suppressed_memory(transfer_model):
                disabled = matched_score(transfer_model,streams['inherited_validation'],tokenizer,
                         cfg,evaluate,device,amp,args.memory_windows,args.batch_size)
            results['memory'] = {'normal':original_score,'gates_zeroed':disabled,
                    'ce_delta_when_zeroed':disabled['val_loss']-original_score['val_loss'],
                    'interpretation':'Inference perturbation only; not matched retraining or proof of benefit.'}
        if not args.no_timing:
            print('Timing trained intermediate-depth inference...',flush=True)
            results['timing'] = time_depths(transfer_model,streams['inherited_validation'],
                                            cfg.context_length,device,amp)
        if not args.no_generation:
            print('Generating deterministic qualitative continuations...',flush=True)
            results['generation'] = []
            for probe in DEFAULT_PROBES:
                item = dict(probe)
                for role, model in (('original', original_model),('transfer',transfer_model)):
                    item[role] = generate_greedy(model,tokenizer,probe['prompt'],device,args.new_tokens)
                results['generation'].append(item)
                print(f"  probe: {probe['id']}",flush=True)
            with (args.out / 'manual_quality_review.csv').open('w', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['id','checkpoint','prompt','completion','factual_accuracy_0_to_2',
                                 'instruction_following_0_to_2','coherence_0_to_2','reviewer_notes'])
                for probe in results['generation']:
                    for role in ('original','transfer'):
                        writer.writerow([probe['id'],role,probe['prompt'],probe[role],'','','',''])
    finally:
        del original_model,transfer_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        atomic_json(args.out / 'audit.json',results)
        write_summary(args.out / 'REPORT.md',results)
    print(f"Complete. Report: {args.out / 'REPORT.md'}",flush=True)
    print(f"Machine-readable results: {args.out / 'audit.json'}",flush=True)
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'AUDIT FAILED: {exc}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
