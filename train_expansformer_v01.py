#!/usr/bin/env python3
"""Expansformer v0.1 — from-scratch next-token training on user-supplied text.

Uses TransformerLM from the user's ORIGINAL expansformer_model.zip (model.py),
with a corrected, self-contained byte/BPE tokenizer and step-based trainer.
No pretrained model or pretrained tokenizer is loaded.

Requires: Python 3.10+, PyTorch, and expansformer_model.zip beside this file,
or --source PATH pointing to the ZIP or extracted model directory.

Examples:
  python train_expansformer_v01.py smoke
  python train_expansformer_v01.py train --data ./corpus --out ./runs/micro --steps 1000
  python train_expansformer_v01.py train --data ./corpus --out ./runs/bpe --tokenizer bpe --vocab-size 512 --steps 1000
  python train_expansformer_v01.py train --data ./corpus --out ./runs/micro --steps 1000 --stop-after 250
  python train_expansformer_v01.py train --data ./corpus --out ./runs/micro --resume ./runs/micro/latest.pt --steps 1000
  python train_expansformer_v01.py generate --checkpoint ./runs/micro/best.pt --prompt 'Hello'

This trains from random initialization. It does NOT export GGUF or import into Ollama.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
BYTE_OFFSET = len(SPECIAL)


class OwnTokenizer:
    """Reversible byte tokenizer, optionally trained with deterministic BPE.

    Unlike the supplied tokenizer.py, BPE training and encoding both index
    bytes at BYTE_OFFSET + byte_value. Merge IDs are persistent and ordered.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self.merge_to_id = {tuple(pair): BYTE_OFFSET + 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {BYTE_OFFSET + b: bytes([b]) for b in range(256)}
        for i, (left, right) in enumerate(self.merges):
            self.vocab[BYTE_OFFSET + 256 + i] = self.vocab[left] + self.vocab[right]

    @property
    def vocab_size(self) -> int:
        return BYTE_OFFSET + 256 + len(self.merges)

    def train_bpe(self, text: str, target_vocab: int, max_bytes: int):
        if target_vocab < BYTE_OFFSET + 256:
            raise ValueError("BPE vocabulary must contain at least 260 entries")
        if max_bytes < 1024:
            raise ValueError("--tokenizer-sample-bytes must be at least 1024")
        raw = text.encode("utf-8")[:max_bytes]
        ids = [BYTE_OFFSET + value for value in raw]
        target_merges = target_vocab - (BYTE_OFFSET + 256)
        print(f"Training BPE from {len(ids):,} bytes (up to {target_merges:,} merges)...", flush=True)
        for i in range(target_merges):
            frequencies = Counter(zip(ids, ids[1:]))
            if not frequencies:
                break
            pair, count = frequencies.most_common(1)[0]
            if count < 2:
                break
            new_id = BYTE_OFFSET + 256 + i
            left, right = pair
            merged = []
            at = 0
            while at < len(ids):
                if at + 1 < len(ids) and ids[at] == left and ids[at + 1] == right:
                    merged.append(new_id)
                    at += 2
                else:
                    merged.append(ids[at])
                    at += 1
            ids = merged
            self.merges.append(pair)
            self.merge_to_id[pair] = new_id
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} merges | {len(ids):,} sample tokens", flush=True)
        print(f"Tokenizer vocabulary: {self.vocab_size:,}", flush=True)

    def encode(self, text: str) -> list[int]:
        symbols = [BYTE_OFFSET + v for v in text.encode("utf-8")]
        n = len(symbols)
        if n < 2 or not self.merges:
            return symbols
        # Linked list + priority heap: choose earliest-trained adjacent merge.
        # This avoids scanning the entire corpus once per merge during encoding.
        next_idx = list(range(1, n)) + [-1]
        prev_idx = [-1] + list(range(n - 1))
        alive = bytearray(b"\x01") * n
        heap = []

        def offer(i: int):
            if i < 0 or not alive[i]:
                return
            j = next_idx[i]
            if j < 0:
                return
            pair = (symbols[i], symbols[j])
            rank = self.merge_to_id.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i, j, symbols[i], symbols[j]))

        for idx in range(n - 1):
            offer(idx)
        while heap:
            rank, i, j, left, right = heapq.heappop(heap)
            if not alive[i] or not alive[j] or next_idx[i] != j:
                continue
            if symbols[i] != left or symbols[j] != right:
                continue
            symbols[i] = rank
            alive[j] = 0
            k = next_idx[j]
            next_idx[i] = k
            if k != -1:
                prev_idx[k] = i
            offer(prev_idx[i])
            offer(i)
        output = []
        current = 0
        while current != -1:
            output.append(symbols[current])
            current = next_idx[current]
        return output

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i in self.vocab).decode("utf-8", errors="replace")

    def save(self, path: Path):
        path.write_text(json.dumps({"format": "expansformer-byte-bpe-v1", "merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OwnTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "expansformer-byte-bpe-v1":
            raise ValueError(f"Tokenizer format mismatch: {path}")
        return cls([tuple(p) for p in data["merges"]])


def load_model_source(source: str | None):
    """Import the original TransformerLM without copying/changing its source."""
    candidates = ([Path(source)] if source else []) + [
        Path(__file__).resolve().with_name("expansformer_model.zip"),
        Path.cwd() / "expansformer_model.zip",
        Path(__file__).resolve().parent / "expansformer_model",
        Path.cwd() / "expansformer_model",
    ]
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                if "expansformer_model/model.py" in zf.namelist():
                    sys.path.insert(0, f"{path.resolve()}/expansformer_model")
                    from model import ModelConfig, TransformerLM
                    return ModelConfig, TransformerLM
        if path.is_dir():
            folder = path if (path / "model.py").exists() else path / "expansformer_model"
            if (folder / "model.py").exists():
                sys.path.insert(0, str(folder.resolve()))
                from model import ModelConfig, TransformerLM
                return ModelConfig, TransformerLM
    raise FileNotFoundError(
        "Cannot find original model.py. Put expansformer_model.zip beside this script "
        "or pass --source /path/to/expansformer_model.zip"
    )


def read_corpus(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise ValueError(f"No .txt or .md files found at {path}")
    docs = []
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            docs.append((str(p.relative_to(path) if path.is_dir() else p.name), raw))
    if not docs:
        raise ValueError("Corpus contains no nonempty text documents")
    return docs


def split_corpus(docs: list[tuple[str, str]], seed: int) -> tuple[str, str]:
    if len(docs) == 1:
        content = docs[0][1]
        split = int(len(content) * 0.9)
        return content[:split], content[split:]
    shuffled = docs[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * 0.1))
    return "\n\n".join(text for _, text in shuffled[n_val:]), "\n\n".join(text for _, text in shuffled[:n_val])


def get_texts(args) -> tuple[str, str]:
    data = read_corpus(Path(args.data))
    if args.val_data:
        train_text = "\n\n".join(t for _, t in data)
        val_text = "\n\n".join(t for _, t in read_corpus(Path(args.val_data)))
    else:
        train_text, val_text = split_corpus(data, args.seed)
    return train_text, val_text


def dataset_signature(train_text: str, val_text: str) -> str:
    h = hashlib.sha256()
    for content in (train_text, val_text):
        data = content.encode("utf-8")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def lr_for_step(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return floor
    progress = max(0, min(1, (step - warmup) / max(1, total - warmup)))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def choose_device(name: str):
    result = "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)
    device = torch.device(result)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch does not detect a compatible GPU")
    return device


def make_precision(device, enabled):
    if device.type != "cuda" or not enabled:
        return nullcontext(), torch.amp.GradScaler("cuda", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype), torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)


def sample_batch(ids: torch.Tensor, context: int, batch_size: int, generator: torch.Generator, device):
    start = torch.randint(0, ids.numel() - context, (batch_size,), generator=generator)
    offsets = torch.arange(context)
    positions = start[:, None] + offsets[None, :]
    x = ids[positions].to(device)
    y = ids[positions + 1].to(device)
    return x, y


@torch.no_grad()
def evaluate(model, ids: torch.Tensor, context: int, batch_size: int, n_batches: int, device, amp_ctx):
    was_training = model.training
    model.eval()
    max_start = ids.numel() - context - 1
    starts = torch.linspace(0, max_start, steps=batch_size * n_batches).long()
    offsets = torch.arange(context)
    total, count = 0.0, 0
    for at in range(0, len(starts), batch_size):
        loc = starts[at:at + batch_size, None] + offsets[None, :]
        x, y = ids[loc].to(device), ids[loc + 1].to(device)
        with amp_ctx:
            _, loss = model(x, y)
        total += float(loss.item()) * x.shape[0]
        count += x.shape[0]
    model.train(was_training)
    return total / count


def atomic_save(checkpoint: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def train(args):
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after must be positive")
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.context_length < 2:
        raise ValueError("steps/batch-size/grad-accum must be positive and context-length >= 2")
    if args.n_layers < 1 or args.n_heads < 1 or args.embed_dim < 1 or args.embed_dim % args.n_heads:
        raise ValueError("embed-dim must be divisible by n-heads, and layers/heads must be positive")
    if (args.embed_dim // args.n_heads) % 2:
        raise ValueError("RoPE requires an even head dimension")
    if not (0 < args.lr and 0 <= args.min_lr <= args.lr):
        raise ValueError("require 0 <= min-lr <= lr with lr > 0")
    if args.eval_every < 1 or args.save_every < 1 or args.eval_batches < 1:
        raise ValueError("eval-every, save-every and eval-batches must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be nonnegative")

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    model_config_cls, model_cls = load_model_source(args.source)
    device = choose_device(args.device)
    print(f"Device: {device} | PyTorch: {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_text, val_text = get_texts(args)
    signature = dataset_signature(train_text, val_text)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(Path(args.resume), map_location="cpu", weights_only=True)
        if signature != resume_checkpoint.get("dataset_sha256"):
            raise ValueError("Dataset changed since checkpoint. Refusing unsafe resume.")
        tok = OwnTokenizer.load(out / "tokenizer.json")
        cfg = model_config_cls(**resume_checkpoint["config"])
        if tok.vocab_size != cfg.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match resumed checkpoint")
        for arg, old in (("batch_size", args.batch_size), ("grad_accum", args.grad_accum),
                         ("seed", args.seed), ("learning_rate", args.lr), ("min_lr", args.min_lr),
                         ("warmup_steps", args.warmup_steps)):
            if resume_checkpoint["settings"].get(arg) != old:
                raise ValueError(f"Resume option changed: {arg}; use original training settings")
        if args.steps != resume_checkpoint["settings"]["steps"]:
            raise ValueError("Resume requires original --steps value, preserving the learning-rate schedule")
    else:
        tok = OwnTokenizer()
        if args.tokenizer == "bpe":
            tok.train_bpe(train_text, args.vocab_size, args.tokenizer_sample_bytes)
        tok.save(out / "tokenizer.json")
        cfg = model_config_cls(vocab_size=tok.vocab_size, context_length=args.context_length,
                               n_layers=args.n_layers, n_heads=args.n_heads, embed_dim=args.embed_dim,
                               dropout=args.dropout)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    if train_ids.numel() <= cfg.context_length + 1 or val_ids.numel() <= cfg.context_length + 1:
        raise ValueError(f"Not enough tokens (train {train_ids.numel()}, val {val_ids.numel()}); "
                         f"need > context_length+1 ({cfg.context_length + 1}) in each. "
                         "Try --context-length 64 or supply more text.")
    print(f"Train tokens: {train_ids.numel():,} | Validation tokens: {val_ids.numel():,} | vocab: {tok.vocab_size}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = model_cls(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2 or "norm" in name or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                    {"params": no_decay, "weight_decay": 0.0}],
                                   lr=args.lr, betas=(0.9, 0.95))
    amp_ctx, scaler = make_precision(device, args.amp)
    sampling_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1729)
    start_step = 0
    best_val_loss = float("inf")
    settings = {"batch_size": args.batch_size, "grad_accum": args.grad_accum, "seed": args.seed,
                "learning_rate": args.lr, "min_lr": args.min_lr,
                "warmup_steps": args.warmup_steps, "steps": args.steps}
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scaler.is_enabled() and resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_step = resume_checkpoint["step"]
        best_val_loss = resume_checkpoint["best_val_loss"]
        sampling_rng.set_state(resume_checkpoint["sampling_rng"])
        torch.set_rng_state(resume_checkpoint["torch_rng"])
        if device.type == "cuda" and resume_checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng"])
        print(f"Resuming after optimizer step {start_step}", flush=True)
    target_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    if start_step >= target_step:
        print("Checkpoint already reached requested execution target.")
        return

    def checkpoint(step: int) -> dict:
        return {
            "format": "expansformer-from-scratch-v0.1",
            "step": step, "config": asdict(cfg), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_loss": best_val_loss, "dataset_sha256": signature, "settings": settings,
            "sampling_rng": sampling_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }

    metrics_path = out / "metrics.jsonl"
    model.train()
    elapsed_start = time.monotonic()
    for step in range(start_step, target_step):
        lr = lr_for_step(step, args.steps, args.warmup_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = sample_batch(train_ids, cfg.context_length, args.batch_size, sampling_rng, device)
            with amp_ctx:
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            weighted_loss += float(loss.detach().item()) / args.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        do_eval = completed % args.eval_every == 0 or completed == target_step
        val_loss = None
        if do_eval:
            val_loss = evaluate(model, val_ids, cfg.context_length, args.batch_size,
                                args.eval_batches, device, amp_ctx)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                atomic_save(checkpoint(completed), out / "best.pt")
        record = {"step": completed, "train_loss": round(weighted_loss, 6),
                  "val_loss": round(val_loss, 6) if val_loss is not None else None,
                  "lr": lr, "tokens_seen": completed * args.batch_size * args.grad_accum * cfg.context_length}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if completed == 1 or do_eval or completed % args.save_every == 0:
            message = f"step {completed}/{args.steps} | train_loss {weighted_loss:.4f} | lr {lr:.2e}"
            if val_loss is not None:
                message += f" | val_loss {val_loss:.4f} | best {best_val_loss:.4f}"
            print(message, flush=True)
        if completed % args.save_every == 0 or completed == target_step:
            atomic_save(checkpoint(completed), out / "latest.pt")
    print(f"Reached step {target_step}/{args.steps} in {time.monotonic() - elapsed_start:.1f}s. Checkpoints: {out/'best.pt'}, {out/'latest.pt'}", flush=True)


@torch.no_grad()
def generate(args):
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    model_config_cls, model_cls = load_model_source(args.source)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "expansformer-from-scratch-v0.1":
        raise ValueError("Unsupported checkpoint format")
    tok = OwnTokenizer.load(checkpoint_path.parent / "tokenizer.json")
    cfg = model_config_cls(**checkpoint["config"])
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    device = choose_device(args.device)
    model = model_cls(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = tok.encode(args.prompt)
    if not ids:
        ids = [SPECIAL["<BOS>"]]
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    output = model.generate(input_tensor, max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(tok.decode(output[0].tolist()))


def parser():
    p = argparse.ArgumentParser(description="Train Expansformer from scratch using original model.py")
    sub = p.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train", help="Train new random-initialized model or resume a checkpoint")
    train_cmd.add_argument("--data", required=True, help="Text file or directory of .txt/.md files")
    train_cmd.add_argument("--val-data", help="Optional separate held-out text file/directory")
    train_cmd.add_argument("--out", default="./expansformer_run")
    train_cmd.add_argument("--source", help="Original Expansformer ZIP or extracted directory")
    train_cmd.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    train_cmd.add_argument("--vocab-size", type=int, default=512)
    train_cmd.add_argument("--tokenizer-sample-bytes", type=int, default=131072)
    train_cmd.add_argument("--n-layers", type=int, default=4)
    train_cmd.add_argument("--n-heads", type=int, default=4)
    train_cmd.add_argument("--embed-dim", type=int, default=256)
    train_cmd.add_argument("--context-length", type=int, default=128)
    train_cmd.add_argument("--dropout", type=float, default=0.1)
    train_cmd.add_argument("--batch-size", type=int, default=8)
    train_cmd.add_argument("--grad-accum", type=int, default=2)
    train_cmd.add_argument("--steps", type=int, default=1000, help="Total optimizer updates, including resume")
    train_cmd.add_argument("--stop-after", type=int, help="Optional early checkpoint at a step below --steps, for controlled continuation")
    train_cmd.add_argument("--lr", type=float, default=3e-4)
    train_cmd.add_argument("--min-lr", type=float, default=1e-5)
    train_cmd.add_argument("--warmup-steps", type=int, default=30)
    train_cmd.add_argument("--weight-decay", type=float, default=0.1)
    train_cmd.add_argument("--grad-clip", type=float, default=1.0)
    train_cmd.add_argument("--eval-every", type=int, default=100)
    train_cmd.add_argument("--eval-batches", type=int, default=8)
    train_cmd.add_argument("--save-every", type=int, default=100)
    train_cmd.add_argument("--resume", help="Checkpoint latest.pt from the SAME --out and corpus")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_cmd.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train_cmd.add_argument("--cpu-threads", type=int, default=4)

    gen = sub.add_parser("generate", help="Generate from your trained weights (not Ollama)")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--prompt", default="Hello")
    gen.add_argument("--max-new-tokens", type=int, default=80)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=40)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--source")
    gen.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    gen.add_argument("--cpu-threads", type=int, default=4)

    smoke = sub.add_parser("smoke", help="Verify BPE roundtrip, forward/backprop, save, resume and generate")
    smoke.add_argument("--source")
    return p


def smoke(source):
    sample = "Expansformer learns from text. Causal attention predicts the next token.\n"
    tok = OwnTokenizer()
    tok.train_bpe(sample * 16, 288, 32768)
    for case in ["Hello hello!", "🚀 UTF-8 is supported", sample]:
        assert tok.decode(tok.encode(case)) == case, "Tokenizer roundtrip failed"
    print("PASS: BPE roundtrip, including Unicode", flush=True)
    with tempfile.TemporaryDirectory() as root:
        folder = Path(root)
        (folder / "training.txt").write_text(sample * 120, encoding="utf-8")
        p = parser()
        args = p.parse_args(["train", "--data", str(folder / "training.txt"),
                             "--out", str(folder / "run"), "--steps", "3", "--warmup-steps", "1",
                             "--n-layers", "2", "--n-heads", "4", "--embed-dim", "64",
                             "--context-length", "32", "--batch-size", "2", "--grad-accum", "2",
                             "--eval-every", "1", "--eval-batches", "1", "--save-every", "1",
                             "--device", "cpu", "--tokenizer", "bpe", "--vocab-size", "288"])
        args.source = source
        train(args)
        checkpoint = folder / "run" / "latest.pt"
        assert checkpoint.exists()
        ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert ckpt["step"] == 3
        rows = [json.loads(line) for line in (folder / "run" / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 3 and rows[0]["val_loss"] is not None
        g = p.parse_args(["generate", "--checkpoint", str(checkpoint), "--prompt", "Expansformer",
                          "--max-new-tokens", "8", "--device", "cpu"])
        g.source = source
        generate(g)
        # The completed-checkpoint resume verifies integrity without altering weights.
        args.resume = str(checkpoint)
        train(args)
        print("PASS: training, validation, checkpoint, reload, generation, safe resume", flush=True)


if __name__ == "__main__":
    opts = parser().parse_args()
    if opts.command == "train":
        train(opts)
    elif opts.command == "generate":
        generate(opts)
    else:
        smoke(opts.source)
