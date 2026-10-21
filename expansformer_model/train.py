"""
================================================================================
TRAINING LOOP
Full training pipeline: AMP, gradient accumulation, cosine schedule,
checkpointing, validation. Optimized for dual-GPU setups.
================================================================================
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from model import ModelConfig, TransformerLM
from tokenizer import CharTokenizer, BPETokenizer
from dataset import load_text_files, create_dataloaders


# ==============================================================================
# TRAINING CONFIG
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train your transformer model")

    # Data
    p.add_argument("--data", type=str, required=True,
                   help="Path to text file or directory of .txt files")
    p.add_argument("--tokenizer", type=str, default="char",
                   choices=["char", "bpe"], help="Tokenizer type")
    p.add_argument("--vocab-size", type=int, default=4096,
                   help="BPE vocabulary size (ignored for char tokenizer)")

    # Model architecture
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Mixed-precision training (default: on)")

    # Checkpointing
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=500,
                   help="Save checkpoint every N steps")
    p.add_argument("--eval-every", type=int, default=100,
                   help="Evaluate every N steps")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, cuda, cpu")

    return p.parse_args()


# ==============================================================================
# LR SCHEDULE
# ==============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ==============================================================================
# TRAINING
# ==============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device, amp_ctx) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= 50:  # cap eval batches for speed
            break
    model.train()
    return total_loss / max(n_batches, 1)


def train():
    args = parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # DATA + TOKENIZER
    # ========================================================================
    print("\n--- Loading data ---")
    data_path = Path(args.data)
    if data_path.is_file():
        raw_text = load_text_files([data_path])
    elif data_path.is_dir():
        raw_text = load_text_files([data_path])
    else:
        print(f"ERROR: {args.data} not found")
        sys.exit(1)

    print(f"Raw text: {len(raw_text):,} characters")

    if args.tokenizer == "bpe":
        print(f"\nTraining BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(raw_text)
    else:
        print("\nBuilding character tokenizer...")
        tokenizer = CharTokenizer()
        tokenizer.train(raw_text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save(str(out_dir / "tokenizer.json"))

    token_ids = tokenizer.encode(raw_text)
    print(f"Total tokens: {len(token_ids):,}")

    train_loader, val_loader = create_dataloaders(
        token_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
    )

    # ========================================================================
    # MODEL
    # ========================================================================
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )

    model = TransformerLM(config).to(device)
    params = model.count_parameters()
    print(f"\nModel: {params['total_M']} parameters")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Attention: {params['attention']:,}")
    print(f"  FFN:       {params['ffn']:,}")

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # ========================================================================
    # OPTIMIZER + AMP
    # ========================================================================
    # Separate weight decay groups — no decay on biases, norms, embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"\nAMP: {use_amp} ({amp_dtype if use_amp else 'N/A'})")

    # ========================================================================
    # RESUME
    # ========================================================================
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed at step {start_step}, best val loss {best_val_loss:.4f}")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    max_steps = args.epochs * len(train_loader)
    print(f"\nTraining for {args.epochs} epochs ({max_steps:,} steps)")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print("=" * 60)

    model.train()
    step = start_step
    running_loss = 0.0
    tokens_processed = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Learning rate schedule
            lr = get_lr(step, args.warmup_steps, max_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward with gradient accumulation
            is_accum_step = (batch_idx + 1) % args.grad_accum_steps != 0

            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if not is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                running_loss += loss.item() * args.grad_accum_steps
                tokens_processed += x.numel() * args.grad_accum_steps

                # Logging
                if step % 10 == 0:
                    elapsed = time.time() - t_start
                    avg_loss = running_loss / 10
                    tok_per_sec = tokens_processed / elapsed
                    print(
                        f"step {step:>6d} | epoch {epoch+1}/{args.epochs} | "
                        f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                        f"{tok_per_sec:,.0f} tok/s"
                    )
                    running_loss = 0.0
                    tokens_processed = 0
                    t_start = time.time()

                # Validation
                if step % args.eval_every == 0 and len(val_loader) > 0:
                    val_loss = evaluate(model, val_loader, device, amp_ctx)
                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                    marker = " *BEST*" if improved else ""
                    print(f"  >> val loss: {val_loss:.4f}{marker}")

                    if improved:
                        save_checkpoint(model, optimizer, scaler, config, step, best_val_loss,
                                       out_dir / "best.pt")

                # Periodic checkpoint
                if step % args.save_every == 0:
                    save_checkpoint(model, optimizer, scaler, config, step, best_val_loss,
                                   out_dir / f"step_{step}.pt")

    # Final checkpoint
    save_checkpoint(model, optimizer, scaler, config, step, best_val_loss,
                   out_dir / "final.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}/")


def save_checkpoint(model, optimizer, scaler, config, step, best_val_loss, path):
    """Save full training state."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "config": config.__dict__,
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  >> checkpoint saved: {path}")


if __name__ == "__main__":
    train()
