"""
================================================================================
GENERATE
Load a trained checkpoint and generate text. Interactive or one-shot.
================================================================================
"""

import argparse
import json
from pathlib import Path

import torch

from model import ModelConfig, TransformerLM
from tokenizer import CharTokenizer, BPETokenizer


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TransformerLM, CharTokenizer | BPETokenizer]:
    """Load model + tokenizer from checkpoint directory."""
    ckpt_path = Path(checkpoint_path)

    if ckpt_path.is_dir():
        # Directory: look for best.pt or final.pt
        model_path = ckpt_path / "best.pt"
        if not model_path.exists():
            model_path = ckpt_path / "final.pt"
        tokenizer_path = ckpt_path / "tokenizer.json"
        config_path = ckpt_path / "config.json"
    else:
        # Direct .pt file
        model_path = ckpt_path
        tokenizer_path = ckpt_path.parent / "tokenizer.json"
        config_path = ckpt_path.parent / "config.json"

    # Load config
    if config_path.exists():
        config_dict = json.loads(config_path.read_text())
        config = ModelConfig(**config_dict)
    else:
        # Fall back to config stored in checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ModelConfig(**ckpt["config"])

    # Load tokenizer
    tok_data = json.loads(tokenizer_path.read_text())
    if tok_data["type"] == "bpe":
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        tokenizer = CharTokenizer.load(str(tokenizer_path))

    # Load model
    model = TransformerLM(config).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    print(f"Loaded model: {model.count_parameters()['total_M']} params | step {step} | val_loss {val_loss}")

    return model, tokenizer


def generate_text(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(new_tokens)


def main():
    p = argparse.ArgumentParser(description="Generate text from trained model")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint dir or .pt file")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt (if not set, enters interactive mode)")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    model, tokenizer = load_model(args.checkpoint, device)

    if args.prompt:
        # One-shot generation
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n--- Prompt ---\n{args.prompt}")
        print(f"\n--- Generated ---\n{generated}")
    else:
        # Interactive mode
        print("\nInteractive generation (type 'quit' to exit)")
        print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        print("-" * 50)

        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            generated = generate_text(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            print(f"\n{generated}")


if __name__ == "__main__":
    main()
