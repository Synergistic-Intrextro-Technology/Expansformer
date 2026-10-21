"""
================================================================================
DEMO — Quick Smoke Test
Trains a tiny model on sample text to verify the full pipeline.
Run this first to confirm everything works before training on real data.
================================================================================
"""

import torch
from model import ModelConfig, TransformerLM
from tokenizer import CharTokenizer, BPETokenizer
from dataset import TextDataset, create_dataloaders

# Sample corpus — replace with your own data
SAMPLE_TEXT = """
The transformer architecture revolutionized natural language processing.
Self-attention mechanisms allow the model to weigh the importance of different
parts of the input sequence. Unlike recurrent neural networks, transformers
process all positions simultaneously, enabling massive parallelization.
The key innovation is the scaled dot-product attention function, which computes
a weighted sum of values based on the compatibility of queries and keys.
Multi-head attention extends this by running multiple attention functions in
parallel, allowing the model to jointly attend to information from different
representation subspaces. Layer normalization and residual connections help
stabilize training of deep transformer networks. The feed-forward network
in each transformer block consists of two linear transformations with a
nonlinear activation in between, processing each position independently.
Position information is injected through positional encodings or rotary
position embeddings, giving the model awareness of token order.
""" * 50  # Repeat for enough training data

def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Tokenizer
    print("\n--- Tokenizer ---")
    tokenizer = CharTokenizer()
    tokenizer.train(SAMPLE_TEXT)
    print(f"Vocab size: {tokenizer.vocab_size}")

    token_ids = tokenizer.encode(SAMPLE_TEXT)
    print(f"Tokens: {len(token_ids):,}")

    # Verify roundtrip
    decoded = tokenizer.decode(token_ids)
    assert decoded == SAMPLE_TEXT, "Tokenizer roundtrip failed!"
    print("Roundtrip: OK")

    # 2. Model
    print("\n--- Model ---")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        n_layers=4,
        n_heads=4,
        embed_dim=128,
        dropout=0.0,
    )
    model = TransformerLM(config).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total_M']}")

    # 3. Training sanity check
    print("\n--- Training (20 steps) ---")
    train_loader, val_loader = create_dataloaders(
        token_ids, context_length=128, batch_size=8,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    losses = []
    for i, (x, y) in enumerate(train_loader):
        if i >= 20:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if (i + 1) % 5 == 0:
            print(f"  step {i+1:>3d} | loss {loss.item():.4f}")

    # Verify loss decreased
    first_5_avg = sum(losses[:5]) / 5
    last_5_avg = sum(losses[-5:]) / 5
    print(f"\nFirst 5 avg loss: {first_5_avg:.4f}")
    print(f"Last 5 avg loss:  {last_5_avg:.4f}")
    assert last_5_avg < first_5_avg, "Loss did not decrease — something is wrong"
    print("Loss decreasing: OK")

    # 4. Generation
    print("\n--- Generation ---")
    model.eval()
    prompt = "The transformer"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100, temperature=0.8)

    generated = tokenizer.decode(output[0, len(prompt):].tolist())
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated[:200]}'")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — pipeline is functional.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
