"""
================================================================================
TOKENIZER
Two options: CharTokenizer (instant, zero training) or BPETokenizer (real).
Train the BPE on your corpus for production use.
================================================================================
"""

import json
import re
from pathlib import Path
from collections import Counter


class CharTokenizer:
    """
    Character-level tokenizer — zero training required.
    Good for prototyping and small datasets. Bad for efficiency.
    """

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        # Reserve special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def train(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        for i, ch in enumerate(chars, len(self.special_tokens)):
            self.char_to_id[ch] = i
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        unk_id = self.char_to_id[self.unk_token]
        return [self.char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        special_ids = {self.char_to_id[t] for t in self.special_tokens}
        return "".join(self.id_to_char.get(i, "") for i in ids if i not in special_ids)

    def save(self, path: str):
        Path(path).write_text(json.dumps({"type": "char", "vocab": self.char_to_id}))

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.char_to_id = data["vocab"]
        tok.id_to_char = {int(v): k for k, v in tok.char_to_id.items()}
        return tok


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer — train on your corpus.
    No external dependencies. Suitable for real training runs.
    """

    def __init__(self, vocab_size: int = 32000):
        self.target_vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self._inverse_vocab: dict[bytes, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, verbose: bool = True):
        """Train BPE from raw text."""
        n_special = len(self.special_tokens)

        # Initialize with byte-level vocab
        tokens = list(text.encode("utf-8"))
        self.vocab = {i + n_special: bytes([i]) for i in range(256)}
        for tok, idx in self.special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")

        num_merges = self.target_vocab_size - 256 - n_special
        if num_merges <= 0:
            self._rebuild_inverse()
            return

        for i in range(num_merges):
            # Count adjacent pairs
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += 1

            if not pairs:
                break

            # Most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            new_id = 256 + n_special + i

            # Merge
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            # Record merge
            self.merges.append(best_pair)
            a_bytes = self.vocab.get(best_pair[0], b"")
            b_bytes = self.vocab.get(best_pair[1], b"")
            self.vocab[new_id] = a_bytes + b_bytes

            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i+1}/{num_merges} | vocab={len(self.vocab)} | tokens={len(tokens)}")

        self._rebuild_inverse()
        if verbose:
            print(f"BPE training complete: {len(self.vocab)} tokens, {len(self.merges)} merges")

    def _rebuild_inverse(self):
        self._inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        n_special = len(self.special_tokens)
        tokens = [b + n_special for b in text.encode("utf-8")]

        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_id = n_special + 256 + self.merges.index(pair)
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        special_ids = set(self.special_tokens.values())
        byte_chunks = []
        for idx in ids:
            if idx in special_ids:
                continue
            if idx in self.vocab:
                byte_chunks.append(self.vocab[idx])
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str):
        data = {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["target_vocab_size"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.special_tokens = data["special_tokens"]
        # Rebuild vocab from merges
        n_special = len(tok.special_tokens)
        tok.vocab = {i + n_special: bytes([i]) for i in range(256)}
        for t, idx in tok.special_tokens.items():
            tok.vocab[idx] = t.encode("utf-8")
        for i, (a, b) in enumerate(tok.merges):
            new_id = 256 + n_special + i
            tok.vocab[new_id] = tok.vocab.get(a, b"") + tok.vocab.get(b, b"")
        tok._rebuild_inverse()
        return tok
