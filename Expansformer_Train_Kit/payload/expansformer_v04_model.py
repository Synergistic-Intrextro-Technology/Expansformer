"""Opt-in trainable neural additions inspired by original user modules.

No hidden external calls, pretrained weights, online self-training, or correctness claims.
The v0.3/base implementation remains identical with both new features disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from expansformer_v03_model import ResearchConfig, ResearchTransformerLM, SparseExperts
from train_expansformer_v01 import load_model_source

try:
    from model import ModelConfig
except ModuleNotFoundError:
    load_model_source(None)
    from model import ModelConfig


@dataclass
class NeuralConfig(ResearchConfig):
    memory_slots: int = 0
    memory_every: int = 2
    next_token_heads: bool = False
    next_token_head_weight: float = 0.1

    def validate(self, layers: int):
        super().validate(layers)
        if self.memory_slots < 0 or self.memory_slots > 1024:
            raise ValueError('memory_slots must be 0..1024')
        if self.memory_every < 1 or (self.memory_slots and self.memory_every > layers):
            raise ValueError('memory_every must be >= 1 and <= layers when slots are enabled')
        if not 0 <= self.next_token_head_weight <= 1:
            raise ValueError('next_token_head_weight must be in [0,1]')
        if self.next_token_heads and self.next_token_head_weight <= 0:
            raise ValueError('next_token_head_weight must be positive when heads are enabled')
        if self.next_token_heads and not self.auxiliary_exits:
            raise ValueError('next-token prediction heads require --auxiliary-exits')
        if self.next_token_heads and layers < 2:
            raise ValueError('next-token prediction heads require >=2 transformer layers')


class TrainableSlotMemory(nn.Module):
    """Read-only *parametric* latent slots; NOT a fact store or episodic memory.

    Each position attends to the same learned slots, not to future sequence tokens.
    All slots update by gradient descent only when the training optimizer steps.
    This block cannot import assertions or grant factual provenance.
    """
    def __init__(self, dim: int, slots: int, layers: int, every: int):
        super().__init__()
        self.every = every
        self.keys = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.values = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.query = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        # Small but nonzero gates allow all memory weights to train immediately.
        self.gates = nn.Parameter(torch.full((layers,), 0.01))
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        if (layer + 1) % self.every:
            return hidden
        q = self.query(hidden)
        weights = torch.softmax((q.float() @ self.keys.float().T) / math.sqrt(hidden.size(-1)), dim=-1)
        read = weights.to(hidden.dtype) @ self.values.to(hidden.dtype)
        # Gated residual allows network to suppress unhelpful memory.
        return hidden + self.gates[layer].tanh().to(hidden.dtype) * self.output(read)


class NeuralTransformerLM(ResearchTransformerLM):
    """From-scratch Expansformer; v0.3 dense/MoE/exit flags retained.

    Auxiliary head estimates accuracy of early next-token argmax on the training
    distribution. It does not indicate truth, evidence, certainty, or calibration.
    """
    def __init__(self, config: ModelConfig, research: NeuralConfig | None = None):
        options = research or NeuralConfig()
        options.validate(config.n_layers)
        super().__init__(config, options)
        self.research_config = options
        self.slot_memory = (TrainableSlotMemory(config.embed_dim, options.memory_slots,
                                               config.n_layers, options.memory_every)
                            if options.memory_slots else None)
        self.next_token_heads = (nn.ModuleList(nn.Linear(config.embed_dim, 1)
                                               for _ in range(config.n_layers - 1))
                                 if options.next_token_heads else nn.ModuleList())
        self._last_next_token_head_loss: torch.Tensor | None = None

    def _path(self, ids, targets=None, stop_layer=None, collect_exits=False):
        if self.slot_memory is None and not self.next_token_heads:
            return super()._path(ids, targets, stop_layer, collect_exits)
        batch, length = ids.shape
        if length > self.config.context_length:
            raise ValueError('Sequence longer than context length')
        if stop_layer is not None and not 1 <= stop_layer <= self.config.n_layers:
            raise ValueError('Invalid inference depth')
        x = self.token_embed(ids)
        if not self.config.rope:
            x = x + self.pos_embed(torch.arange(length, device=ids.device))
        x = self.drop(x)
        rope = self.rope_freqs if self.config.rope else None
        exit_losses, moe_losses, predictive_losses = [], [], []
        depth = stop_layer or self.config.n_layers
        for idx, block in enumerate(self.blocks[:depth]):
            x = block(x, rope)
            if self.slot_memory is not None:
                x = self.slot_memory(x, idx)
            if isinstance(block.ffn, SparseExperts) and targets is not None:
                moe_losses.append(block.ffn.balance_loss)
            if idx < self.config.n_layers - 1 and targets is not None and collect_exits:
                mid_logits = self.lm_head(self.exit_norms[idx](x))
                exit_losses.append(F.cross_entropy(mid_logits.reshape(-1, mid_logits.size(-1)),
                                                   targets.reshape(-1), ignore_index=-1))
                if self.next_token_heads:
                    valid = targets != -1
                    # Teacher signal = whether early argmax equals actual NEXT TOKEN,
                    # not whether the generated claim is true or supported.
                    prediction_is_correct = (mid_logits.detach().argmax(dim=-1) == targets).float()
                    confidence_logit = self.next_token_heads[idx](x.detach()).squeeze(-1)
                    if valid.any():
                        predictive_losses.append(F.binary_cross_entropy_with_logits(
                            confidence_logit[valid].float(), prediction_is_correct[valid]))
        norm = self.exit_norms[depth-1] if depth < self.config.n_layers else self.norm
        logits = self.lm_head(norm(x))
        primary = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1) if targets is not None else None)
        self._last_next_token_head_loss = (torch.stack(predictive_losses).mean()
                                            if predictive_losses else None)
        return logits, primary, exit_losses, moe_losses

    def forward(self, input_ids, targets=None):
        if self.slot_memory is None and not self.next_token_heads:
            return super().forward(input_ids, targets)
        logits, loss, _, _ = self._path(input_ids, targets)
        return logits, loss

    def training_loss(self, input_ids, targets):
        if self.slot_memory is None and not self.next_token_heads:
            return super().training_loss(input_ids, targets)
        _, lm, exits, moe = self._path(input_ids, targets, collect_exits=bool(self.exit_norms))
        exit_loss = torch.stack(exits).mean() if exits else lm * 0.0
        balance = torch.stack(moe).mean() if moe else lm * 0.0
        predict = (self._last_next_token_head_loss if self._last_next_token_head_loss is not None
                   else lm * 0.0)
        loss = (lm + self.research_config.exit_loss_weight * exit_loss +
                self.research_config.moe_balance_weight * balance +
                self.research_config.next_token_head_weight * predict)
        return loss, {'lm_loss': lm.detach(), 'exit_loss': exit_loss.detach(),
                      'balance_loss': balance.detach(), 'next_token_head_loss': predict.detach()}

    @torch.no_grad()
    def adaptive_logits(self, input_ids: torch.Tensor, threshold: float = 0.95):
        """Select a depth by *UNVERIFIED* head predictions; research only.

        Checks the latest next-token prediction at each early depth. It may
        choose an incorrect shallow answer. No speed or accuracy guarantee.
        """
        if not self.next_token_heads:
            raise ValueError('No trained next-token heads; pass --auxiliary-exits --next-token-heads')
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be (0,1]')
        if input_ids.size(0) != 1:
            raise ValueError('adaptive inference currently accepts batch size 1')
        if input_ids.size(1) > self.config.context_length:
            raise ValueError('Sequence longer than context length')
        x = self.token_embed(input_ids)
        if not self.config.rope:
            x = x + self.pos_embed(torch.arange(input_ids.size(1), device=input_ids.device))
        x = self.drop(x)
        rope = self.rope_freqs if self.config.rope else None
        for idx, block in enumerate(self.blocks):
            x = block(x, rope)
            if self.slot_memory is not None:
                x = self.slot_memory(x, idx)
            if idx < self.config.n_layers - 1:
                confidence = torch.sigmoid(self.next_token_heads[idx](x[:, -1]).float()).item()
                if confidence >= threshold:
                    logits = self.lm_head(self.exit_norms[idx](x))
                    return logits, idx + 1, confidence
        return self.lm_head(self.norm(x)), self.config.n_layers, None
