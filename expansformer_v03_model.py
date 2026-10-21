"""Optional, trainable research extensions to the original Expansformer TransformerLM.

Defaults are mathematically and structurally the original model. The original
expansformer_model/model.py is deliberately unchanged. This file has NO pretrained
weights or dependency on external cognitive modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from model import ModelConfig, RMSNorm, SwiGLU, TransformerLM
except ModuleNotFoundError:
    from train_expansformer_v01 import load_model_source
    load_model_source(None)
    from model import ModelConfig, RMSNorm, SwiGLU, TransformerLM


@dataclass
class ResearchConfig:
    moe_experts: int = 0  # 0 => original dense SwiGLU
    moe_top_k: int = 2
    moe_balance_weight: float = 0.01
    auxiliary_exits: bool = False  # train shared-vocab intermediate prediction heads
    exit_loss_weight: float = 0.15

    def validate(self, layers: int):
        if self.moe_experts < 0:
            raise ValueError('moe_experts must be nonnegative')
        if self.moe_experts and not 1 <= self.moe_top_k <= self.moe_experts:
            raise ValueError('moe_top_k must lie between 1 and moe_experts')
        if not 0 <= self.moe_balance_weight <= 1:
            raise ValueError('moe_balance_weight must be in [0,1]')
        if not 0 <= self.exit_loss_weight <= 1:
            raise ValueError('exit_loss_weight must be in [0,1]')
        if self.auxiliary_exits and layers < 2:
            raise ValueError('auxiliary exits need at least 2 transformer layers')


class SparseExperts(nn.Module):
    """Top-k token routing. Experts are actual SwiGLU feedforward networks.

    Router loss is a differentiable squared deviation from uniform average
    probabilities. With k=1, token-to-expert selection is non-differentiable;
    balance loss still trains router probabilities. No throughput gain asserted.
    """

    def __init__(self, dim: int, ff_dim: int, count: int, top_k: int, bias: bool):
        super().__init__()
        self.top_k = top_k
        self.count = count
        self.router = nn.Linear(dim, count, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, ff_dim, bias=bias) for _ in range(count)])
        self.balance_loss = torch.tensor(0.0)
        self.last_routing_counts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        flat = x.reshape(-1, dim)
        router_logits = self.router(flat).float()
        full_probs = router_logits.softmax(dim=-1)
        picks = torch.topk(router_logits, self.top_k, dim=-1)
        weights = picks.values.softmax(dim=-1).to(x.dtype)
        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            rows, slot = torch.where(picks.indices == expert_idx)
            if rows.numel():
                contributions = expert(flat.index_select(0, rows)) * weights[rows, slot].unsqueeze(-1)
                output = output.index_add(0, rows, contributions)
        # 0 at uniform importance. Nonzero otherwise. Gradients reach router.
        importance = full_probs.mean(dim=0)
        self.balance_loss = self.count * (importance - (1.0 / self.count)).square().sum()
        self.last_routing_counts = torch.bincount(picks.indices.detach().flatten(), minlength=self.count)
        return output.reshape(batch, length, dim)


class ResearchTransformerLM(TransformerLM):
    """Training-enhanced Expansformer with optional MoE and auxiliary early exits.

    Standard forward still returns (logits, primary cross entropy). Auxiliary
    losses are only added by training_loss(); evaluation uses primary CE only.
    """

    def __init__(self, config: ModelConfig, research: ResearchConfig | None = None):
        self.research_config = research or ResearchConfig()
        self.research_config.validate(config.n_layers)
        super().__init__(config)
        if self.research_config.moe_experts:
            for block in self.blocks:
                prior = block.ffn
                ff_dim = prior.w1.out_features
                block.ffn = SparseExperts(config.embed_dim, ff_dim,
                                          self.research_config.moe_experts,
                                          self.research_config.moe_top_k, config.bias)
                # Original model initializer also scales residual projection w2.
                for expert in block.ffn.experts:
                    self._init_weights(expert.w1)
                    self._init_weights(expert.w3)
                    nn.init.normal_(expert.w2.weight, std=0.02 / (2 * config.n_layers) ** .5)
                self._init_weights(block.ffn.router)
        if self.research_config.auxiliary_exits:
            # Exit positions are all internal depths; tied decoder doesn't grow vocab head.
            self.exit_norms = nn.ModuleList(
                [RMSNorm(config.embed_dim, eps=config.norm_eps) for _ in range(config.n_layers - 1)]
            )
        else:
            self.exit_norms = nn.ModuleList()

    def _path(self, ids: torch.Tensor, targets: torch.Tensor | None,
              stop_layer: int | None = None, collect_exits: bool = False):
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
        exit_losses = []
        moe_losses = []
        depth = stop_layer or self.config.n_layers
        for idx, block in enumerate(self.blocks[:depth]):
            x = block(x, rope)
            if isinstance(block.ffn, SparseExperts) and targets is not None:
                moe_losses.append(block.ffn.balance_loss)
            if collect_exits and idx < self.config.n_layers - 1 and targets is not None:
                mid_logits = self.lm_head(self.exit_norms[idx](x))
                exit_losses.append(F.cross_entropy(mid_logits.reshape(-1, mid_logits.size(-1)),
                                                    targets.reshape(-1), ignore_index=-1))
        # An internal early exit uses its separately trained exit normalization.
        head_norm = self.exit_norms[depth-1] if depth < self.config.n_layers else self.norm
        logits = self.lm_head(head_norm(x))
        primary = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1) if targets is not None else None)
        return logits, primary, exit_losses, moe_losses

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if not self.research_config.moe_experts and not self.research_config.auxiliary_exits:
            return super().forward(input_ids, targets)
        logits, primary, _, _ = self._path(input_ids, targets)
        return logits, primary

    def training_loss(self, input_ids: torch.Tensor, targets: torch.Tensor):
        if not self.research_config.moe_experts and not self.research_config.auxiliary_exits:
            _, loss = super().forward(input_ids, targets)
            return loss, {'lm_loss': loss.detach(), 'exit_loss': loss.detach() * 0,
                          'balance_loss': loss.detach() * 0}
        _, lm, exits, moe = self._path(input_ids, targets, collect_exits=bool(self.exit_norms))
        exit_loss = torch.stack(exits).mean() if exits else lm * 0.0
        balance = torch.stack(moe).mean() if moe else lm * 0.0
        objective = lm + self.research_config.exit_loss_weight * exit_loss + self.research_config.moe_balance_weight * balance
        return objective, {'lm_loss': lm.detach(), 'exit_loss': exit_loss.detach(),
                           'balance_loss': balance.detach()}

    @torch.no_grad()
    def logits_at_depth(self, input_ids: torch.Tensor, depth: int):
        if depth != self.config.n_layers and not self.research_config.auxiliary_exits:
            raise ValueError('Intermediate exit was not trained; full depth required')
        if depth == self.config.n_layers:
            return self(input_ids)[0]
        return self._path(input_ids, targets=None, stop_layer=depth)[0]
