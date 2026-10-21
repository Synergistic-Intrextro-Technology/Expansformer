# experiment_skeleton.py
import random, csv, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Synthetic data ----------
def make_task_data(seed, n_classes=3, n_per=500, dim=32, shift=0.0):
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for c in range(n_classes):
        mu = rng.randn(dim) + (c * 3.0) + shift
        X.append(mu + 0.5 * rng.randn(n_per, dim))
        y += [c] * n_per
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return torch.tensor(X[perm]).to(device), torch.tensor(y[perm]).to(device)

# ---------- Tiny MoE model ----------
class TinyExpert(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x): return self.net(x)

class TinyMoE(nn.Module):
    def __init__(self, dim, n_experts=4, k=2, noise_std=0.1):
        super().__init__()
        self.experts = nn.ModuleList([TinyExpert(dim) for _ in range(n_experts)])
        self.router = nn.Linear(dim, n_experts)
        self.classifier = nn.Linear(128, 3)  # assume 3 classes
        self.n_experts = n_experts
        self.k = k
        self.noise_std = noise_std

    def noisy_top_k_gating(self, x):
        logits = self.router(x)
        noisy = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(noisy, dim=-1)
        topk = probs.topk(self.k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, topk.indices, 1.0)
        return mask, probs

    def forward(self, x):
        mask, probs = self.noisy_top_k_gating(x)
        # aggregate expert outputs (mean of selected experts)
        expert_outs = []
        for i, e in enumerate(self.experts):
            out = e(x)  # [B, H]
            expert_outs.append(out.unsqueeze(-1))
        stacked = torch.cat(expert_outs, dim=-1)  # [B, H, n_experts]
        mask = mask.unsqueeze(1)  # [B,1,n_experts]
        routed = (stacked * mask).sum(-1) / (mask.sum(-1).clamp_min(1.0))  # [B,H]
        logits = self.classifier(routed)
        return logits, probs

# ---------- Prioritized replay buffer ----------
class PrioritizedReplay:
    def __init__(self, capacity=5000, alpha=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, x, y, loss):
        p = (loss + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.cpu(), y.cpu()))
            self.priorities.append(p)
        else:
            self.buffer[self.pos] = (x.cpu(), y.cpu())
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
        probs = np.array(self.priorities) / sum(self.priorities)
        idx = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        xs, ys = zip(*[self.buffer[i] for i in idx])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

# ---------- Emotion engine ----------
class EmotionEngine:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, clip=3.0, ema_decay=0.9):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.clip = clip
        self.ema_decay = ema_decay
        self.loss_history = deque(maxlen=100)
        self.ema_E = 0.0

    def update_loss(self, loss_val):
        self.loss_history.append(loss_val)

    def compute(self, swarm_probs, world_model_error=0.0):
        L = np.array(self.loss_history) if len(self.loss_history) > 1 else np.array([0.0])
        delta_L = float(L.var()) if L.size > 1 else 0.0
        p = swarm_probs.detach().cpu()
        H_S = - (p * (p + 1e-8).log()).sum(dim=-1).mean().item()
        N_W = float(world_model_error)
        E = self.alpha * delta_L + self.beta * H_S + self.gamma * N_W
        E = max(0.0, min(self.clip, E))
        # EMA smoothing
        self.ema_E = self.ema_decay * self.ema_E + (1 - self.ema_decay) * E
        return float(self.ema_E)

# ---------- Adaptive optimizer wrapper ----------
class AdaptiveOptimizer:
    def __init__(self, params, base_lr=1e-3, kappa=0.5, min_mult=0.1):
        self.base_lr = base_lr
        self.kappa = kappa
        self.min_mult = min_mult
        self.opt = torch.optim.Adam(params, lr=base_lr)

    def step(self, emotion_state):
        mult = max(self.min_mult, 1.0 + self.kappa * emotion_state)
        for g in self.opt.param_groups:
            g['lr'] = self.base_lr * mult
        self.opt.step()

# ---------- Training / evaluation helpers ----------
def evaluate(model, X, y, batch=128):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            yb = y[i:i+batch]
            logits, _ = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total

# ---------- Main experiment ----------
def run_experiment(seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dim = 32
    X_A, y_A = make_task_data(seed=1, dim=dim, shift=0.0)
    X_B, y_B = make_task_data(seed=2, dim=dim, shift=5.0)  # shifted distribution

    model = TinyMoE(dim, n_experts=4, k=2, noise_std=0.1).to(device)
    emo = EmotionEngine()
    replay = PrioritizedReplay(capacity=2000, alpha=1.0)
    opt = AdaptiveOptimizer(model.parameters(), base_lr=1e-3, kappa=0.7)

    # Phase 1: pretrain on Task A
    epochs_A = 5
    for ep in range(epochs_A):
        model.train()
        idx = torch.randperm(len(X_A))
        for i in range(0, len(X_A), 64):
            xb = X_A[idx[i:i+64]]
            yb = y_A[idx[i:i+64]]
            logits, probs = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.opt.zero_grad()
            loss.backward()
            opt.opt.step()
        accA = evaluate(model, X_A, y_A)
        print(f"Pretrain A epoch {ep} accA={accA:.4f}")

    # Save baseline performance on A
    baseline_A = evaluate(model, X_A, y_A)

    # Phase 2: train on Task B with replay + emotion-driven LR
    epochs_B = 10
    for ep in range(epochs_B):
        model.train()
        idx = torch.randperm(len(X_B))
        for i in range(0, len(X_B), 64):
            xb = X_B[idx[i:i+64]]
            yb = y_B[idx[i:i+64]]
            logits, probs = model(xb)
            loss = F.cross_entropy(logits, yb)
            # add failures to replay
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask_fail = (preds != yb)
                if mask_fail.any():
                    for j in torch.where(mask_fail)[0]:
                        replay.add(xb[j], yb[j], float(F.cross_entropy(logits[j].unsqueeze(0), yb[j].unsqueeze(0)).item()))
            # compute emotion state
            emo.update_loss(loss.item())
            E = emo.compute(probs, world_model_error=0.0)
            # step with adaptive lr
            opt.opt.zero_grad()
            loss.backward()
            opt.step(E)
            # occasional replay
            if random.random() < 0.2:
                sample = replay.sample(32)
                if sample is not None:
                    xr, yr = sample
                    logits_r, probs_r = model(xr)
                    loss_r = F.cross_entropy(logits_r, yr)
                    opt.opt.zero_grad()
                    loss_r.backward()
                    opt.opt.step()
        accA = evaluate(model, X_A, y_A)
        accB = evaluate(model, X_B, y_B)
        retention = accA / baseline_A if baseline_A > 0 else accA
        print(f"Epoch {ep} accA={accA:.4f} accB={accB:.4f} retention={retention:.4f} E={E:.4f}")

    # Final metrics
    final_A = evaluate(model, X_A, y_A)
    final_B = evaluate(model, X_B, y_B)
    print("FINAL", final_A, final_B)
    return {"baseline_A": baseline_A, "final_A": final_A, "final_B": final_B}

if __name__ == "__main__":
    run_experiment()