#!/usr/bin/env python3
"""Estimate model weight/optimizer floor and inspect local PyTorch/CUDA availability.
Not a GPU memory guarantee; activations, attention buffers, CUDA kernels omitted.
"""
from __future__ import annotations
import argparse
import json
import math
import torch


def estimate(layers, heads, dim, vocab):
    if dim % heads or (dim // heads) % 2:
        raise ValueError('embed dimension must be divisible by heads, even head dimension')
    ff = math.ceil((dim * 4 * 2 / 3) / 64) * 64
    # ModelConfig uses bias=False and tied output weights.
    total = vocab * dim + layers * (4 * dim * dim + 3 * dim * ff + 2 * dim) + dim
    return {'layers': layers, 'heads': heads, 'embedding_dim': dim, 'vocab': vocab,
            'parameters': total, 'weight_gib_fp32': total * 4 / 2**30,
            'training_states_floor_gib': total * 16 / 2**30,
            'warning': 'Gradient/optimizer state floor only. Excludes activations, attention matrices, model compilation and CUDA memory overhead.'}


def report():
    available = torch.cuda.is_available()
    gpu = []
    if available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu.append({'device': i, 'name': props.name, 'vram_gib': round(props.total_memory / 2**30, 2)})
    return {'pytorch': torch.__version__, 'cuda_available': available, 'gpus': gpu,
            'micro': estimate(4, 4, 256, 512),
            'small': estimate(6, 8, 512, 8192),
            'base': estimate(12, 12, 768, 16384)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--layers', type=int)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--vocab', type=int, default=512)
    a=p.parse_args()
    d=report()
    if a.layers:
        d['custom']=estimate(a.layers, a.heads, a.dim, a.vocab)
    print(json.dumps(d, indent=2))


if __name__=='__main__':
    main()
