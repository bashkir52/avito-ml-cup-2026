"""Exercise full-attention optimization on the configured GPU before using dataset resources."""
import argparse
import json
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F


def gpu_device(backend):
    if backend == 'directml':
        import torch_directml

        if not torch_directml.is_available():
            raise RuntimeError('DirectML does not detect an adapter')
        return torch_directml.device(), torch_directml.device_name(0)
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access a CUDA GPU')
    return torch.device('cuda'), torch.cuda.get_device_name()


class AttentionBlock(nn.Module):
    """Dense attention with explicit GEMMs and a key mask, including on DirectML."""

    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('attention width must be divisible by heads')
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
                                nn.Linear(2 * width, width))

    def forward(self, x, valid):
        batch, tokens, width = x.shape
        size = width // self.heads
        q, k, v = self.qkv(self.norm(x)).reshape(batch, tokens, 3, self.heads, size).unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        logits = (q @ k.transpose(-2, -1)) * size ** -.5
        attention = logits.masked_fill(~valid[:, None, None, :], -1e4).softmax(dim=-1)
        context = (attention @ v).transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.out(context)
        return x + self.ff(x)


def measure(device, batch, candidates, steps):
    torch.manual_seed(2026)
    layers = nn.ModuleList(AttentionBlock(64) for _ in range(2)).to(device)
    head = nn.Linear(64, 1).to(device)
    parameters = [*layers.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    inputs = torch.randn(batch, candidates + 13, 64).to(device)
    valid = torch.ones(inputs.shape[:2], dtype=torch.bool)
    valid[:, -1] = False
    valid = valid.to(device)
    initial = head.weight.detach().cpu().clone()
    start = perf_counter()
    durations, losses = [], []
    for i in range(steps + 2):
        before = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x = inputs
        for layer in layers:
            x = layer(x, valid)
        score = head(x[:, 13:]).squeeze(-1).masked_fill(~valid[:, 13:], -1e4)
        log_prob = F.log_softmax(score, dim=-1)
        loss = -log_prob[:, 1].mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        # A parameter read synchronizes the optimizer update as well as the forward pass.
        updated = head.weight.detach().cpu()
        if i >= 2:
            durations.append(perf_counter() - before)
    if not torch.isfinite(updated).all() or torch.equal(initial, updated) or not losses[-1] < losses[0]:
        raise RuntimeError('attention training did not make a finite useful parameter update')
    return {'device': str(device), 'batch': batch, 'candidates': candidates, 'steps_timed': steps,
            'seconds_per_step': sum(durations) / steps,
            'feeds_per_second': steps * batch / sum(durations), 'total_seconds': perf_counter() - start,
            'loss_first': losses[0], 'loss_last': losses[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('directml', 'cuda'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    device, name = gpu_device(args.backend)
    print(json.dumps({'torch': torch.__version__, 'adapter': name, 'device': str(device)}), flush=True)
    for candidates in (96, 300):
        print(json.dumps(measure(device, 32, candidates, 8)), flush=True)


if __name__ == '__main__':
    main()
