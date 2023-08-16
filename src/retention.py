from typing import Optional, List, Tuple, Union

import torch
from torch import nn, Tensor

class XPos(nn.Module):
    def __init__(self, dim: int, theta: int = 10000, scale_base: int = 512):
        super().__init__()
        self.dim, self.scale_base, self.theta = dim, scale_base, theta
        self.register_buffer('scale', (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim))

    def forward(self, x: Tensor, offset: int = 0, inverse_scale: bool = False) -> Tensor:
        l, d = x.size(-2), x.size(-1)
        assert d <= self.dim
        scale = self.scale ** (torch.arange(offset, offset + l) / self.scale_base)[:, None]
        freqs = 1. / (self.theta ** (torch.arange(scale.size(-1)) / scale.size(-1)))
        sin_in = torch.einsum('i, j -> i j', torch.arange(offset, offset + scale.size(0)), freqs)
        sin, cos = torch.sin(sin_in), torch.cos(sin_in)
        scale, sin, cos = scale[-l:], sin[-l:], cos[-l:]
        if inverse_scale: scale = 1 / scale
        y = x * XPos.duplicate_interleave(cos * scale) + XPos.rotate_half(x) * XPos.duplicate_interleave(sin * scale)
        return y

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        return torch.stack((-x[:, :, 1::2], x[:, :, ::2]), dim=-1).flatten(-2)

    @staticmethod
    def duplicate_interleave(x: Tensor) -> Tensor:
        return x.view(-1, 1).repeat(1, 2).view(x.size(0), -1)

class Retention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: Optional[int] = None, gamma: float = 1.0, 
                 kdim: Optional[int] = None, vdim: Optional[int] = None, head_vdim: Optional[int] = None, 
                 layer_norm: bool = False, add_bias_kv: bool = False):
        super().__init__()
        self.embed_dim, self.head_dim, self.kdim, self.vdim, self.head_vdim, self.gamma = embed_dim, head_dim or embed_dim, kdim or embed_dim, vdim or embed_dim, head_vdim or (vdim or embed_dim), gamma
        self.query, self.key, self.value = nn.Linear(embed_dim, self.head_dim, bias=False), nn.Linear(self.kdim, self.head_dim, bias=add_bias_kv), nn.Linear(self.vdim, self.head_vdim, bias=add_bias_kv)
        self.xpos = XPos(self.head_dim)
        self.layer_norm = nn.LayerNorm() if layer_norm else nn.Identity()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, 
                offset: int = 0, state: Optional[Tensor] = None, need_state: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        l1 = q.size(-2) == 1
        q, k, v = self.query(q), self.key(k), self.value(v)
        q, k = self.xpos(q, offset), self.xpos(k, offset, True)
        if not mask and not (need_state and l1): mask = Retention.decay_mask(q.size(-2), self.gamma)
        if need_state:
            if state is None: state = self.empty_state.repeat(q.size(-3), 1, 1)
            if l1 and not mask:
                state = self.gamma * state + (k.transpose(-1, -2) @ v)
                o = q @ state
            else:
                ir = (q @ k.transpose(-1, -2)) * mask.unsqueeze(0) @ v
                power = (self.gamma ** torch.arange(1, q.size(-2) + 1)).view(1, q.size(-2), 1).repeat(q.size(-3), 1, 1)
                cr = (q @ state) * power
                state = (self.gamma ** q.size(-2)) * state + (k.transpose(-1, -2) @ (v * mask[-1].view(1, -1, 1)))
                o = ir + cr
        else:
            o = (q @ k.transpose(-1, -2)) * mask.unsqueeze(0) @ v
        return (self.layer_norm(o), state) if need_state else self.layer_norm(o)

    @property
    def empty_state(self) -> Tensor:
        return torch.zeros(1, self.head_dim, self.head_vdim)

    @staticmethod
    def decay_mask(l: int, gamma: float) -> Tensor:
        return torch.nan_to_num((gamma ** (torch.arange(l).view(-1, 1) - torch.arange(l).view(1, -1))) * (torch.arange(l).view(-1, 1) >= torch.arange(l).view(1, -1)))

class MultiscaleRetention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, gammas: Optional[List[float]] = None, 
                 kdim: Optional[int] = None, vdim: Optional[int] = None, batch_first: bool = True):
        super().__init__()
        self.embed_dim, self.kdim, self.vdim, self.num_heads, self.head_dim = embed_dim, kdim or embed_dim, vdim or embed_dim, num_heads, embed_dim // num_heads
        self.gammas = gammas or (1 - 2 ** (-5. - torch.arange(0, num_heads))).tolist()
        self.heads = nn.ModuleList([Retention(embed_dim, self.head_dim, gamma, kdim=self.kdim, vdim=self.vdim, head_vdim=self.vdim // num_heads) for gamma in self.gammas])
        self.group_norm, self.group, self.output = nn.GroupNorm(num_heads, self.vdim), nn.Linear(embed_dim, self.vdim, bias=False), nn.Linear(self.vdim, embed_dim, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, 
                offset: int = 0, state: Optional[List[Tensor]] = None, need_state: bool = False, 
                _always_return_state: bool = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        if need_state or state:
            if not state: state = []
            o, state = zip(*[head(q, k, v, mask, offset=offset, state=state[i] if i < len(state) else None, need_state=True) for i, head in enumerate(self.heads)])
        else: o = [head(q, k, v, mask, offset=offset) for head in self.heads]
        o = torch.cat(o, dim=-1)
        o = self.group_norm(o.view(-1, self.vdim)).view(o.size())
        return (self.output(torch.sigmoid(self.group(q)) * o), list(state) if state else []) if need_state or _always_return_state else self.output(torch.sigmoid(self.group(q)) * o)
