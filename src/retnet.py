from typing import Optional, List, Tuple, Union

from torch import nn, Tensor

from .retention import MultiscaleRetention
from .base import ModelArgs
from .activations import get_activation_fn, DEFAULT_ACTIVATION

Activation = Union[str, nn.Module]

class RetNetDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: Optional[int] = None,
                 dropout: float = 0.0, activation: Activation = DEFAULT_ACTIVATION, 
                 layer_norm_eps: float = 1e-6, batch_first: bool = True, 
                 norm_first: bool = True, only_self_rttn: bool = True,
                 feed_forward: Optional[nn.Module] = None):
        super().__init__()

        dim_feedforward = dim_feedforward or (d_model * 4)
        self.norm_first = norm_first
        self.only_self_rttn = only_self_rttn

        self.norm = nn.ModuleList([
            nn.LayerNorm(d_model, eps=layer_norm_eps)
            for _ in range(2 if only_self_rttn else 3)
        ])
        self.dropout = nn.Dropout(dropout)

        self.self_retention = MultiscaleRetention(d_model, nhead, batch_first=batch_first)
        if not only_self_rttn:
            self.cross_retention = MultiscaleRetention(d_model, nhead, batch_first=batch_first)

        self.feed_forward = feed_forward or nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation_fn(activation),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, offset: int = 0, state: Optional[List] = None,
                need_state: bool = False) -> Union[Tensor, Tuple[Tensor, List]]:
        x = tgt

        tgt_state, mem_state = (state or [None] * 2) if not self.only_self_rttn else (state, None)
        y, tgt_state = self._block(self.norm[0](x), self.self_retention, mask=tgt_mask, offset=offset, state=tgt_state)
        x = x + y

        if not self.only_self_rttn:
            y, mem_state = self._block(self.norm[1](x), self.cross_retention, memory=memory, mask=memory_mask, offset=offset, state=mem_state)
            x = x + y

        x = x + self._ff_block(self.norm[-1](x))
        return (x, (tgt_state, mem_state) if not self.only_self_rttn else tgt_state) if need_state else x

    def _block(self, x: Tensor, retention: nn.Module, memory: Optional[Tensor] = None, mask: Optional[Tensor] = None, offset: int = 0, 
               state: Optional[List] = None) -> Tuple[Tensor, List]:
        x, state = retention(x, memory or x, memory or x, mask, offset=offset, state=state, _always_return_state=True)
        return self.dropout(x), state

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout(self.feed_forward(x))

class RetNetDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([
            RetNetDecoderLayer(args.dim, args.num_heads, args.dim_ffn, dropout=args.dropout, 
                               layer_norm_eps=args.norm_eps, only_self_rttn=True, activation=DEFAULT_ACTIVATION, 
                               **(args.layer_args or {})) for _ in range(args.num_layers)
        ])
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, offset: int = 0, state: Optional[List] = None,
                need_state: bool = False) -> Union[Tensor, Tuple[Tensor, List]]:
        output = tgt

        new_states = []
        for i, layer in enumerate(self.layers):
            output, layer_state = layer(output, memory, tgt_mask, memory_mask, offset=offset, state=state[i] if state else None, need_state=True)
            new_states.append(layer_state)

        output = self.norm(output)
        return (output, new_states) if need_state else output

class CausalRetNet(nn.Module):
   
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(params.vocab_size, params.dim)
        self.decoder = RetNetDecoder(params)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward_with_logits(self, x: Tensor, offset: int = 0, state: Optional[List] = None, 
                need_state: bool = False) -> Union[Tensor, Tuple[Tensor, List]]:
        embed = self.embedding(x)
        embed, new_state = self.decoder(embed, offset=offset, state=state, need_state=True)
        output = self.output(embed)
        return (output.float(), new_state) if need_state else output.float()

    def forward_with_embeddings(self, x: Tensor, offset: int = 0, state: Optional[List] = None, 
                need_state: bool = False) -> Union[Tensor, Tuple[Tensor, List]]:
        embed = self.embedding(x)
        embed, new_state = self.decoder(embed, offset=offset, state=state, need_state=True)
        return (embed.float(), new_state) if need_state else embed.float()

    def forward(self, x: Tensor, want_logits: bool = True, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, List]]:
        if want_logits:
            return self.forward_with_logits(x, *args, **kwargs)
        else:
            return self.forward_with_embeddings(x, *args, **kwargs)