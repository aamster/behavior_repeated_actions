import os
from enum import Enum
from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F


class ActivationFunction(Enum):
    GELU = "gelu"
    RELU = "relu"

class PositionalEncodingType(Enum):
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"

class ProjectionType(Enum):
    LINEAR = "linear"
    CNN = 'CNN'

class MLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        activation_function: ActivationFunction = ActivationFunction.GELU,
    ):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._activation_function = (
            F.relu if activation_function == ActivationFunction.RELU else F.gelu
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = self._activation_function(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self._d_model = d_model
        self.n_head = n_head
        # output projection
        self.output_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    @property
    def d_qkv(self) -> int:
        return int(self._d_model / self.n_head)

    def forward(self, **kwargs):
        raise NotImplementedError

    def _calc_attention(
        self,
        q,
        k,
        v,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        q = q.view(B, T_q, self.n_head, self.d_qkv).transpose(1, 2)
        k = k.view(B, T_k, self.n_head, self.d_qkv).transpose(1, 2)
        v = v.view(B, T_k, self.n_head, self.d_qkv).transpose(1, 2)

        if return_attention_weights:
            import math
            import torch.nn.functional as F

            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            y = attn_weights @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
            )
            attn_weights = None

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T_q, self._d_model)

        y = self.proj_dropout(self.output_proj(y))

        if return_attention_weights:
            return y, attn_weights
        else:
            return y

    @staticmethod
    def _create_attn_mask(
        key_padding_mask: Optional[torch.Tensor], T_q: int, T_k: int, is_causal: bool
    ):
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_k)
        else:
            attn_mask = torch.ones(
                (1, 1, 1, T_k), device=os.environ["DEVICE"], dtype=torch.bool
            )

        if is_causal:
            causal_mask = torch.tril(
                torch.ones(T_q, T_k, device=attn_mask.device, dtype=torch.bool)
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T_q, T_k)
            attn_mask = causal_mask & attn_mask

        return attn_mask


class MultiHeadSelfAttention(_MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout,
        )
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward( # type: ignore
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        # calculates q, k, v in a single operation rather than in 3 separate operations for
        # efficiency but is equivalent
        q, k, v = self.qkv_proj(x).split(self._d_model, dim=2)

        attn_mask = self._create_attn_mask(
            key_padding_mask=key_padding_mask,
            T_k=k.shape[1],
            T_q=q.shape[1],
            is_causal=is_causal,
        )

        y = self._calc_attention(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            return_attention_weights=return_attention_weights,
        )

        return y

class _EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_attention_heads: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        mlp_activation: ActivationFunction = ActivationFunction.GELU,
    ):
        super().__init__()
        self.layer_norm = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])
        self.multi_head_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_head=n_attention_heads,
            dropout=dropout,
        )
        self.mlp = MLP(
            d_model=d_model,
            dropout=dropout,
            hidden_dim=feedforward_hidden_dim,
            activation_function=mlp_activation,
        )

    def forward(
        self, x, key_padding_mask: torch.Tensor, return_attention_weights: bool = False
    ):
        x = self.layer_norm[0](x)
        attn_out = self.multi_head_attention(
            x,
            key_padding_mask=key_padding_mask,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            attn_out, attn_weights = attn_out
        else:
            attn_weights = None
        x = x + attn_out
        x = x + self.mlp(self.layer_norm[1](x))

        if return_attention_weights:
            return x, attn_weights
        else:
            return x


class TopKPoolHead(torch.nn.Module):
    def __init__(self, d_model: int, num_classes: int, k: int):
        super().__init__()
        self._cls_head = torch.nn.Linear(d_model, num_classes)  # per-frame logits
        self.gesture_score_head = torch.nn.Linear(d_model, 1)          # gestureness g_t
        self._k = k

    def forward(self, H):  # H: [B, T, d_model]
        logits_t = self._cls_head(H)                 # [B, T, C]
        scores   = self.gesture_score_head(H).squeeze(-1)   # [B, T]
        k = min(self._k, H.size(1))
        # top-K along time
        idx = scores.topk(k, dim=1).indices         # [B, K]
        # gather the top-K logits
        b = torch.arange(H.size(0)).unsqueeze(-1).to(H.device)
        topk_logits = logits_t[b, idx, :]           # [B, K, C]
        pooled = topk_logits.mean(dim=1)            # [B, C]
        return pooled, logits_t, scores

class EncoderTransformer(nn.Module):
    def __init__(
        self,
        n_attention_heads: int,
        n_layers: int,
        d_model: int,
        block_size: int,
        num_channels: int,
        n_classes: int,
        dropout: float = 0.0,
        feedforward_hidden_dim: int = 2048,
        num_top_gesture_idx: int = 15,
        mlp_activation: ActivationFunction = ActivationFunction.GELU,
        projection_type: ProjectionType = ProjectionType.LINEAR
    ):
        super().__init__()
        self._block_size = block_size
        self._d_model = d_model
        self._dropout = dropout
        self._n_attention_heads = n_attention_heads
        self._n_layers = n_layers
        self.positional_embedding = nn.Embedding(self._block_size, self._d_model)
        self.layer_norm = LayerNorm(self._d_model)
        self.head = TopKPoolHead(d_model=d_model, num_classes=n_classes, k=num_top_gesture_idx)
        self.blocks = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model=d_model,
                    n_attention_heads=n_attention_heads,
                    dropout=dropout,
                    feedforward_hidden_dim=feedforward_hidden_dim,
                    mlp_activation=mlp_activation,
                )
                for _ in range(n_layers)
            ]
        )

        if projection_type == ProjectionType.LINEAR:
            projection = nn.Linear(in_features=num_channels, out_features=d_model)
        elif projection_type == ProjectionType.CNN:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        self._projection = projection

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        _, t, _ = x.size()
        x = self._projection(x)
        
        x = x + self.positional_embedding(torch.arange(0, t, dtype=torch.long, device=x.device))

        attention_weights = []

        for block in self.blocks:
            x = block(
                x,
                key_padding_mask=key_padding_mask,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                x, attn_weights = x
                attention_weights.append(attn_weights)
        if return_attention_weights:
            return attention_weights

        x = self.layer_norm(x)
        pooled_gesture_logits, _, _ = self.head(x)
        return pooled_gesture_logits

    @property
    def num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.positional_embedding is not None:
                n_params -= self.positional_embedding.weight.numel()
        return n_params