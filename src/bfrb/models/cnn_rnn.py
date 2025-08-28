from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bfrb.dataset import ORIENTATION_MAP
from bfrb.models._classifier_head import ClassifierHead


class TCNBlock(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=ch,
            out_channels=ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=ch,
            out_channels=ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        """
        x: [B, T, C]
        """
        # to [B, C, T]
        y = x.transpose(1, 2)
        y = self.conv1(y).transpose(1, 2)
        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)

        y = y.transpose(1, 2)
        y = self.conv2(y).transpose(1, 2)
        y = self.norm2(y)
        y = self.act(y)
        y = self.drop(y)

        out = x + y
        # Zero out padded positions in place
        out = out.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        return out


class StackedTCN(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        n_blocks: int,
        kernel_size: int,
        base_dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                TCNBlock(
                    ch=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=(base_dilation ** i),
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.handedness_embedding = nn.Embedding(2, hidden_dim)
        self.orientation_embedding = nn.Embedding(len(ORIENTATION_MAP), hidden_dim)

    def forward(self, x: Tensor, mask: Tensor, handedness: Optional[Tensor] = None, orientation: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, T, C_in]
        """
        h = self.proj(x)

        if handedness is not None:
            handedness = self.handedness_embedding(handedness)

        if handedness is not None:
            h = h + handedness.unsqueeze(1)

        if orientation is not None:
            orientation = self.orientation_embedding(orientation)

        if orientation is not None:
            h = h + orientation.unsqueeze(1)

        for blk in self.blocks:
            h = blk(h, pad_mask=mask)
        return h  # [B, T, hidden_dim]


class MaskedRNN(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
    def forward(self, x, mask: Tensor):
        B, T, _ = x.shape
        lengths_b = mask.sum(dim=1).to(torch.int64)  # [B]

        # pack -> GRU -> unpack (keeps only valid timesteps)
        packed = pack_padded_sequence(
            x, lengths=lengths_b.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)
        h_btH, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        # (optional) zero padded positions to be extra safe downstream
        h_btH = h_btH.masked_fill(~mask.unsqueeze(-1), 0.0)
        return h_btH, h_n

class CNNRNNModel(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int,
        d_model: int = 128,
        tcn_blocks: int = 3,
        tcn_kernel: int = 3,
        tcn_base_dilation: int = 2,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.tcn = StackedTCN(
            in_dim=num_channels,
            hidden_dim=d_model,
            n_blocks=tcn_blocks,
            kernel_size=tcn_kernel,
            base_dilation=tcn_base_dilation,
            dropout=dropout,
        )

        self.rnn = MaskedRNN(
            d_model=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out = rnn_hidden * (2 if bidirectional else 1)

        self.head = ClassifierHead(d_model=rnn_out, num_classes=num_classes)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        handedness: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        h_tcn = self.tcn(x, mask=key_padding_mask, handedness=handedness, orientation=orientation)                     # [B, T, d_model]
        h_rnn, _ = self.rnn(h_tcn, mask=key_padding_mask)              # [B, T, rnn_out]

        sequence_logits, cls_attn_weights = self.head(h_rnn, pad_mask=key_padding_mask)
        return sequence_logits, cls_attn_weights

    @property
    def num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params