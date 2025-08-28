from torch import nn
from torch.nn import functional as F


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)  # scores over time
        self.cls = nn.Linear(d_model, num_classes)  # sequence-level classifier

    def forward(self, H, pad_mask=None):
        # H: [B,T,D]
        # pad_mask: [B,T]=True for valid tokens
        a = self.attn(H).squeeze(-1)  # [B,T]
        if pad_mask is not None:
            a = a.masked_fill(~pad_mask, float('-inf'))
        w = F.softmax(a, dim=1)  # [B,T]
        z = (w.unsqueeze(-1) * H).sum(dim=1)  # [B,D]
        return self.cls(z), w  # logits: [B,C], attn weights: [B,T]
