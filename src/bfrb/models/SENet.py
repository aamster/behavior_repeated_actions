import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from bfrb.config.config import SENetConfig


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)  # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)  # -> (B, C, 1)
        return x * se


class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # SE
        self.se = SEBlock(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)  # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # (B, out, L)
        out = out + identity
        return self.relu(out)

def residual_se_cnn_block(in_channels, out_channels, pool_size=2, drop=0.3):
    return nn.Sequential(
        ResNetSEBlock(in_channels, out_channels),
        nn.MaxPool1d(pool_size),
        nn.Dropout(drop)
    )


class CMIModel(nn.Module):
    def __init__(self, d_model: int, imu_dim, thm_dim, tof_dim, n_classes, config: SENetConfig, droput: float = 0.0):
        super().__init__()
        self.imu_branch = nn.Sequential(
            residual_se_cnn_block(
                in_channels=imu_dim,
                out_channels=config.imu_num_channels,
                drop=droput
            ),
            residual_se_cnn_block(
                in_channels=config.imu_num_channels,
                out_channels=d_model,
                drop=droput
            )
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bert = BertModel(BertConfig(
            hidden_size=d_model,
            num_hidden_layers=config.bert_num_layers,
            num_attention_heads=config.bert_num_attention_heads,
            intermediate_size=d_model * 4
        ))

        self.classifier = nn.Sequential(
            nn.Linear(d_model, config.classifier_num_channels[0], bias=False),
            nn.BatchNorm1d(config.classifier_num_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(droput),
            nn.Linear(config.classifier_num_channels[0], config.classifier_num_channels[1], bias=False),
            nn.BatchNorm1d(config.classifier_num_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(droput),
            nn.Linear(config.classifier_num_channels[1], n_classes)
        )

        self.handedness_embedding = nn.Embedding(2, d_model)

    def forward(self, imu, thm, tof, key_padding_mask: torch.Tensor, handedness: torch.Tensor):
        imu_feat = self.imu_branch(imu.permute((0, 2, 1)))

        imu_feat = imu_feat.permute((0, 2, 1))

        if handedness is not None:
            handedness = self.handedness_embedding(handedness)

        if handedness is not None:
            imu_feat = imu_feat + handedness.unsqueeze(1)

        bert_input = imu_feat

        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)  # (B,1,H)
        bert_input = torch.cat([cls_token, bert_input], dim=1)  # (B,T+1,H)


        outputs = self.bert(inputs_embeds=bert_input)
        pred_cls = outputs.last_hidden_state[:, 0, :]

        return self.classifier(pred_cls)

    @property
    def num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params