from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from bfrb.models.transformer import ActivationFunction, PositionalEncodingType


class Config(BaseModel):
    batch_size: int = 64
    data_dir: Path
    n_epochs: Optional[int] = None
    weights_out_dir: Optional[Path] = None
    limit: Optional[int] = None
    learning_rate: float = 1e-3
    seed: Optional[int] = None
    load_from_checkpoint_path: Optional[Path] = None
    evaluate_only: bool = False
    git_commit: Optional[str] = None
    dropout: float = 0.0
    weight_decay: float = 0.0
    compile: bool = False
    decay_learning_rate: bool = True
    loss_eval_interval: int = 2000
    accuracy_eval_interval: int = 10000
    eval_iters: int = 200
    eval_out_path: Optional[Path] = None
    is_test: bool = False
    use_ddp: bool = False
    num_layers: int = 3
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None
    label_smoothing: float = 0.1
    use_mixed_precision: bool = True
    dtype: str = "float16"

    class Config:
        extra = "forbid"


class TransformerConfig(Config):
    d_model: int = 256
    n_head: int = 4
    feedforward_hidden_dim: int = 1024
    activation: ActivationFunction = ActivationFunction.GELU
    positional_encoding_type: PositionalEncodingType = PositionalEncodingType.LEARNED
    window_length: int = 64