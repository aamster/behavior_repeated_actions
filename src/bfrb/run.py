import json
import os
import random
from contextlib import nullcontext
from enum import Enum
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

from bfrb.collate_function import CollateFunction
from bfrb.config.config import TransformerConfig, SENetConfig
from bfrb.dataset import BFRBDataset, ACTION_ID_MAP, PAD_TOKEN_ID, BFRB_BEHAVIORS, \
    NON_BFRB_BEHAVIORS
from bfrb.models.SENet import CMIModel
from bfrb.models.cnn_rnn import CNNRNNModel
from bfrb.models.transformer import EncoderTransformer, ProjectionType
from bfrb.train_evaluate import train, evaluate
from bfrb.utils.model_loading import fix_model_state_dict


class ModelType(Enum):
    SENet = 0

@click.command()
@click.option(
    '--config-path',
    type=click.Path(exists=True, readable=True),
    required=True
)
@click.option(
    '--model-type',
    type=click.Choice(choices=[x for x in ModelType]),
    required=True
)
def main(config_path: Path, model_type: ModelType):
    with open(config_path) as f:
        config = json.load(f)

    if model_type == ModelType.SENet:
        config = SENetConfig.model_validate(config)
    else:
        raise ValueError('unsupported')

    if config.seed is not None:
        torch.random.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

    os.environ["USE_WANDB"] = str(config.use_wandb)
    if config.use_wandb:
        if config.wandb_api_key is None:
            raise ValueError("Must provide wandb_api_key")

        wandb.login(key=config.wandb_api_key)

        wandb.init(
            project="bfrb_kaggle",
            config={
                k: v
                for k, v in config.model_dump().items()
                if k
                not in (
                    "data_path",
                    "model_weights_out_dir",
                    "model_weights_path",
                    "evaluate_only",
                )
            },
        )
        wandb.config.update({"git_commit": config.git_commit})

    train_dataset = BFRBDataset(
        data_path=config.data_dir / 'train.parquet',
        meta_path=config.data_dir / 'train.json',
        is_train=True,
        window_length=config.window_length,
        features=config.features,
    )
    val_dataset = BFRBDataset(
        data_path=config.data_dir / 'val.parquet',
        meta_path=config.data_dir / 'val.json',
        is_train=False,
        window_length=config.window_length,
        features=config.features,
    )

    logger.info(f'Train N: {len(train_dataset)}')
    logger.info(f'Val N: {len(val_dataset)}')

    train_data_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=CollateFunction(
            fixed_length=config.window_length,
            pad_token_id=PAD_TOKEN_ID,
            include_handedness=config.features is None or "handedness" in config.features,
            include_orientation=config.features is None or "orientation" in config.features
        ),
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=CollateFunction(
            fixed_length=config.window_length,
            pad_token_id=PAD_TOKEN_ID,
            include_handedness=config.features is None or "handedness" in config.features,
            include_orientation=config.features is None or "orientation" in config.features
        ),
        pin_memory=True,
    )

    test_data_loader = None # TODO implement

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["DEVICE"] = device

    device = torch.device(device)

    if model_type == ModelType.SENet:
        model = CMIModel(
            d_model=config.d_model,
            imu_dim=train_dataset.num_channels,
            thm_dim=None,
            tof_dim=None,
            n_classes=len(BFRB_BEHAVIORS)+len(NON_BFRB_BEHAVIORS),
            config=config
        ).to(device)
    else:
        raise ValueError('unsupported')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.load_from_checkpoint_path is not None:
        checkpoint = torch.load(
            config.load_from_checkpoint_path, map_location=device
        )
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            model.load_state_dict(fix_model_state_dict(checkpoint["model"]))

        optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info(f"Model num params: {model.num_params / 1e6}M")

    if config.compile:
        logger.info("compiling model")
        model = torch.compile(model) # type: ignore

    if device.type == "cuda" and config.use_mixed_precision:
        if config.dtype == "bfloat16":
            model_dtype = torch.bfloat16
        elif config.dtype == "float16":
            model_dtype = torch.float16
        else:
            raise ValueError(f'{config.dtype} not supported')
        ctx = torch.amp.autocast(device.type, dtype=model_dtype) # type: ignore
    else:
        ctx = nullcontext()
    logger.info(f"using ctx {ctx}")

    if config.evaluate_only:
        with ctx:
            val_sequence_ids, val_gestures, val_f1 = (
                evaluate(
                    model=model,
                    data_loader=(
                        test_data_loader if config.is_test else val_data_loader
                    ), # type: ignore
                )
            ) # type: ignore
        logger.info(f"F1: {val_f1}")
        df = pd.DataFrame(
            {
                "sequence_id": val_sequence_ids,
                "gesture": val_gestures,
            }
        )
        df.to_csv(config.eval_out_path,
            index=False,
        )
    else:
        train(
            train_dataloader=train_data_loader,
            val_dataloader=val_data_loader,
            model=model,
            optimizer=optimizer,
            model_weights_out_dir=str(config.weights_out_dir),
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            decay_learning_rate=config.decay_learning_rate,
            loss_eval_interval=config.loss_eval_interval,
            accuracy_eval_interval=config.accuracy_eval_interval,
            eval_iters=config.eval_iters,
            label_smoothing=config.label_smoothing,
            autocast_context=ctx,
        )

if __name__ == '__main__':
    main()
