import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, ContextManager

import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from loguru import logger
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from tqdm import tqdm

from bfrb.dataset import BFRB_BEHAVIORS, PAD_TOKEN_ID, ACTION_ID_MAP
from bfrb.models.transformer import EncoderTransformer

logger.remove()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logger.add(sys.stderr, level=LOG_LEVEL)


@dataclass
class LearningRateDecayConfig:
    lr_decay_iters: int  # should be ~= max_iters per Chinchilla
    learning_rate: float = 5e-4
    warmup_iters: int = 2000
    min_lr: float = 5e-5  # should be ~= learning_rate/10 per Chinchilla

def _compute_f1(labels: torch.Tensor, preds: torch.Tensor) -> tuple[float, float, float]:
    binary_labels = torch.tensor([x in BFRB_BEHAVIORS.values() for x in labels])
    binary_preds = torch.tensor([x in BFRB_BEHAVIORS.values() for x in preds])
    binary_f1 = BinaryF1Score()(binary_preds, binary_labels)

    macro_labels = torch.tensor([x-3 if x in BFRB_BEHAVIORS.values() else len(BFRB_BEHAVIORS) for x in labels], dtype=torch.long)
    macro_preds = torch.tensor([x-3 if x in BFRB_BEHAVIORS.values() else len(BFRB_BEHAVIORS) for x in preds], dtype=torch.long)
    macro_f1 = MulticlassF1Score(num_classes=len(BFRB_BEHAVIORS)+1, average='macro')(macro_preds, macro_labels)

    return binary_f1, macro_f1, (binary_f1 + macro_f1) / 2

@torch.no_grad()
def estimate_performance_metrics(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: EncoderTransformer,
    epoch: int,
    eval_iters: int = 200,
    label_smoothing: float = 0.0,
):
    out = {"train": {}, "val": {}}
    model.eval()

    train_sampler = RandomSampler(train_loader.dataset) # type: ignore
    val_sampler = SequentialSampler(val_loader.dataset) # type: ignore

    train_data_loader = DataLoader(
        dataset=train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=train_sampler,
        collate_fn=train_loader.collate_fn,
    )
    val_data_loader = DataLoader(
        dataset=val_loader.dataset,
        batch_size=val_loader.batch_size,
        sampler=val_sampler,
        collate_fn=val_loader.collate_fn,
    )

    eval_iters = min(eval_iters, len(train_data_loader), len(val_data_loader))

    for data_loader_name in ("train", "val"):
        if data_loader_name == "train":
            data_loader = train_data_loader
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
        else:
            data_loader = val_data_loader
            if isinstance(val_sampler, DistributedSampler):
                val_sampler.set_epoch(epoch)

        data_loader_iter = iter(data_loader)

        local_losses = torch.zeros(eval_iters, device=os.environ["DEVICE"])

        iterator = tqdm(
            range(eval_iters),
            desc=f"Evaluate performance on {data_loader_name} set",
            leave=False,
        )

        preds = []
        labels = []
        for eval_iter in iterator:
            (
                input_tensor,
                _,
                sequence_labels
            ) = next(data_loader_iter)

            if torch.cuda.is_available():
                input_tensor = input_tensor.to(torch.device(os.environ["DEVICE"]))
                sequence_labels = sequence_labels.to(torch.device(os.environ["DEVICE"]))

            logits, predsi = inference(
                model=model,
                input_tensor=input_tensor,
                pad_token_id=PAD_TOKEN_ID,
            )

            C = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, C),
                sequence_labels.view(-1),
                label_smoothing=label_smoothing,
            )
            local_losses[eval_iter] = loss
            preds += predsi
            labels += sequence_labels

        binary_f1, macro_f1, f1 = _compute_f1(labels=torch.stack(labels), preds=torch.tensor(preds, dtype=torch.long))

        cm = wandb.plot.confusion_matrix(
            y_true=labels, preds=preds, class_names=list(ACTION_ID_MAP.keys())
        )

        avg_loss = local_losses.mean()
        out[data_loader_name] = {"loss": avg_loss, "f1": f1, "binary_f1": binary_f1, "macro_f1": macro_f1, "confusion_matrix": cm}

    model.train()
    return out


def train_epoch(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    model: EncoderTransformer,
    optimizer,
    epoch: int,
    model_weights_out_dir: Path,
    best_loss: float,
    decay_learning_rate: bool = True,
    learning_rate_decay_config: Optional[LearningRateDecayConfig] = None,
    loss_eval_interval: int = 2000,
    accuracy_eval_interval: int = 10000,
    eval_iters: int = 200,
    autocast_context: ContextManager = nullcontext(),
    label_smoothing: float = 0.0,
):
    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.get_autocast_gpu_dtype() == torch.float16
    )

    total_loss = 0
    prog_bar = tqdm(
        train_data_loader, total=len(train_data_loader), desc=f"train epoch {epoch}"
    )
    for epoch_iter, data in enumerate(prog_bar):
        (
            input_tensor,
            target_tensor,
            sequence_labels
        ) = data
        input_tensor: torch.Tensor
        target_tensor: torch.Tensor
        sequence_labels: torch.Tensor

        global_iter_num = (epoch - 1) * len(train_data_loader) + epoch_iter

        if torch.cuda.is_available():
            input_tensor = input_tensor.to(
                torch.device(os.environ["DEVICE"]), non_blocking=True
            )
            target_tensor = target_tensor.to(
                torch.device(os.environ["DEVICE"]), non_blocking=True
            )
            sequence_labels = sequence_labels.to(
                torch.device(os.environ["DEVICE"]), non_blocking=True
            )

        if decay_learning_rate:
            assert learning_rate_decay_config is not None
            lr = _get_lr(
                iteration=global_iter_num,
                warmup_iters=learning_rate_decay_config.warmup_iters,
                learning_rate=learning_rate_decay_config.learning_rate,
                lr_decay_iters=learning_rate_decay_config.lr_decay_iters,
                min_lr=learning_rate_decay_config.min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]

        if global_iter_num % loss_eval_interval == 0:
            logger.info("Calculating performance metrics")
            with autocast_context:
                metrics = estimate_performance_metrics(
                    train_loader=train_data_loader,
                    val_loader=val_data_loader,
                    model=model,
                    eval_iters=eval_iters,
                    epoch=epoch,
                    label_smoothing=label_smoothing,
                )

            if global_iter_num % accuracy_eval_interval == 0:
                logger.info(
                    f"step {global_iter_num}: train loss {metrics['train']['loss']:.4f}, "
                    f"val loss {metrics['val']['loss']:.4f}, "
                    f"train f1 {metrics['train']['f1']:.4f}, "
                    f"val f1 {metrics['val']['f1']:.4f}"
                )
            else:
                logger.info(
                    f"step {global_iter_num}: train loss {metrics['train']['loss']:.4f}, "
                    f"val loss {metrics['val']['loss']:.4f}"
                )
            if os.environ.get("USE_WANDB") == "True":
                wandb.log(
                    {
                        "iter": global_iter_num,
                        "lr": lr,
                        "train_cross_entropy_loss": metrics["train"]["loss"],
                        "val_cross_entropy_loss": metrics["val"]["loss"],
                        "train_f1": metrics["train"]["f1"],
                        "val_f1": metrics["val"]["f1"],
                        "train_binary_f1": metrics["train"]["binary_f1"],
                        "val_binary_f1": metrics["val"]["binary_f1"],
                        "train_macro_f1": metrics["train"]["macro_f1"],
                        "val_macro_f1": metrics["val"]["macro_f1"],
                        "train_confusion_matrix": metrics["train"]["confusion_matrix"],
                        "val_confusion_matrix": metrics["val"]["confusion_matrix"],
                    }
                )

            if global_iter_num % accuracy_eval_interval == 0:
                if metrics["val"]["loss"] < best_loss:
                    best_loss = metrics["val"]["loss"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": global_iter_num,
                        "best_loss": best_loss,
                    }
                    torch.save(checkpoint, Path(model_weights_out_dir) / "ckpt.pt")

        with autocast_context:
            key_padding_mask = (
                input_tensor != PAD_TOKEN_ID
            ).bool()[:, :, 0].squeeze(-1)
            logits = model(
                x=input_tensor, key_padding_mask=key_padding_mask
            )

            C = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, C),
                sequence_labels.view(-1),
                label_smoothing=label_smoothing,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()


    return total_loss / len(train_data_loader), best_loss


@torch.no_grad()
def inference(
    model: EncoderTransformer,
    input_tensor: torch.Tensor,
    pad_token_id: int,
):
    key_padding_mask = (input_tensor != pad_token_id).bool()[:, :, 0].squeeze(-1)
    logits = model(
        x=input_tensor, key_padding_mask=key_padding_mask
    )

    probs = F.softmax(logits, dim=-1)
    _, topi = probs.topk(1)
    preds = topi.squeeze()


    return logits, preds


@torch.no_grad()
def evaluate(
    model: EncoderTransformer,
    data_loader: DataLoader,
):
    raise NotImplementedError()


def train(
    train_dataloader,
    val_dataloader,
    model: EncoderTransformer,
    optimizer,
    n_epochs,
    model_weights_out_dir: str,
    learning_rate=0.001,
    decay_learning_rate: bool = True,
    loss_eval_interval: int = 2000,
    accuracy_eval_interval: int = 10000,
    eval_iters: int = 200,
    label_smoothing: float = 0.0,
    autocast_context: ContextManager = nullcontext(),
):
    os.makedirs(model_weights_out_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch=epoch)

        _, best_loss = train_epoch(
            train_data_loader=train_dataloader,
            val_data_loader=val_dataloader,
            model=model,
            optimizer=optimizer,
            label_smoothing=label_smoothing,
            epoch=epoch,
            decay_learning_rate=decay_learning_rate,
            learning_rate_decay_config=LearningRateDecayConfig(
                learning_rate=learning_rate,
                lr_decay_iters=len(train_dataloader) * n_epochs,
                min_lr=learning_rate / 10,
            ),
            best_loss=best_loss,
            model_weights_out_dir=Path(model_weights_out_dir),
            eval_iters=eval_iters,
            loss_eval_interval=loss_eval_interval,
            accuracy_eval_interval=accuracy_eval_interval,
            autocast_context=autocast_context,
        )


# https://github.com/karpathy/nanoGPT/blob/master/train.py
def _get_lr(
    iteration: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr
):
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)