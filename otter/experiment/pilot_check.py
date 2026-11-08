import logging
import math
import os

import torch
from otter.experiment import train

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run this inside the Slurm GPU allocation.")

logging.basicConfig(level=logging.INFO)


def report(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is nonfinite: {value}")
    print(f"CHECK {name}: {value}", flush=True)
    return value


original_clip = torch.nn.utils.clip_grad_norm_


def checked_clip(*args, **kwargs):
    norm = original_clip(*args, **kwargs)
    if report("unscaled_gradient_norm", norm) <= 0:
        raise RuntimeError("Gradient norm is zero.")
    return norm


torch.nn.utils.clip_grad_norm_ = checked_clip

original_loss_log = train.log_loss_info_to_wandb


def checked_loss_log(loss_info, prefix):
    report(f"{prefix}/fair_crps", loss_info["fair_crps"])
    return original_loss_log(loss_info=loss_info, prefix=prefix)


train.log_loss_info_to_wandb = checked_loss_log

original_log = train.WandbLogger.log


def checked_log(self, values):
    for name, value in values.items():
        if name in {
            "val/fair_crps",
            "val/ensemble_mean_rmse",
            "val/ensemble_spread",
        }:
            result = report(name, value)
            if name == "val/ensemble_spread" and result <= 0:
                raise RuntimeError("Validation ensemble has zero spread.")
    return original_log(self, values)


train.WandbLogger.log = checked_log

torch.cuda.reset_peak_memory_stats()
train.main()
print(
    "Peak allocated GPU GiB:",
    torch.cuda.max_memory_allocated() / 2**30,
    flush=True,
)
print(
    "Peak reserved GPU GiB:",
    torch.cuda.max_memory_reserved() / 2**30,
    flush=True,
)
