"""
File: /train_uhm.py
Created Date: Wednesday June 16th 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Wednesday June 16th 2021 3:35:57 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
"""
import argparse
import os
import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp

from pytorch_lightning.accelerators import accelerator

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from uhc.khrylib.utils import create_logger

from uhc.khrylib.utils import *
from torch.utils.tensorboard import SummaryWriter
from uhc.models.uhm_vae import VAEDynamV1
from uhc.data_loaders.dataset_amass_batch import DatasetAMASSBatch
from uhc.utils.config_utils.uhm_config import Config
from uhc.utils.lightning_utils import TextLogger
from uhc.utils.flags import flags


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id * 7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    cfg = Config(cfg_id=args.cfg, create_dirs=not (args.epoch > 0))
    cfg.update(args)
    seed_everything(cfg.seed, workers=False)
    flags.debug = args.debug

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using: {device}")
    loggers = []
    gpus = [i for i in range(torch.cuda.device_count())]

    if args.debug:
        cfg.no_log = True
        gpus = [0]

    if not cfg.no_log:
        wandb_logger = WandbLogger(
            name=args.cfg,
            project="uhm",
            resume=not args.resume is None,
            id=args.resume,
            notes=cfg.notes,
        )
        loggers.append(wandb_logger)

    """Setup Dataset"""
    train_dataset = DatasetAMASSBatch(cfg, data_mode="train")
    train_loader = train_dataset.sampling_loader(
        batch_size=cfg.data_specs.get("batch_szie", 256),
        num_samples=cfg.data_specs.get("num_samples", 5000),
        num_workers=8,
        fr_num=cfg.data_specs.get("t_total", 90),
    )
    val_dataset = DatasetAMASSBatch(cfg, data_mode="test")
    val_loader = val_dataset.iter_loader(
        batch_size=32, num_workers=8, fr_num=cfg.data_specs.get("t_total", 90)
    )
    data_sample = train_dataset.sample_seq()

    """Setup Model and logging"""
    model = VAEDynamV1(cfg=cfg, data_sample=data_sample)

    # logger
    text_logger = TextLogger(
        cfg=cfg,
        filename=os.path.join(cfg.log_dir, "log.txt"),
        file_handle=not cfg.no_log,
    )
    loggers.append(text_logger)

    checkpoint_epoch_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{cfg.model_dir}/checkpoints",
        filename="model-{epoch:04d}",
        save_last=True,
        save_top_k=-1,
        mode="min",
        every_n_val_epochs=1,
    )

    checkpoint_best_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{cfg.model_dir}/checkpoints",
        filename="model-best-{epoch:04d}",
        save_top_k=1,
        mode="min",
    )

    if args.epoch > 0:
        cp_name = "last" if args.epoch == -1 else f"model-epoch={args.epoch:04d}"
        resume_cp = f"{cfg.model_dir}/checkpoints/{cp_name}.ckpt"
    else:
        resume_cp = None

    """Train!"""
    trainer = pl.Trainer(
        logger=loggers,
        gpus=gpus,
        accelerator="ddp",
        resume_from_checkpoint=resume_cp,
        auto_select_gpus=True,
        max_epochs=cfg.num_epoch,
        callbacks=[checkpoint_epoch_cb, checkpoint_best_cb],
        check_val_every_n_epoch=cfg.save_n_epochs,
    )
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)
