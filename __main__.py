import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

## for cluster bash script
from importlib import import_module
module = import_module('experiments.' + 'chembl')
cfg, tokenizer, callbacks, tloader, vloader = module.cfg, module.tokenizer, module.callbacks, module.tloader, module.vloader

# from experiments.mof import cfg, tokenizer, callbacks, tloader, vloader
from selfies_diffusion.model import DiffusionTransformer

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium') # as per warning
    logger = WandbLogger(
        project="selfies_diffusion",
        name=cfg.name,
        log_model=True,
    )
    if hasattr(cfg, 'load_model'):
        # check if path is a directory
        model_path = cfg.load_model
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, os.listdir(model_path)[0])
        model = DiffusionTransformer.load_from_checkpoint(model_path)
    else:
        model = DiffusionTransformer(cfg, tokenizer)
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator="auto",
        max_epochs=cfg.num_epochs,
        logger=logger,
        # log_every_n_steps=1,
        val_check_interval=5000,
        enable_progress_bar=False,
        # devices=1,
    )
    trainer.fit(model, tloader, vloader)
