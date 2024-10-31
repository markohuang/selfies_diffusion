import os
import torch
import multiprocessing
from datasets import Dataset
from torch.utils.data import DataLoader
from selfies_diffusion.custom_callbacks import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer
from .config import cfg

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
cfg.vocab_size = tokenizer.vocab_size

callbacks = [
    GenerateOnValidationCallback(cfg.output_file),
    EarlyStopping(monitor="val/loss", mode="min", patience=3),
    LearningRateMonitor(logging_interval='epoch'),
    ModelCheckpoint(monitor='val/loss'),
]

def my_collator(batch):
    features = {}
    inputs = tokenizer(
        [b['text'] for b in batch],
        max_length=cfg.max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
        padding='max_length', # keep commented for dynamic padding
        # padding=True,
    )
    features['input_ids'] = inputs.input_ids
    features['attention_mask'] = inputs.attention_mask
    return features


trainset, valset = Dataset.from_text(cfg.data_path).train_test_split(0.2).values()

num_workers = multiprocessing.cpu_count()
tloader = DataLoader(
    trainset,
    batch_size=cfg.batch_size,
    collate_fn=my_collator,
    num_workers=num_workers
)
vloader = DataLoader(
    valset,
    batch_size=cfg.batch_size,
    collate_fn=my_collator,
    num_workers=num_workers
)


# Export the tokenizer and dataset for other modules to use
__all__ = ['cfg', 'tokenizer', 'callbacks', 'tloader', 'vloader']
