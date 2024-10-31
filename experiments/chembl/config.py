import os
from pathlib import Path
from ml_collections import ConfigDict

root = Path(__file__).parent.parent.parent.absolute()

cfg = ConfigDict()
cfg.name            = "chembl_experiments"
cfg.log_dir         = str(root / "chembl_logs")
cfg.output_file     = str(root / "chembl_logs" / "generated_selfies.txt")
cfg.data_path       = str(root / "chembl_selfies_subset.txt")
cfg.tokenizer_path  = str(root / "chembl_tokenizer")
os.makedirs(cfg.log_dir, exist_ok=True)

cfg.num_epochs      = 300
cfg.batch_size      = 16
cfg.max_length      = 107
cfg.vocab_size      = -1

cfg.hyperparams = hyperparams = ConfigDict()
hyperparams.warmup_steps    = 10000
hyperparams.h_size          = 48
hyperparams.hh_size         = 384
hyperparams.n_layers        = 12
hyperparams.lr              = 5e-6
# diffusion
hyperparams.time_channels   = 128
hyperparams.timesteps       = 1000
hyperparams.noise_schedule  = "sqrt"
hyperparams.dropout         = 0.1


cfg.bert_cfg = bert_cfg = ConfigDict()
bert_cfg.num_attention_heads            = 12
bert_cfg.max_position_embeddings        = cfg.get_ref('max_length')
bert_cfg.hidden_size                    = hyperparams.get_ref('h_size')
bert_cfg.num_hidden_layers              = hyperparams.get_ref('n_layers')
bert_cfg.position_embedding_type        = "relative_key"
bert_cfg.hidden_dropout_prob            = 0.1
bert_cfg.attention_probs_dropout_prob   = 0.1
bert_cfg.vocab_size                     = cfg.get_ref('vocab_size')
bert_cfg.use_cache                      =   False

