from transformers import get_cosine_schedule_with_warmup
import time
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import *
from typing import *

from .utils import *

# model imports
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
)


class PretrainedEmbedding(nn.Module):
    def __init__(
        self,
        embed_mat: Literal["vocab", "embed"],
        use_normalization: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        self.use_normalization = use_normalization
        _, embed_dim = embed_mat.shape
        self.scale = math.sqrt(embed_dim)  # √D
        self.embed = nn.Embedding.from_pretrained(
            embed_mat.detach().clone(), freeze=freeze
        )

    def forward(
        self, x: Literal["batch", "vocab"]
    ) -> Literal["batch", "embed"]:
        embeds = self.embed(x)
        if self.use_normalization:
            embeds = F.normalize(embeds, p=2, dim=-1) * self.scale
        return embeds


class PretrainedUnembedding(nn.Module):
    def __init__(
        self,
        embed_mat: Literal["vocab", "embed"],
        use_renormalization: bool = True,
    ):
        super().__init__()
        self.renormalize = use_renormalization
        vocab_size, embed_dim = embed_mat.shape
        # LM head style scoring
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.unembed.weight.copy_(embed_mat.detach().clone())

    def forward(
        self, x: Literal["batch", "pos", "dim"]
    ) -> Literal["batch", "pos", "vocab"]:
        # CDCD Framework: Apply L2-normalisation to the embedding estimate
        # before calculating the score estimate (__renormalisation__)
        if self.renormalize:
            x = F.normalize(x, p=2, dim=-1)
        return self.unembed(x)


class SelfiesNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hyperparams = cfg.hyperparams
        self.bertconfig = BertConfig(**cfg.bert_cfg)
        # marko/apr17: use pretrained chemberta embeddings
        # self.embedder = BertEmbeddings(self.bertconfig)
        embed_mat = nn.Embedding(self.cfg.vocab_size, self.hyperparams.h_size)
        embed_mat.weight.data.normal_(mean=0.0, std=BertConfig().initializer_range)
        embed_mat, embed_dim = embed_mat.weight, self.hyperparams.h_size

        # Discrete-to-continuous fixed read-in matrix: E ϵ Rⱽˣᴰ
        self.read_in = nn.Sequential(
            PretrainedEmbedding(embed_mat, use_normalization=True, freeze=True),
            *[
                # Bottleneck layer to shrink word embeddings: D → D'
                nn.Linear(embed_dim, self.hyperparams.h_size),
                nn.LayerNorm(self.hyperparams.h_size),
            ]
            if self.hyperparams.h_size != embed_dim
            else [nn.Identity()],
        )
        # Continous-to-discrete learnable read-out matrix as an LM head
        self.read_out = nn.Sequential(
            *[
                # "Add a linear output projection layer E′ which takes the output of
                # the transformer y ∈ Rᴺˣᴰ and projects each element (yᵢ) 1 ≤ i ≤ N
                # back to the same size as the word embeddings, `embed_dim`."
                nn.Linear(self.hyperparams.h_size, embed_dim),
                nn.LayerNorm(embed_dim),
            ]
            if self.hyperparams.h_size != embed_dim
            else [nn.Identity()],
            # Initalize read-out (R) to: Eᵀ ϵ Rᴰˣⱽ
            PretrainedUnembedding(embed_mat, use_renormalization=False),
        )  # E′
        self.bertconfig.hidden_size = self.hyperparams.hh_size
        self.encoder = BertEncoder(self.bertconfig)
        self.in_channels = self.hyperparams.h_size * 2
        self.out_channels = self.hyperparams.h_size
        self.model_channels = self.hyperparams.time_channels
        
        time_embed_dim = self.model_channels * 2
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.hyperparams.hh_size),
        )
        self.input_up_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.hyperparams.hh_size), # *2 for self-conditioning
            nn.Tanh(), 
            nn.Linear(self.hyperparams.hh_size, self.hyperparams.hh_size)
        )
        self.output_down_proj = nn.Sequential(
            nn.Linear(self.hyperparams.hh_size, self.hyperparams.hh_size),
            nn.Tanh(),
            nn.Linear(self.hyperparams.hh_size, self.out_channels)
        )
        self.register_buffer("position_ids", torch.arange(self.bertconfig.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(self.bertconfig.max_position_embeddings, self.bertconfig.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hyperparams.hh_size, eps=self.bertconfig.layer_norm_eps)
        self.dropout = nn.Dropout(self.bertconfig.hidden_dropout_prob)
        # self.conditional_nn = nn.Linear(1,2) #TODO: change dimensions to match actual conditional data
        self.init_weights()
    

    def init_weights(self):
        self.apply(self._initialize_weights)
        # self._tie_or_clone_weights(self.get_logits, self.embedder.word_embeddings)


    def _initialize_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    # def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
    #     """Tie or clone module weights depending of whether we are using TorchScript or not"""
    #     if self.bertconfig.torchscript:
    #         output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
    #     else:
    #         output_embeddings.weight = input_embeddings.weight

    #     if getattr(output_embeddings, "bias", None) is not None:
    #         output_embeddings.bias.data = nn.functional.pad(
    #             output_embeddings.bias.data,
    #             (
    #                 0,
    #                 output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
    #             ),
    #             "constant",
    #             0,
    #         )
    #     if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
    #         output_embeddings.out_features = input_embeddings.num_embeddings

    
    def forward(self, x_t, t, conditional=None):
        x_t = self.input_up_proj(x_t)
        seq_length = x_t.shape[1]
        temb = self.time_embed(timestep_embedding(t, self.model_channels))
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + x_t + temb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        # marko/may8: self-conditional, conditional
        # if conditional is not None:
        #     encoder_hidden_states = self.conditional_nn(conditional)[:,None,:]
        # else:
        #     encoder_hidden_states = None
        #     # encoder_extended_attention_mask = None
        encoder_hidden_states = None
        out = self.encoder(
            emb_inputs,
            encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
        )[0]
        out = self.output_down_proj(out)
        return out




class DiffusionTransformer(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self._starttime = None
        self.cfg = cfg
        self.hyperparams = cfg.hyperparams
        # self.dataset = dataset
        self.tokenizer = tokenizer
        self.timesteps = self.hyperparams.timesteps
        self.save_hyperparameters()
        self.model = SelfiesNet(cfg)
        prediction_type = "sample"
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.timesteps,
            prediction_type=prediction_type
        )

    def denoise_sample(self, bs, num_inference_steps=200, skip_special_tokens=False):
        latent_size = self.hyperparams.h_size
        latents_shape = (bs, self.cfg.max_length, latent_size)
        latents = torch.randn(latents_shape, device=self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma
        self.noise_scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.noise_scheduler.timesteps.to(self.device)
        noise_pred = torch.zeros_like(latents).to(self.device)
        for t in timesteps_tensor:
            # t = t.expand(bs)
            latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
            # predict the text embedding
            noise_pred = self.model(torch.cat((latent_model_input, noise_pred), dim=-1), t.expand(bs))
            # noise_pred = self.model(latent_model_input, t.expand(bs))
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        logits = self.model.read_out(latents)  # bsz, seqlen, vocab
        cands = torch.topk(logits, k=1, dim=-1)
        sample = cands.indices
        generated_string = []
        for seq in sample:
            tokens = self.tokenizer.decode(seq.squeeze(-1), skip_special_tokens=skip_special_tokens)
            generated_string.append(tokens)
        return generated_string

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hyperparams.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.hyperparams.warmup_steps, self.hyperparams.warmup_steps*10)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
        # return optimizer

    def get_loss(self, batch, batch_idx):
        input_ids = batch['input_ids']
        # if 'conditionals' in batch:
        #     conditionals = batch['conditionals']
        # else:
        #     conditionals = None
        x_embeds = self.model.read_in(input_ids)
        bs = x_embeds.shape[0]

        noise = torch.randn(x_embeds.shape).to(x_embeds.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=x_embeds.device).long()
        # x_t for lsimple
        x_t = self.noise_scheduler.add_noise(x_embeds, noise, timesteps)
        # marko/may8: self-conditioning, conditioning
        prev_output = torch.zeros_like(x_t).to(x_embeds.device)
        # if random.random() > 0.5:
        #     conditionals = None
        conditionals = None
        if random.random() > 0.5:
            with torch.no_grad():
                prev_output = self.model(torch.cat((x_t, prev_output), dim=-1), timesteps, conditionals).detach()
        # model_output = self.model(x_t, timesteps)
        model_output = self.model(torch.cat((x_t, prev_output), dim=-1), timesteps, conditionals)
        decoder_nll = token_discrete_loss(model_output, self.model.read_out, input_ids)
        lsimple = mean_flat((x_embeds - model_output) ** 2)
        # marko/may16: try tweeking the weight
        loss = lsimple + 2 * decoder_nll
            

        # TODO: implement bells and whisltes: exponential moving average, self conditioning etc.
        # accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model, 0.9999)
        # self.log("recon", decoder_nll.mean())
        # self.log("mse", lsimple.mean())
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)    
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("embedder weight", self.model.embedder.word_embeddings.weight.sum().item())
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, sync_dist=True)

    def on_train_start(self):
        self._starttime = time.monotonic()

    def on_train_epoch_end(self):
        # marko/may26: for comparing training time
        if self.current_epoch == 0:
            time_used =  time.monotonic() - self._starttime
            self.log("train/time", time_used, sync_dist=True)