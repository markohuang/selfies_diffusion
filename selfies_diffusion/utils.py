import os
import math
import torch
import torch.nn as nn

from typing import Tuple, Literal
import transformers

# decoder_nll
def token_discrete_loss(x_t, get_logits, input_ids):
    logits = get_logits(x_t)  # bsz, seqlen, vocab
    # print(logits.shape)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    decoder_nll = loss_fct(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1)).reshape(input_ids.shape)
    # print(decoder_nll.shape)
    decoder_nll = decoder_nll.mean(dim=-1)
    return decoder_nll

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# from functools import cache
# @cache
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def auto_extract_embed_mat(
    model_name: str = "bert-base-uncased",
) -> Tuple[Literal["vocab", "embed"], int]:
    """Extracts a pre-trained word embedding lookup matrix, E ϵ Rᴰˣⱽ, from the
    specified model.
    """
    # Extract the pre-trained word embedding lookup table
    embed_model = transformers.AutoModel.from_pretrained(model_name)
    embeddings = embed_model.get_input_embeddings()  # Word embeddings layer
    embed_mat = embeddings.get_parameter("weight").detach()
    embed_dim = embeddings.embedding_dim
    del embed_model
    return embed_mat, embed_dim

def extract_chemberta_embed_mat(root):
    return torch.load(os.path.join(root, "chemberta/chemberta_embed_mat.pt")), torch.load(os.path.join(root, "chemberta/chemberta_embed_dim.pt"))