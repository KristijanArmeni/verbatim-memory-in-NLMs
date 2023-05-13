
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def shuffle_weights(x: torch.Tensor) -> torch.Tensor:

    torch.manual_seed(12345)

    if len(x.shape) > 1:

        n_rows, n_cols = x.shape

        x_flat = torch.flatten(x)
        shuf_idx = torch.randperm(n=x_flat.shape[0])        # flatten tensor to shuffle across rows and cols
        x_new = x_flat[shuf_idx].unflatten(dim=0, sizes=(n_rows, n_cols))  # unflatten back to original shape

    elif len(x.shape) == 1:

        shuf_idx = torch.randperm(n=x.shape[0])
        x_new = x[shuf_idx]

    return x_new


class GPT2AttentionAblated(GPT2Attention):

    def __init__(self, attn_instance, ablation_type:str, heads: List, *args, **kwargs):

        super(GPT2AttentionAblated, self).__init__(*args, **kwargs)

        # copy parameters of the parent module
        self.register_buffer("bias", attn_instance._buffers["bias"])
        self.register_buffer("masked_bias", attn_instance._buffers["masked_bias"])

        self.c_attn = attn_instance.c_attn
        self.c_proj = attn_instance.c_proj
        self.attn_dropout = attn_instance.attn_dropout
        self.resid_dropout = attn_instance.resid_dropout
        self.scale_attn_weights = attn_instance.scale_attn_weights

        self.num_head = attn_instance.num_heads
        self.embed_dim = attn_instance.embed_dim
        self.split_size = self.embed_dim

        self.ablation_type = ablation_type
        self.head_indxs = heads

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        if self.ablation_type == "zero":
            attn_weights[0, torch.tensor(self.head_indxs), :, :] = 0

        elif self.ablation_type == "shuffle":

            # can't deal with batch dimensions right now
            assert attn_weights.shape[0] == 1
            
            n_seq = attn_weights.shape[-1]
            for h in self.head_indxs:
                for seq in torch.arange(n_seq):
                    row = attn_weights[0, h, seq, :]
                    non_zero_vals = torch.where(row > 0)[0]

                    attn_weights[0, h, seq, non_zero_vals] = shuffle_weights(row[non_zero_vals])

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


def ablate_attn_module(model, layers, heads, ablation_type):

    logging.info(f"Setting GPT2AttentionAblated({ablation_type}) in layers {layers}, heads {heads}.")

    for layer_idx in layers:

        layer = model.transformer.h[layer_idx]
        layer.attn = GPT2AttentionAblated(attn_instance=layer.attn, 
                                          ablation_type=ablation_type, 
                                          heads=heads, 
                                          config=model.config)

    return model
