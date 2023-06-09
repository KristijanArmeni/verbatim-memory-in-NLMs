
import json
import torch
import numpy as np
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import List, Dict, Tuple
import logging
import transformers

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


def find_topk_attn(attn: np.ndarray, topk: int, tokens_of_interest: List, seed: int) -> Dict:

    """
    Takes attn.shape = (samples, timesteps, heads, layer) of attention weights and finds <topk> heads across layers
    that have highest attention scores summed over timesteps <tokens_of_interest> and averaged over samples.

    Parameters:
    attn : np.ndarray, shape = (samples, timesteps, heads, layer)
        array of attention weights
    topk : int
        top-10 criterion

    Returns:
    dict : dict
        A dictionary with every layer as key and selected heads as list entry for each layer key.
    """

    # aggregate over the select time-window
    sel = np.zeros(shape=attn.shape[1], dtype=bool)

    logging.info(f"Finding top-{topk} attn heads across sequence positions {tokens_of_interest}")
    sel[np.array(tokens_of_interest)] = True
    attn_toi = np.sum(attn[:, sel, ...], 1)

    # take the mean across sequences
    attn_toi_avg = np.mean(attn_toi, axis=0)  # shape = (heads, layers)

    # flatten from (heads, layers) and find top-k
    orig_shape = attn_toi_avg.shape
    x = attn_toi_avg.flatten()
    inds = np.argpartition(x, -topk)[-topk:]

    # now create a boolean which is reshaped back to (heads, layers)
    arr_indx = np.zeros(x.shape)
    arr_indx[inds] = True
    arr_indx = arr_indx.reshape(orig_shape)

    # top 20 heads, use these to check that for control we only select non-top20 heads
    top20inds = np.argpartition(x, -20)[-20:]   # these are for control
    top20arr = np.zeros(x.shape)
    top20arr[top20inds] = True
    top20arr = top20arr.reshape(orig_shape)

    rng = np.random.RandomState(seed)

    def select_control_heads(array: np.ndarray, negative_array: np.ndarray, values) -> Dict:

        sel_row, sel_col = np.where(array == True)       # use these to figure out where the control should be
        unsel_row, unsel_col = np.where(negative_array == True)  # the items in these rows should not be selected

        relevant_cols = np.unique(sel_col)
        
        ctrl_dict = {l: [] for l in range(array.shape[1])}
        
        #borrow_from_next_col = 0
        for col in relevant_cols:

            num_heads = int(sum(array[:, col]))   # count number of indices for which we need controls for
            #num_heads += borrow_from_next_col

            # sample from available indices without repetition
            available_indices = np.where(negative_array[:, col] != True)[0]
            gap = int(num_heads - len(available_indices))

            # if there's not enough available indices to sample from
            # make sure we grab some from the ones that are already selected based on the lowest
            # attention score
            if gap > 0:
                
                #borrow_from_next_col = gap
                taken_indices = np.where(negative_array[:, col] == True)[0]
                smallest_values = np.sort(values[taken_indices, col])[0:gap]   # find the heads with <gap> smallest values
                
                extra_indices = np.where(np.in1d(values[:, col], smallest_values))[0]

                print(f"Need {len(extra_indices)} extra indices in layer {col}")
                available_indices = np.hstack([available_indices, extra_indices])
                num_heads = len(available_indices)
            
            ctrl_idx = rng.choice(a=available_indices,
                                  size=num_heads,
                                  replace=False)  # choose among heads that are not in negative array
            
            ctrl_dict[col] = ctrl_idx.tolist()

        return ctrl_dict
            
    topk_heads = {l: np.where(arr_indx[:, l])[0].tolist() for l in range(arr_indx.shape[0])}

    topk_control = select_control_heads(arr_indx, top20arr, attn_toi_avg)

    return topk_heads, topk_control, attn_toi_avg


def ablate_attn_module(model, layer_head_dict, ablation_type):
    """
    Parameters:
    ----------
    model : hugginface model
    layer_head_dict: dict
        dict specifying which heads in which layers to ablate, e.g. {0: [0, 1, 2], 1: [], 2: [5, 6, 12]}
    """
    
    # ablate only layers that have certain heads selected
    layers = [i for i in layer_head_dict.keys() if layer_head_dict[i]]

    print_dict = {i: layer_head_dict[i] for i in layer_head_dict.keys() if layer_head_dict[i]}
    logging.info(f"Setting GPT2AttentionAblated({ablation_type}) in layers and heads:\n{print_dict}.")

    for layer_idx in layers:

        layer = model.transformer.h[layer_idx]
        layer.attn = GPT2AttentionAblated(attn_instance=layer.attn, 
                                          ablation_type=ablation_type, 
                                          heads=layer_head_dict[layer_idx], 
                                          config=model.config)

    return model
