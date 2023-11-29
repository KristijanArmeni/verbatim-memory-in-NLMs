from ast import literal_eval
import torch
import numpy as np
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import List, Dict, Tuple, Union
import logging

# logging.basicConfig(level=logging.INFO, format="%(message)s")

logger = logging.getLogger("wm_suite.utils")


def shuffle_weights(x: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(12345)

    if len(x.shape) > 1:
        n_rows, n_cols = x.shape

        x_flat = torch.flatten(x)
        shuf_idx = torch.randperm(
            n=x_flat.shape[0]
        )  # flatten tensor to shuffle across rows and cols
        x_new = x_flat[shuf_idx].unflatten(
            dim=0, sizes=(n_rows, n_cols)
        )  # unflatten back to original shape

    elif len(x.shape) == 1:
        shuf_idx = torch.randperm(n=x.shape[0])
        x_new = x[shuf_idx]

    return x_new


class GPT2AttentionAblated(GPT2Attention):
    """
    A subclass of GPT2Attention class for ablating attention heads.
    """

    def __init__(self, attn_instance, ablation_type: str, heads: List, *args, **kwargs):
        """
        Parameters
        ----------
        attn_instance : GPT2Attention
            The instance of GPTAttention class whos parameters are coppied.
            The GPT2Attention._attn method will be overriden to 0 attention weights.
        ablation_type : str
            "zero" or "shuffle", whether or not to zero the weights or shuffle them.
        heads : List[int]
            Zero-indexed list of integers, indicating which heads to ablate (0 to 11).


        Examples
        --------
        ```python
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # select a layer to perform ablations on
        layer_idx = 2    # ablate layer 3
        layer = model.transformer.h[layer_idx]

        # override the .attn attribute with the GPT2AttentionAblated class
        layer.attn = GPT2AttentionAblated(attn_instance=layer.attn,
                                          ablation_type="zero",
                                          heads=[1, 2, 3])
        ```
        """

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
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        if self.ablation_type == "zero":
            attn_weights[:, torch.tensor(self.head_indxs), :, :] = 0

        elif self.ablation_type == "shuffle":
            # can't deal with batch dimensions right now
            assert attn_weights.shape[0] == 1

            n_seq = attn_weights.shape[-1]
            for h in self.head_indxs:
                for seq in torch.arange(n_seq):
                    row = attn_weights[0, h, seq, :]
                    non_zero_vals = torch.where(row > 0)[0]

                    attn_weights[0, h, seq, non_zero_vals] = shuffle_weights(
                        row[non_zero_vals]
                    )

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


def find_topk_attn(attn: np.ndarray, 
                   topk: int, 
                   attn_threshold: Union[float, None],
                   tokens_of_interest: List[int],
                   seed: int,
                   bos_renormalize: bool=False,
                   ) -> Tuple[Dict, Dict, np.ndarray]:

    """
    Takes attn.shape = (samples, timesteps, heads, layer) of attention weights and finds <topk> heads across layers
    that have highest attention scores summed over timesteps <tokens_of_interest> and averaged over samples.

    Parameters
    ----------
    attn : np.ndarray
        array of attention weights (shape = (samples, timesteps, heads, layer))
    topk : int
        top-10 criterion
    attn_threshold : float
        threshold indicating to select only heads with attention score > threshold
        if > 0, it is used in conjunction with the topk criterion (if == 0, it means all heads it is ignored)
    tokens_of_interest: list
        a list of indices representing token positions to sum the attention weights over
    seed : int
        random seed for choosing heads in control ablations
    bos_renormalize : bool
        whether or not to exclude begining-of-sequence (BOS) token and renormalize the attention weights
        relative to the sequence without BOS. `tokens_of_interest` are adjusted by -1 to reflect the shorter sequence


    Returns
    -------
    dict : dict
        A dictionary with every layer as key and selected heads as list entry for each layer key.
    """

    if bos_renormalize:
        logger.info("Renormalizing attention weights to exclude BOS token.")

        # renormalize attention weights
        attn = attn[:, 1::, :, :] / np.sum(attn[:, 1::, :, :], 
                                           axis=1, 
                                           keepdims=True)

        logger.info(f"Renormalized attention weights with new shape: {attn.shape}")
        logger.info(
            "Shifting tokens of interest by - 1 to account for dropped BOS token."
        )

        # shift token indices since we're starting at t + 1
        tokens_of_interest = np.array(tokens_of_interest) - 1

    # aggregate over the select time-window
    sel = np.zeros(shape=attn.shape[1], dtype=bool)

    logger.info(
        f"Finding top-{topk} attn heads across sequence positions {tokens_of_interest}"
    )
    sel[np.array(tokens_of_interest)] = True
    attn_toi = np.sum(attn[:, sel, ...], 1)

    # take the mean across sequences
    attn_toi_avg = np.mean(attn_toi, axis=0)  # shape = (heads, layers)

    # flatten from (heads, layers) and find top-k
    orig_shape = attn_toi_avg.shape
    x = attn_toi_avg.flatten()
    topk_inds = np.argpartition(x, -topk)[-topk:]  # indices of shape (20,)

    # if threshold is provided, make sure to select heads that are top-k and with attention > threshold
    if attn_threshold is not None:
        thrsh_bool = x > attn_threshold  # boolean of shape (144,)

        topk_bool = np.zeros(x.shape, dtype=bool)  # create a boolean of shape (144,)
        topk_bool[topk_inds] = True

        # find indices
        inds = np.where((topk_bool & thrsh_bool) == True)[0]
    else:
        inds = topk_inds

    logger.info(
        f"Found {len(inds)} heads with top-{topk} attn scores and attention score > {attn_threshold}."
    )

    # now create a boolean which is reshaped back to (heads, layers)
    arr_indx = np.zeros(x.shape)
    arr_indx[inds] = True
    arr_indx = arr_indx.reshape(orig_shape)

    # top 20 heads, use these to check that for control we only select non-top20 heads
    top20inds = np.argpartition(x, -20)[-20:]  # these are for control
    top20arr = np.zeros(x.shape)
    top20arr[top20inds] = True
    top20arr = top20arr.reshape(orig_shape)

    rng = np.random.RandomState(seed)

    def select_control_heads(
        array: np.ndarray, negative_array: np.ndarray, values
    ) -> Dict:
        sel_row, sel_col = np.where(
            array == True
        )  # use these to figure out where the control should be
        unsel_row, unsel_col = np.where(
            negative_array == True
        )  # the items in these rows should not be selected

        relevant_cols = np.unique(sel_col)

        ctrl_dict = {l: [] for l in range(array.shape[1])}

        # borrow_from_next_col = 0
        for col in relevant_cols:
            num_heads = int(
                sum(array[:, col])
            )  # count number of indices for which we need controls for
            # num_heads += borrow_from_next_col

            # sample from available indices without repetition
            available_indices = np.where(negative_array[:, col] != True)[0]
            gap = int(num_heads - len(available_indices))

            # if there's not enough available indices to sample from
            # make sure we grab some from the ones that are already selected based on the lowest
            # attention score
            if gap > 0:
                # borrow_from_next_col = gap
                taken_indices = np.where(negative_array[:, col] == True)[0]
                smallest_values = np.sort(values[taken_indices, col])[
                    0:gap
                ]  # find the heads with <gap> smallest values

                extra_indices = np.where(np.in1d(values[:, col], smallest_values))[0]

                logger.info(f"Need {len(extra_indices)} extra indices in layer {col}")
                available_indices = np.hstack([available_indices, extra_indices])
                num_heads = len(available_indices)

            ctrl_idx = rng.choice(
                a=available_indices, size=num_heads, replace=False
            )  # choose among heads that are not in negative array

            ctrl_dict[col] = ctrl_idx.tolist()

        return ctrl_dict

    topk_heads = {
        l: np.where(arr_indx[:, l])[0].tolist() for l in range(arr_indx.shape[0])
    }

    topk_control = select_control_heads(arr_indx, top20arr, attn_toi_avg)

    return topk_heads, topk_control, attn_toi_avg


def from_dict_to_labels(layer_head_dict: Dict) -> List[str]:
    """
    Converts a dictionary specifying which heads to ablate in which layers (e.g. {0: [1, 2], 1: [3]}) to a list of labels (e.g. ["L0.H1", "L0.H2", "L1.H3"]).

    Parameters
    ----------
    layer_head_dict : Dict
        A dictionary specifying which heads to ablate in which layers, e.g. {0: [1, 2], 1: [3]}

    Returns
    -------
    List[str]
        A list of labels, e.g. ["L0.H1", "L0.H2", "L1.H3"]

    """

    return [
        f"L{l}.H{h}"
        for i, l in enumerate(layer_head_dict)
        for h in layer_head_dict[l]
        if layer_head_dict[l]
    ]


def from_labels_to_dict(labels: List[str]) -> Dict:
    """
    Converts a list of labels (e.g. ["L0.H1", "L0.H2", "L1.H3"]) to a dictionary specifying which heads to ablate in which layers.

    Parameters
    ----------
    labels : List[str]
        A list of labels, e.g. ["L0.H1", "L0.H2", "L1.H3"]

    Returns
    -------
    Dict
        A dictionary specifying which heads to ablate in which layers, e.g. {0: [1, 2], 1: [3]}

    """
    out = {l: [] for l in range(12)}

    for label in labels:
        l_idx = int(
            label.split(".")[0][1::]
        )  # split and grab first interger, e.g. 'L2.H11'
        h_idx = int(
            label.split(".")[1][1::]
        )  # split and grab first interger, e.g. 'L2.H11'
        out[l_idx].append(h_idx)

    # filter empty fields
    out2 = {l: out[l] for l in out.keys() if out[l]}

    return out2


def find_topk_intersection(attn, tois: List, topk: int, seed: int) -> Dict:
    dict1, _, _ = find_topk_attn(
        attn=attn, topk=topk, tokens_of_interest=tois[0], seed=seed
    )
    dict2, _, _ = find_topk_attn(
        attn=attn, topk=topk, tokens_of_interest=tois[1], seed=seed
    )

    labels1 = from_dict_to_labels(dict1)
    labels2 = from_dict_to_labels(dict2)

    intersection = set(labels1) & set(labels2)

    print("Intesection elements: ", intersection)

    layer_head_dict = from_labels_to_dict(list(intersection))

    return layer_head_dict


def ablate_attn_module(model, layer_head_dict, ablation_type):
    """
    Parameters
    ----------
    model : GPT2LMHeadModel
        instance of the transformers.GPT@LMHeadModel class whose GPT2Attention classes
        will be overriden with GPT2AttentionAblated class.
    layer_head_dict: dict
        A dict specifying which heads in which layers to ablate, e.g. {0: [0, 1, 2], 1: [], 2: [5, 6, 12]}

    Returns
    -------
    GPT2LMHeadModel
        Instance of GPT2LMHeadModel with GPT2AttentionAblation module assigned to the specified layers.

    Examples
    --------
    from transformers import GPT2LMHeadModel

    # initiate a gpt2 model instance
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # inititate a model with ablated heads 0, 1, 2 in layer 0 and heads 10, 11, 12 in layer 1
    model_ablated = ablate_attn_module(model,
                                      layer_head_dict={0: [0, 1, 2], 1: [10, 11, 12]},
                                      ablation_type="zero")

    """

    # ablate only layers that have certain heads selected
    layers = [i for i in layer_head_dict.keys() if layer_head_dict[i]]

    print_dict = {
        i: layer_head_dict[i] for i in layer_head_dict.keys() if layer_head_dict[i]
    }
    logger.info(
        f"Setting GPT2AttentionAblated({ablation_type}) in layers and heads:\n{print_dict}."
    )

    for layer_idx in layers:
        layer = model.transformer.h[layer_idx]
        layer.attn = GPT2AttentionAblated(
            attn_instance=layer.attn,
            ablation_type=ablation_type,
            heads=layer_head_dict[layer_idx],
            config=model.config,
        )

    return model


def get_pairs(lh_dict: Dict) -> List[Dict]:
    """
    Takes a dict with layers as keys and heads as values and returns a list of dicts with all possible pairs of heads (one dict for each pair).

    Parameters
    ----------
    lh_dict : Dict
        A dictionary (output of get_lh_dict()) specifying heads and layers to be ablated

    Returns
    -------
    List[Dict]
        A list of dicts with all possible pairs of heads (one dict for each pair).
    """

    # enumerate all pairs
    all_pairs = [(l, h) for l in lh_dict.keys() for h in lh_dict[l]]

    n_el = len(all_pairs)

    # populate a square matrix of all pair combinations with string formatted as list
    b = np.zeros((n_el, n_el), dtype=object)
    for i in range(n_el):
        for j in range(n_el):
            b[i, j] = (
                "[" + str(all_pairs[i]) + ", " + str(all_pairs[j]) + "]"
            )  # a dict as str

    # get the lower triangular values (without the diagonal)
    low_trig = np.tril(b, -1)
    rows, cols = low_trig.nonzero()  # find non-zero rows and column indices

    def string2dict(instr: str) -> List:
        """
        Helper function to convert string formated as alist into a python dict
        """
        tup1 = literal_eval(instr)[0]
        tup2 = literal_eval(instr)[1]

        # if the two heads are from the same layer, only one key in the dict is needed
        if tup1[0] == tup2[0]:
            out = {tup1[0]: [tup1[1], tup2[1]]}
        # if across layers, just do a normal list comprehension, with a separate key for each layer
        else:
            out = {tup[0]: [tup[1]] for tup in literal_eval(instr)}

        return out

    pairs = []
    for i in range(len(rows)):
        pairs.append(string2dict(low_trig[rows[i], cols[i]]))

    return pairs


def get_triplets(lh_dict: Dict, pair: List) -> List[Tuple]:
    """
    Parameters
    ----------
    lh_dict : Dict
        a dictionary (output of get_lh_dict()) specifying heads and layers to be ablated
    pair : List
        specifying the selected pair of heads to combine with the rest in top-20

    Returns
    -------
    List[Tuple]
        a list containing all triplet pairs
    """

    logger.info(
        f"Finding {len([(l, h) for l in lh_dict.keys() for h in lh_dict[l] if lh_dict[l]])} triplets for pair {pair}"
    )

    heads = from_dict_to_labels(lh_dict)

    # combine all heads with the selected pair
    triplets = [(e, pair[0], pair[1]) for e in heads if e not in pair]

    return triplets


def get_lh_dict(attn, which: str, seed=None) -> List:
    """
    Helper function defining the possible ablation combinations in wm_test_suite based on attention pattenrs in attn_dict
    Parameters:
    ----------
    """

    lh_dict = None

    if seed is None:
        seed = 12345  # a dummy seed, find_topk_attn expects it
        return_random = False
    elif seed is not None:
        return_random = True
        logger.info(
            "Seed argument provided to get_lh_dict, returning random selection..."
        )

    # ===== PAIRED ABLATIONS ===== #
    if which == "top-20-matching-pairs":
        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[13], seed=seed
        )
        dicts = get_pairs(lh_dict)

    elif which == "top-20-postmatching-pairs":
        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[14, 16, 18], seed=seed
        )
        dicts = get_pairs(lh_dict)

    elif which == "top-20-recent-pairs":
        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[42, 43, 44], seed=seed
        )
        dicts = get_pairs(lh_dict)

    elif which == "top-20-matching-triplets":
        # get maximum pair (0-indexing)
        max_pair = ["L0.H10", "L1.H11"]

        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[13], seed=seed
        )

        combs = get_triplets(lh_dict, max_pair)

        dicts = [from_labels_to_dict(l) for l in combs]

    elif which == "top-20-postmatching-triplets":
        # get maximum pair (0-indexing)
        max_pair = ["L10.H11", "L10.H0"]

        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[14, 16, 18], seed=seed
        )

        combs = get_triplets(lh_dict, max_pair)

        dicts = [from_labels_to_dict(l) for l in combs]

    elif which == "top-20-recent-triplets":
        # get maximum pair (0-indexing)
        max_pair = ["L3.H2", "L2.H3"]

        lh_dict, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[42, 43, 44], seed=seed
        )

        combs = get_triplets(lh_dict, max_pair)

        dicts = [from_labels_to_dict(l) for l in combs]

    # ==== MULTI-HEAD ABLATIONS ===== #

    elif which in [
        "top-5-postmatching",
        "top-10-postmatching",
        "top-15-postmatching",
        "top-20-postmatching",
    ]:
        k = int(which.split("-")[1])  # extract the k value from the string
        out_tuple = find_topk_attn(
            attn, topk=k, tokens_of_interest=[14, 16, 18], seed=seed
        )

        lh_dict = out_tuple[0]
        if return_random:
            lh_dict = out_tuple[1]

        dicts = [lh_dict]

    elif which in [
        "top-5-matching",
        "top-10-matching",
        "top-15-matching",
        "top-20-matching",
    ]:
        k = int(which.split("-")[1])  # extract the k value from the string
        out_tuple = find_topk_attn(attn, topk=k, tokens_of_interest=[13], seed=seed)

        lh_dict = out_tuple[0]
        if return_random:
            lh_dict = out_tuple[1]

        dicts = [lh_dict]

    elif which in ["top-5-recent", "top-10-recent", "top-15-recent", "top-20-recent"]:
        k = int(which.split("-")[1])  # extract the k value from the string
        out_tuple = find_topk_attn(
            attn, topk=k, tokens_of_interest=[44, 43, 42], seed=seed
        )

        lh_dict = out_tuple[0]
        if return_random:
            lh_dict = out_tuple[1]

        dicts = [lh_dict]

    elif which in ["induction-matching-intersect"]:
        lh_dict = find_topk_intersection(
            attn, tois=([13], [14, 16, 18]), topk=20, seed=12345
        )

        dicts = [lh_dict]

    elif which == "matching-bottom5":
        # find effect of ablation of onlyt the last 5 matching heads (seems to have disproportionaly strong effect)
        top15_matching, _, _ = find_topk_attn(
            attn, topk=15, tokens_of_interest=[13], seed=12345
        )
        top20_matching, _, _ = find_topk_attn(
            attn, topk=20, tokens_of_interest=[13], seed=12345
        )

        top15labels, top20labels = (
            from_dict_to_labels(top15_matching),
            from_dict_to_labels(top20_matching),
        )

        resid_labels = list(set(top20labels) - set(top15labels))

        lh_dict = from_labels_to_dict(resid_labels)

        dicts = [lh_dict]

    return dicts
