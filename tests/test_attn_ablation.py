import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from src.wm_suite.wm_ablation import ablate_attn_module
from typing import Tuple, Dict, List
from matplotlib import pyplot as plt


##### ===== ######
def plot_attentions(data: List, layer: int, head: int, ticklabels: List):

    fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey="all")

    d = []
    for tens in data:
        d.append(tens[layer][0, head, :, :].detach().cpu().numpy())

    ax[0].imshow(d[0], aspect='auto')
    ax[0].set_title("Unablated model")

    ax[1].imshow(d[1], aspect='auto')
    ax[1].set_title("Zeroing attention scores")

    im3 = ax[2].imshow(d[2], aspect='auto')
    ax[2].set_title("Shuffling attention scores")

    ax[0].set_yticks(np.arange(len(ticklabels)))
    ax[0].set_yticklabels(ticklabels)
    ax[0].set_ylabel("Queries")
    

    cax = ax[2].inset_axes([1.04, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im3, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Attention weight", rotation=90)

    for a in ax:
        a.set_xticks(np.arange(len(ticklabels)))
        a.set_xticklabels(ticklabels, rotation=45)
        a.set_xlabel(f"Keys (L{layer+1}.H{head+1})")

    return fig


def plot_head_attentions(data, layer:int, heads:List, ticklabels:List):

    fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey="all")

    d = []
    for head in heads:
        d.append(data[layer][0, head, :, :].detach().cpu().numpy())

    for i, a in enumerate(ax):
        im = a.imshow(d[i], aspect="auto")
        a.set_title(f"Head {heads[i]+1}")

        a.set_xticks(np.arange(len(ticklabels)))
        a.set_xticklabels(ticklabels, rotation=45)
        a.set_xlabel(f"Keys (L{layer+1}.H{heads[i]+1})")

    cax = ax[-1].inset_axes([1.02, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Attention weight", rotation=90)

    return fig


def test_attn_ablation():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model0 = GPT2LMHeadModel.from_pretrained("gpt2")
    model1 = GPT2LMHeadModel.from_pretrained("gpt2")
    model2 = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    test_input = "A cat and a dog walk into a bar. The dog said hello."
    inputs = torch.tensor(tokenizer.encode(test_input))

    layer = 11
    heads = list(range(12))
    model0 = ablate_attn_module(model0, layers=[layer], heads=heads, ablation_type="zero")
    model1 = ablate_attn_module(model1, layers=[layer], heads=heads, ablation_type="shuffle")
    model2 = ablate_attn_module(model2, layers=[5], heads=heads, ablation_type="zero")

    # test that model parameters are not modified by ablations (only attention activations) etc.
    params1 = model.transformer.h[layer].attn.parameters()
    params2 = model0.transformer.h[layer].attn.parameters()
    params3 = model1.transformer.h[layer].attn.parameters()

    # Check that the parameters are not affected
    for p1, p2 in zip(params1, params2):
        assert torch.all(p1 == p2)
    for p2, p3 in zip(params2, params3):
        assert torch.all(p2 == p3)

    # extract attention matrices and hidden states
    data = []
    hs = []
    for m in (model, model0, model1, model2):
        m.eval()
        outputs = m(inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        hs.append(outputs.hidden_states)
        data.append(outputs.attentions)

    # check that attentions are zeroed as expected
    ablated_attentions_L11 = data[1][11]     # layer 12
    ablated_attentions_L5 = data[3][5]       # layer 5
    nonablated_attentions_L0 = data[1][0]    # layer 1

    assert torch.all(ablated_attentions_L11 == 0)
    assert torch.all(ablated_attentions_L5 == 0)
    assert torch.any(nonablated_attentions_L0 != 0)

    # check hidden states
    h, h6, h11 = hs[0][1::], hs[3][1::], hs[1][1::]  # grab non-initial elements, the first elements are embeddings
    intact_vs_6 = []
    intact_vs_11 = []
    for l in range(12):
        intact_vs_6.append(torch.any(h[l] != h6[l]).item())   # there should be at least some non equal entries (i.e. hiddens states should not be exactly equal)
        intact_vs_11.append(torch.any(h[l] != h11[l]).item())

    # inequality from layer 5 onwards should hold
    assert np.all(np.array(intact_vs_6)[5::])              # only hidden states after layer 5 should be different from intact model
    assert np.where(np.array(intact_vs_11))[0][0] == 11    # only hidden states at layer eleven should be different from intact model

    # ===== PLOTS ===== #
    savedir = "/home/ka2773/project/lm-mem/src/tests/fig"

    ticklabels = [e.strip("Ġ") for e in tokenizer.convert_ids_to_tokens(inputs)]

    l, h = 11, 8
    fig = plot_attentions(data, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    l, h = 6, 8
    fig = plot_attentions(data, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    l, h = 2, 8
    fig = plot_attentions(data, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    plt.close("all")

    del model, model0, model1, model2, data


    # ===== TEST INDIVIDUAL HEAD ABLATIONS ===== #

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model2 = GPT2LMHeadModel.from_pretrained("gpt2")
    model3 = GPT2LMHeadModel.from_pretrained("gpt2")

    layer = 1
    heads = list(range(12))
    model2 = ablate_attn_module(model2, layers=[layer], heads=heads, ablation_type="zero")
    model3 = ablate_attn_module(model3, layers=[layer], heads=heads, ablation_type="shuffle")

    data2 = []
    for m in (model, model2, model3):
        m.eval()
        data2.append(m(inputs, output_attentions=True).attentions)

    l, h = 11, 8
    fig = plot_attentions(data2, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    plt.close()

    l, h = 6, 8
    fig = plot_attentions(data2, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    plt.close()

    l, h = 1, 8
    fig = plot_attentions(data2, layer=l, head=h, ticklabels=ticklabels)
    fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

    plt.close()

    del model, model2, model3, data2


    # ===== TEST INDIVIDUAL HEAD ABLATIONS ===== #

    model1 = GPT2LMHeadModel.from_pretrained("gpt2")
    model2 = GPT2LMHeadModel.from_pretrained("gpt2")

    layer = 0
    heads = [0]
    model1 = ablate_attn_module(model1, layers=[layer], heads=heads, ablation_type="zero")

    data3 = []
    for m in (model1, model2):
        m.eval()
        data3.append(m(inputs, output_attentions=True).attentions)

    fig = plot_head_attentions(data3[0], layer=0, heads=[0, 1, 2], ticklabels=ticklabels)
    fig.suptitle(f"Single head ablation (abl. L.{int(layer)+1}-H.{heads[0]+1})")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"attn_head-abl-{int(layer)+1}-{heads[0]}.png"), dpi=300)

    fig = plot_head_attentions(data3[1], layer=0, heads=[0, 1, 2], ticklabels=ticklabels)
    fig.suptitle(f"Unablated heads")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"unablated_heads-{int(layer)+1}-{heads[0]}.png"), dpi=300)

    plt.close()