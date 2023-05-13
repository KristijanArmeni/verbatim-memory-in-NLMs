import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from src.src.wm_suite.wm_ablation import ablate_attn_module
from typing import Tuple, Dict, List
from matplotlib import pyplot as plt



##### ===== ######
def plot_attentions(data: List, layer: int, head: int):

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

    ticklabels = [e.strip("Ġ") for e in tokenizer.convert_ids_to_tokens(inputs)]

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


model = GPT2LMHeadModel.from_pretrained("gpt2")
model0 = GPT2LMHeadModel.from_pretrained("gpt2")
model1 = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

test_input = "A cat and a dog walk into a bar. The dog said hello."
inputs = torch.tensor(tokenizer.encode(test_input))

layer = 11
heads = list(range(12))
model0 = ablate_attn_module(model0, layers=[layer], heads=heads, ablation_type="zero")
model1 = ablate_attn_module(model1, layers=[layer], heads=heads, ablation_type="shuffle")

data = []
for m in (model, model0, model1):
    m.eval()
    data.append(m(inputs, output_attentions=True).attentions)


savedir = "/home/ka2773/project/lm-mem/src/tests/fig"

l, h = 11, 8
fig = plot_attentions(data, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

l, h = 6, 8
fig = plot_attentions(data, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

l, h = 2, 8
fig = plot_attentions(data, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

plt.close("all")

del model, model0, model1

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
fig = plot_attentions(data2, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

plt.close()

l, h = 6, 8
fig = plot_attentions(data2, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

plt.close()

l, h = 1, 8
fig = plot_attentions(data2, layer=l, head=h)
fig.suptitle(f"Comparing two different attention ablation methods (abl. L.{int(layer)+1}-H.{heads[0]+1}-{heads[-1]+1})")
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"attn_abl-{int(layer)+1}-{heads[0]-heads[-1]}_L{l+1}-H{h+1}.png"), dpi=300)

plt.close()