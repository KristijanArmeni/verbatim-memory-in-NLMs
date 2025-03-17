from wm_suite.io.prepare_transformer_inputs import get_inputs_targets_path_patching
from wm_suite.io.attn_circuits import get_circuit
from wm_suite.wm_ablation import from_labels_to_dict

import os
import numpy as np
import torch
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching
import transformer_lens.utils as utils
import einops
import matplotlib.pyplot as plt


def get_logit_diff(logits: torch.tensor, answer_token_indices: torch.tensor):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]

    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))

    return (correct_logits - incorrect_logits).mean()


def run_patching_experiment(batch_id=0):
    
    inps, targets = get_inputs_targets_path_patching(batch_size=5)

    # initialize the transformerlens model
    model = HookedTransformer.from_pretrained("gpt2")

    # get the batch (5 samples per batch)
    inps_clean = inps[0][batch_id]
    inps_corrupted = inps[1][batch_id]
    target_ids = targets[batch_id]

    # run the model and store the cache for patching
    logits_clean, cache_clean = model.run_with_cache(inps_clean)
    logits_corrupted, cache_corrupted = model.run_with_cache(inps_corrupted)

    CLEAN_DIFF_BASELINE = get_logit_diff(logits_clean.cpu(), target_ids)
    CORRUPTED_DIFF_BASELINE = get_logit_diff(logits_corrupted.cpu(), target_ids)

    def normalized_logit_diff(logits, answer_token_indices=target_ids.cuda()):
        return (
            get_logit_diff(logits, answer_token_indices) - CORRUPTED_DIFF_BASELINE
        ) / (CLEAN_DIFF_BASELINE - CORRUPTED_DIFF_BASELINE)

    # do the patching from clean cache on corrupted inputs
    attn_patch_out = patching.get_act_patch_attn_head_out_by_pos(
        model, inps_corrupted.cuda(), cache_clean, normalized_logit_diff
    )

    return attn_patch_out


def make_plot(patching_output):
    fig, ax = plt.subplots(figsize=(4, 4))

    mat = patching_output.cpu().detach().numpy()[:, -1, :]  # shape = (layer, pos, head)

    absmax = np.round(np.abs(mat).max(), 2)
    im = ax.imshow(mat, cmap="RdBu", origin="lower", vmin=-absmax, vmax=absmax)

    cax = ax.inset_axes([1.04, 0, 0.05, 1])

    ax.set_title("Patching effect")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    cbar = fig.colorbar(im, cax=cax)

    cbar.ax.set_ylabel("Logit difference\n(correct noun - incorrect noun)")

    return fig, ax


def main(input_args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run patching experiment")
    parser.add_argument(
        "--batch_id",
        type=int,
        required=True,
        default=0,
        help="Batch id to run patching on",
    )
    parser.add_argument(
        "--savedir", type=str, default=None, help="Directory to save results to"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    out = run_patching_experiment(batch_id=args.batch_id)

    if args.savedir:
        np.save(
            os.path.join(args.savedir, f"patching_batch-{args.batch_id:02d}.npy"),
            out.cpu().detach().numpy(),
        )

    return None


if __name__ == "__main__":
    main()
