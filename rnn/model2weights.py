
import argparse
import os
import model
import torch

# collect input arguments
parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str)
parser.add_argument("--input_fname", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--output_fname", type=str)

args = parser.parse_args()

# do the work
in_fullfile = os.path.join(args.input_path, args.input_fname)
model_object = torch.load(in_fullfile, map_location="cpu")

# check if filename is provided, else append weights to input file name for saving
if args.output_fname is None:
    output_fname = args.input_fname.replace(".pt", "-weights.pt")
else:
    output_fname = args.output_fname

# save the state dict only
out_fullfile = os.path.join(args.output_path, output_fname)
print("Saving {}".format(out_fullfile))
torch.save(model_object.cpu().state_dict(), out_fullfile)


