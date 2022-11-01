#!/bin/bash
#SBATCH --job-name=bert_post_processing_script.sh
#SBATCH --time=03:30:00
#SBATCH --mem 4gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=end
#SBATCH --mail-user=karmeni1@jhu.edu
#SBATCH --output=/scratch/ka2773/project/lm-mem/logs/wm_eval/bert/bert_post_processing_script.log
#SBATCH --error=/scratch/ka2773/project/lm-mem/logs/wm_eval/bert/bert_post_processing_script.sh.err


singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh

conda activate core_env

python /home/ka2773/project/lm-mem/src/preprocess_and_merge_csvfiles.py --arch bert --model_id b-10 --scenario sce1 --output_dir /scratch/ka2773/project/lm-mem/output/bert

"