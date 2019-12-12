#!/bin/bash

set job_details="sat_resnet50_lstm1_e512_h512_b256"

#SBATCH --job-name=%job_details%
#SBATCH --output="%job_details%.%j.%N.out"
#SBATCH --err="%job_details%.%j.%N.out"

#SBATCH --partition=gpux1
#SBATCH --cpu_per_gpu=16
#SBATCH --time=24

source /opt/apps/anaconda3/etc/profile.d/conda.sh 
conda activate image_captions

python train.py \
	--models_dir %job_details% \
	--vocab_path vocab2014.pkl \
	--image_root ../CS-547/image_captions/train2014 \
	--captions_json ../CS-547/image_captions/annotations/captions_train2014.json \
