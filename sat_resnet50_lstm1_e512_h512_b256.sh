#!/bin/bash

#SBATCH --job-name=sat_resnet50_lstm1_e512_h512_b256
#SBATCH --output="sat_resnet50_lstm1_e512_h512_b256.%j.%N.out"
#SBATCH --err="sat_resnet50_lstm1_e512_h512_b256.%j.%N.out"

#SBATCH --partition=gpux1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=36

source /opt/apps/anaconda3/etc/profile.d/conda.sh 
conda activate image_captions

python train.py \
	--models_dir sat_resnet50_lstm1_e512_h512_b256 \
	--vocab_path vocab2014.pkl \
	--image_root ../CS-547/image_captions/train2014 \
	--captions_json ../CS-547/image_captions/annotations/captions_train2014.json \