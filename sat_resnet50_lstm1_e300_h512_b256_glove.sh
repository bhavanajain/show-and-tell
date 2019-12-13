#!/bin/bash

#SBATCH --job-name="sat_resnet50_lstm1_e512_h512_b256_glove"
#SBATCH --output="sat_resnet50_lstm1_e512_h512_b256_glove.%j.%N.out"
#SBATCH --error="sat_resnet50_lstm1_e512_h512_b256_glove.%j.%N.out"
#SBATCH --partition=gpux4
#SBATCH --gres=gpu:v100:1
##SBATCH --cpus-per-gpu=16
#SBATCH --time=36:00:00

source /opt/apps/anaconda3/etc/profile.d/conda.sh 
conda activate image_captions

srun  python train.py \
	--models_dir sat_resnet50_lstm1_e512_h512_b256_glove \
	--vocab_path vocab2014.pkl \
	--image_root train2014/ \
	--captions_json annotations/captions_train2014.json \
	--embed_size 300 \
	--rnn_type lstm \
	--resnet_size 50 \
	--glove_embed_path glove_embeddings.pkl


