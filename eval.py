import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, Decoder
from utils import collate_fn
from vocab import build_vocab
import pickle
import argparse
import os
from torchvision import transforms
from dataset import ImageDataset
import numpy as np
import json

def main(args):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Evaluating on {device}")

	with open(args.vocab_path, 'rb') as f:
		vocab_object = pickle.load(f)
	print(f"Loaded the vocabulary object from {args.vocab_path}, total size={len(vocab_object)}")

	if args.glove_embed_path is not None:
		with open(args.glove_embed_path, 'rb') as f:
			glove_embeddings = pickle.load(f)
		print(f"Loaded the glove embeddings from {args.glove_embed_path}, total size={len(glove_embeddings)}")

		# We are using 300d glove embeddings
		args.embed_size = 300

		weights_matrix = np.zeros((len(vocab_object), args.embed_size))

		for word, index in vocab_object.word2index.items():
			if word in glove_embeddings:
				weights_matrix[index] = glove_embeddings[word]
			else:
				weights_matrix[index] = np.random.normal(scale=0.6, size=(args.embed_size, ))

		weights_matrix = torch.from_numpy(weights_matrix).float().to(device)

	else:
		weights_matrix = None


	img_transforms = transforms.Compose([
		                transforms.Resize((224, 224)),
		                transforms.ToTensor(),
		                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	                ])

	val_dataset = ImageDataset(args.image_root, img_transforms)

	val_dataloader = torch.utils.data.DataLoader(
		dataset=val_dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=args.num_workers)

	encoder = Encoder(args.resnet_size, (3, 224, 224), args.embed_size)
	encoder = encoder.eval().to(device)
	decoder = Decoder(args.rnn_type, weights_matrix, len(vocab_object), args.embed_size, args.hidden_size)
	decoder = decoder.eval().to(device)
	
	model_ckpt = torch.load(args.eval_ckpt_path, map_location=lambda storage, loc: storage)
	encoder.load_state_dict(model_ckpt['encoder'])
	decoder.load_state_dict(model_ckpt['decoder'])
	print(f"Loaded model from {args.eval_ckpt_path}")

	val_results = []

	total_examples = len(val_dataloader)
	for i, (images, image_ids) in enumerate(val_dataloader):
		import pdb; pdb.set_trace();
		images = images.to(device)

		with torch.no_grad():
			image_embeddings = encoder(images)
			captions_wid = decoder.sample_batch(image_embeddings, args.caption_maxlen)

		captions_wid = captions_wid.cpu().numpy()
		captions = []
		for caption_wid in captions_wid:
			caption_words = []
			for word_id in caption_wid:
				word = vocab_object.index2word[word_id]
				caption_words.append(word)
				if word == '<end>':
					break
			captions.append(' '.join(caption_words[1:-2]))

		for image_id, caption in zip(image_ids, captions):
			val_results.append({'image_id': image_id, 'caption': caption})

	with open(args.results_json_path,'w') as f:
		json.dump(val_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_path', type=str, help='vocabulary pickle path', required=True)
    parser.add_argument('--image_root', type=str, default='val2014')
    parser.add_argument('--eval_ckpt_path', type=str, required=True)
    parser.add_argument('--glove_embed_path', type=str, default=None)
    parser.add_argument('--results_json_path', type=str, required=True)


    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    parser.add_argument('--rnn_type', type=str, default='lstm')
    parser.add_argument('--resnet_size', type=int, choices=[18, 34, 50, 101, 152], default=50)
    parser.add_argument('--caption_maxlen', type=int, default=15)


    args = parser.parse_args()
    print(args)
    main(args)







