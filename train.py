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
from dataset import cocoDataset
import numpy as np

def main(args):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Training on {device}")

	if not os.path.exists(args.models_dir):
		os.makedirs(args.models_dir)		

	if args.build_vocab:
		print(f"Building vocabulary from captions at {args.captions_json} and with count threshold={args.threshold}")
		vocab_object = build_vocab(args.captions_json, args.threshold)
		with open(args.vocab_path, "wb") as vocab_f:
			pickle.dump(vocab_object, vocab_f)
		print(f"Saved the vocabulary object to {args.vocab_path}, total size={len(vocab_object)}")
	else:
		with open(args.vocab_path, 'rb') as f:
			vocab_object = pickle.load(f)
		print(f"Loaded the vocabulary object from {args.vocab_path}, total size={len(vocab_object)}")

	img_transforms = transforms.Compose([
		                transforms.Resize((256, 256)),
		                transforms.RandomCrop((224, 224)),
		                transforms.ToTensor(),
		                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	                ])

	train_dataset = cocoDataset(args.image_root, args.captions_json, vocab_object, img_transforms)
	train_dataloader = torch.utils.data.DataLoader(
		dataset=train_dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=args.num_workers, 
		collate_fn=collate_fn)

	encoder = Encoder((3, 224, 224), args.embed_size).to(device)
	decoder = Decoder(len(vocab_object), args.embed_size, args.hidden_size).to(device)

	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)

	start_epoch = 0

	if args.ckpt_path is not None:
		model_ckpt = torch.load(args.ckpt_path)
		start_epoch = model_ckpt['epoch'] + 1
		prev_loss = model_ckpt['loss']
		encoder.load_state_dict(model_ckpt['encoder'])
		decoder.load_state_dict(model_ckpt['decoder'])
		optimizer.load_state_dict(model_ckpt['optimizer'])
		print(f"Loaded model and optimizer state from {args.ckpt_path}; start epoch at {start_epoch}; prev loss={prev_loss}")

	total_examples = len(train_dataloader)
	for epoch in range(start_epoch, args.num_epochs):
		for i, (images, captions, lengths) in enumerate(train_dataloader):
			images = images.to(device)
			captions = captions.to(device)
			targets = pack_padded_sequence(captions, lengths, batch_first=True).data

			image_embeddings = encoder(images)
			outputs = decoder(image_embeddings, captions, lengths)

			loss = criterion(outputs, targets)

			decoder.zero_grad()
			encoder.zero_grad()

			loss.backward()
			optimizer.step()

			if i % args.log_interval == 0:
				loss_val = "{:.4f}".format(loss.item())
				perplexity_val = "{:5.4f}".format(np.exp(loss.item()))
				print(f"epoch=[{epoch}/{args.num_epochs}], iteration=[{i}/{total_examples}], loss={loss_val}, perplexity={perplexity_val}")

		torch.save({
			'epoch': epoch,
			'encoder': encoder.state_dict(),
			'decoder': decoder.state_dict(),
			'optimizer': optimizer.state_dict(),
			'loss': loss
		}, os.path.join(args.models_dir, 'model-after-epoch-{}.ckpt'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Name models_dir appropriately across experiments lest models will get overwritten
    parser.add_argument('--models_dir', type=str, required=True)

    parser.add_argument('--build_vocab', default=False, action="store_true")
    parser.add_argument('--vocab_path', type=str, help='vocabulary pickle path', required=True)
    parser.add_argument('--image_root', type=str, default='train2014')
    parser.add_argument('--captions_json', type=str, default="annotations/captions_train2014.json")
    parser.add_argument('--threshold', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--ckpt_path', type=str, default=None)

    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--log_interval', type=int, default=500)

    parser.add_argument('--learning_rate', type=float, default=5e-4)

    args = parser.parse_args()
    print(args)
    main(args)







