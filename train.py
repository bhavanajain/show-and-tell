import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, Decoder
from utils import collate_fn
from vocab import construct_vocab
import pickle

def main(args):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Training on {device}")

	if not os.path.exists(args.models_dir):
		os.makedirs(args.model_path)

	if args.build_vocab:
		print(f"Constructing vocabulary from captions at {args.captions_json} and with count threshold={args.threshold}")
		vocab_object = construct_vocab(args.captions_json, args.threshold)
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

	encoder = Encoder((3, 224, 224), args.embed_size)
	decoder = Decoder(len(vocab_object), args.embed_size, args.hidden_size)

	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)

	total_examples = len(train_dataloader)
	for epoch in range(args.num_epochs):
		for 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_vocab', type=bool, default=True)
    parser.add_argument('--vocab_path', type=str, default=None, help='path for vocabulary')
    parser.add_argument('--image_root', type=str, default='train2014')
    parser.add_argument('--captions_json', type=str, default="annotations/captions_train2014.json")
    parser.add_argument('--threshold', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--models_dir', type=str, default='models_dir')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int, default=512)



   
    args = parser.parse_args()
    print(args)
    main(args)







