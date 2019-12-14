from pycocotools.coco import COCO
import torch.utils.data as data
import nltk
from PIL import Image
import torch
import os

class cocoDataset(data.Dataset):
	def __init__(self, root, json, vocab_object, transforms=None):
		self.root = root
		self.coco_object = COCO(json)
		self.example_ids = list(self.coco_object.anns.keys())
		self.vocab = vocab_object
		self.transforms = transforms

	def __getitem__(self, index):
		curr_index = self.example_ids[index]
		caption = self.coco_object.anns[curr_index]['caption']
		img_id = self.coco_object.anns[curr_index]['image_id']

		img_path = self.coco_object.loadImgs(img_id)[0]['file_name']
		image = Image.open(os.path.join(self.root, img_path)).convert('RGB')

		if self.transforms is not None:
			image = self.transforms(image)

		caption_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(self.vocab('<start>'))
		caption.extend([self.vocab(token) for token in caption_tokens])
		caption.append(self.vocab('<end>'))

		target = torch.Tensor(caption)

		return image, target

	def __len__(self):
		return len(self.example_ids)

class ImageDataset(data.Dataset):
	def __init__(self, root, transforms=None):
		self.root = root
		self.image_files = sorted(os.listdir(root))

	def __getitem__(self, index):
		curr_image_file = self.image_files[index]

		image_object = Image.open(os.path.join(self.root, curr_image_file)).convert('RGB')

		# extract image id from the image
		image_id = int(curr_image_file.split('.')[0].split('_')[-1])
		if self.transforms is not None:
			image_object = self.transforms(image_object)

		return image_object, image_id, curr_image_file

	def __len__(self):
		return len(self.image_files)







