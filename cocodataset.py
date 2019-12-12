from pycocotools.coco import COCO

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

		img_path = coco.loadImgs(img_id)[0]['file_name']
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







