import torch

def collate_fn(data):
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)

	images = torch.stack(images, 0)

	lengths = [len(caption) for caption in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, caption in enumerate(captions):
		curr_end = lengths[i]
		targets[i, :curr_end] = caption[:curr_end]

	return images, targets, lengths