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

def torch_tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)