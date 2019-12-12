import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
	def __init__(self, image_shape, embed_size):
		super(Encoder, self).__init__()
		# Give option to support multiple resnet models
		resnet = models.resnet50(pretrained=True)
		self.features = nn.Sequential(*list(resnet.children())[:-1])

		with torch.no_grad():
			features = self.features(torch.zeros(*image_shape).unsqueeze(0))
			features_size = features.view(1, -1).shape[1]

		self.linear = nn.Linear(features_size, embed_size)

	def forward(self, image):
		with torch.no_grad():
			out = self.features(image)
		out = out.view(out.shape[0], -1)
		out = self.linear(out)
		return out

class Decoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(Decoder, self).__init__()
		# Add an option to import golve embeddings
		self.embedding = nn.Embedding(vocab_size, embed_size)

		# Support GRU or LSTM and give an option for setting numlayers and hidden unit size
		self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, image_embedding, sequence, lengths):
		seq_embedding = self.embedding(sequence)
		inputs_embedding = torch.cat((image_embedding.unsqueeze(1), seq_embedding), 1)
		packed_inputs = pack_padded_sequence(inputs_embedding, lengths, batch_first=True)
		hidden_states, last_hidden_state = self.lstm(packed_inputs)
		# hidden_states is packed input, extract data and feed into linear
		outputs = self.linear(hidden_states.data)

		return outputs