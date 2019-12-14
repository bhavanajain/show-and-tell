import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
	def __init__(self, resnet_size, image_shape, embed_size):
		super(Encoder, self).__init__()

		# supports multiple resnet models
		if resnet_size == 18:
			resnet = models.resnet18(pretrained=True)
			print('Using resnet18')
		elif resnet_size == 34:
			resnet = models.resnet34(pretrained=True)
			print('Using resnet34')
		elif resnet_size == 50:
			resnet = models.resnet50(pretrained=True)
			print('Using resnet50')
		elif resnet_size == 101:
			resnet = models.resnet101(pretrained=True)
			print('Using resnet101')
		elif resnet_size == 152:
			resnet = models.resnet152(pretrained=True)
			print('Using resnet152')

		else:
			print('Incorrect resnet size', resnet_size)

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
	def __init__(self, rnn_type, weights_matrix, vocab_size, embed_size, hidden_size):
		super(Decoder, self).__init__()
		
		self.embedding = nn.Embedding(vocab_size, embed_size)
		if weights_matrix is not None:
			self.embedding.load_state_dict({'weight': weights_matrix})
			self.embedding.weight.requires_grad = False

		# Support GRU or LSTM and give an option for setting numlayers and hidden unit size
		if rnn_type == 'gru':
			self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
		else:
			self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)

		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, image_embedding, sequence, lengths):
		seq_embedding = self.embedding(sequence)
		inputs_embedding = torch.cat((image_embedding.unsqueeze(1), seq_embedding), 1)
		packed_inputs = pack_padded_sequence(inputs_embedding, lengths, batch_first=True)
		hidden_states, last_hidden_state = self.rnn(packed_inputs)
		# hidden_states is packed input, extract data and feed into linear
		outputs = self.linear(hidden_states.data)

		return outputs

	def sample_single(self, image_embedding, caption_maxlen):
		caption_word_ids = []
		input_embedding = image_embedding.unsqueeze(1)
		for i in range(caption_maxlen):
			if i == 0:
				hidden, state = self.lstm(input_embedding)
			else:
				hidden, state = self.lstm(input_embedding, state)
			output = self.linear(hidden.squeeze(1))
			max_val, predicted_index = output.max(1)
			caption_word_ids.append(predicted_index)

			input_embedding = self.embedding(predicted_index).unsqueeze(1)

		return caption_word_ids







