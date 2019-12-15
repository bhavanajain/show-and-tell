import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from utils import torch_tile

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

		self.rnn_type = rnn_type

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

	def sample_batch(self, image_embeddings, caption_maxlen):
		caption_word_ids = []
		input_embeddings = image_embeddings.unsqueeze(1)
		for i in range(caption_maxlen):
			if i == 0:
				hiddens, states = self.rnn(input_embeddings)
			else:
				hiddens, states = self.rnn(input_embeddings, states)

			outputs = self.linear(hiddens.squeeze(1))
			_, predicted = outputs.max(1)
			caption_word_ids.append(predicted)

			input_embeddings = self.embedding(predicted)
			input_embeddings = input_embeddings.unsqueeze(1)

		caption_word_ids = torch.stack(caption_word_ids, 1)
		return caption_word_ids

	def sample_beam(self, features, states=None, beam_size=1, device=None, vocab=None):
		"""Generate captions for given image features using beam search."""
		sampled_ids = []
		features = features.unsqueeze(1)
		inputs = features.repeat(beam_size,1,1)        
		k_sampled_ids = [[] for i in range(beam_size)] #k full sentences
		
		probs = torch.ones(beam_size, 1).to(device) #batch_size*1
		batch_size = beam_size #to begin with, then as a sentence ends, batch_size will decrease by 1
		finished_k_sample_ids = []
		for i in range(self.max_seg_length):
			hiddens, states = self.rnn(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
			outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
			outputs = torch.nn.functional.softmax(outputs, dim=1)
			sentence_probs = torch.mul(outputs, probs) #(batch_size, vocab_size)
			if i==1:
				res, ind = sentence_probs[0].view(-1).topk(beam_size)   
			else:
				res, ind = sentence_probs.view(-1).topk(beam_size)

			new_indices = ind/outputs.shape[1]
			new_words = ind%outputs.shape[1] #batch_size*1
			extract_indices = []
			new_k_sampled_ids = []
			for j in range(batch_size):
				extended_sentence = k_sampled_ids[new_indices[j].item()] + [new_words[j]]
				if vocab.index2word[new_words[j].item()]=='<end>':
					finished_k_sample_ids.append((extended_sentence, sentence_probs[new_indices[j]][new_words[j]]))
					finished_k_sample_ids.sort(key=lambda x:-x[1])
					batch_size -= 1
					if batch_size == 0:
						return [finished_k_sample_ids[0][0]], finished_k_sample_ids[0][1]
				else:
					new_k_sampled_ids.append(extended_sentence)
					extract_indices.append(j)

			k_sampled_ids = [x for x in new_k_sampled_ids]

			new_indices = new_indices[extract_indices]
			res = res[extract_indices]

			if self.rnn_type=='lstm':
				states= (states[0][0][new_indices].unsqueeze(0), states[1][0][new_indices].unsqueeze(0))
			elif self.rnn_type=='gru':
				states = states[0][new_indices].unsqueeze(0)
			probs = res.unsqueeze(1)

			inputs = self.embed(new_words[extract_indices]) #batch_size*1
			inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)

		if len(finished_k_sample_ids)==0:
			return [k_sampled_ids[0]], _

		return [finished_k_sample_ids[0][0]], finished_k_sample_ids[0][1]









