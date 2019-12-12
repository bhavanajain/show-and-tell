from pycocotools.coco import COCO
from collections import Counter
import pickle
import nltk

class Vocabulary:
	def __init__(self):
		self.word2index = {}
		self.index2word = {}
		self.curr_index = 0

	def add_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.curr_index
			self.index2word[self.curr_index] = word
			self.curr_index += 1

	def __call__(self, word):
		if word not in self.word2index:
			return self.word2index['<unk>']
		return self.word2index[word]

	def __len__(self):
		return len(self.word2index)

def build_vocab(captions_json, threshold):
	coco = COCO(captions_json)
	word_counts = Counter()
	for i, sample_id in enumerate(coco.anns.keys()):
		caption = str(coco.anns[sample_id]['caption'])
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		word_counts.update(tokens)

	filtered_words = [word for word in word_counts if word_counts[word] >= threshold]

	vocab = Vocabulary()
	# note: pad should be at index 0, since we pad with zeros
	helper_words = ['<pad>', '<start>', '<end>', '<unk>']
	for word in helper_words:
		vocab.add_word(word)

	for word in filtered_words:
		vocab.add_word(word)

	return vocab










