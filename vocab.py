from pycocotools.coco import COCO
from collections import Counter
import pickle

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

def construct_vocab(captions_json, threshold):
	coco = COCO(captions_json)
	word_counts = Counter()
	for i, sample_id in enumerate(coco.anns.keys()):
		caption = str(coco.anns[sample_id]['caption'])
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		counter.update(tokens)

	filtered_words = [word for word in counter if counter[word] >= threshold]

	vocab = Vocabulary()
	# note: pad should be at index 0, since we pad with zeros
	helper_words = ['<pad>', '<start>', '<end>', '<unk>']
	for word in helper_words:
		vocab.add_word(word)

	for word in filtered_words:
		vocab.add_word(word)

	return vocab

captions_json = "annotations/captions_train2014.json"
threshold = 5
vocab_path = "vocab_train2014.json"

print(f"Constructing vocabulary from captions at {captions_json} and with count threshold={threshold}")

vocab_object = construct_vocab(captions_json, threshold)
with open(vocab_path, "wb") as vocab_f:
	pickle.dump(vocab_object, vocab_f)

print(f"Saved the vocabulary object to {vocab_path}, total size={len(vocab_object)}")










