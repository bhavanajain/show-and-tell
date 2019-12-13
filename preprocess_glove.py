import numpy as np
import pickle
import argparse
import io

def main(args):
	with open(args.vocab_path, 'rb') as f:
		vocab_object = pickle.load(f)
	print(f"Loaded the vocabulary object from {args.vocab_path}, total size={len(vocab_object)}")

	glove_embeddings = {}
	with io.open(args.glove_filepath, 'r', encoding='utf-8') as f:
	    lines = f.readlines()

	count = 0
	for line in lines:
		line = line.strip().split(' ')
		word = line[0]
		if word in vocab_object.word2index:
			embedding = np.asarray(line[1:],dtype=np.float)
			import pdb; pdb.set_trace();
			glove_embeddings[word] = embedding
			count += 1

	print(f"Total vocab size = {len(vocab_object)}, found glove embeddings for {count}")

	with open(args.glove_embed_path, 'wb') as glove_f:
		pickle.dump(glove_embeddings, glove_f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Name models_dir appropriately across experiments lest models will get overwritten
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--glove_filepath', type=str, required=True)
    parser.add_argument('--glove_embed_path', type=str, required=True)

    args = parser.parse_args()
    print(args)
    main(args)



