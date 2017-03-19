from utils.defs import LBLS
import numpy as np

import ast 

def load_dataset():
	f = open('../Data/sklearn_train.txt')
	x1 = list()
	y1 = list()

	for line in f:
		line = line.strip()
		if len(line) == 0 or line.startswith("-DOCSTART-") or '\t' not in line:
			continue
		else:
			sentence, labels = line.split('\t')

			sentence = ast.literal_eval(sentence)
			new_labels = ast.literal_eval(labels)
			if '' in sentence or len(sentence) != 5:
				continue
			bucket = new_labels.index(max(new_labels))
			x1.append((' ').join(sentence))
			y1.append(int(LBLS[bucket]))

		x1.append('This is old english .')
		y1.append(1800)
	f.close()

	f = open('../Data/sklearn_test.txt')
	x2 = list()
	y2 = list()


	for line in f:
		line = line.strip()
		if len(line) == 0 or line.startswith("-DOCSTART-") or '\t' not in line:
			continue
		else:
			sentence, labels = line.split('\t')

			sentence = ast.literal_eval(sentence)
			new_labels = ast.literal_eval(labels)
			if '' in sentence or len(sentence) != 5:
				continue
			x2.append((' ').join(sentence))
			y2.append(new_labels)
			# d2.append((sentence, new_labels))
	f.close()

	return ((x1, y1), (x2, y2))

def np_kl_divergence(P, Q, epsilon=0.01):
    P = P + epsilon
    Q = Q + epsilon
    cross_entropy = -np.sum(np.multiply(np.log(Q), P), axis=1)
    entropy = -np.sum(np.multiply(np.log(P), P), axis=1)
    kl_divergence = np.mean(cross_entropy - entropy)
    return kl_divergence


# load_dataset()
