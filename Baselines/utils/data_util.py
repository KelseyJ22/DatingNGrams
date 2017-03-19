#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import logging
from collections import Counter

import numpy as np
from util import read_ngrams, one_hot, window_iterator, ConfusionMatrix, load_word_vector_mapping
from defs import LBLS, NONE, NUM, UNK, EMBED_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"

def casing(word):
    if len(word) == 0: return word
    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def normalize(word):
    if word.isdigit(): return NUM
    else: return word.lower()


def featurize(embeddings, word):
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))


def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()


class ModelHelper(object):
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        self.max_length = max_length

    def lookup_embeddings(self, inputs):
        all_sentences = list()
        for example in inputs:
            ngram = inputs[0]
            sentence = list()
            for word in ngram:
                sentence.append(self.tok2id.get(normalize(word), self.tok2id[UNK]))
            all_sentences.append(sentence)
        return all_sentences


    def vectorize_example(self, sentence, label):
        sentence_ = []
        for word in sentence:
            sentence_.append([self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]])
            #sentence_.append([self.tok2id.get(normalize(word), self.tok2id[UNK])])       
        label_ = LBLS.index(label)
        return sentence_, label_


    def vectorize(self, data):
        return [self.vectorize_example(sentence, label) for sentence, label in data]


    def vectorize_ngram(self, ngram):
        vector = []
        for word in ngram:
            vector += [self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]]
        return vector


    def vectorize_ngrams(self, ngrams):
        return [self.vectorize_ngram(ngram) for ngram in ngrams]


    @classmethod
    def build(cls, data):
        tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=100000)
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        max_length = max(len(sentence) for sentence, _ in data)

        return cls(tok2id, max_length)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path):
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)



def load_and_preprocess_data():
    logger.info("Loading training data...")
    train = read_ngrams('../data/train_for_hyperparameters.txt')
    logger.info("Done. Read %d training sentences", len(train))
    logger.info("Loading dev data...")
    dev = read_ngrams('../data/test_for_hyperparameters.txt')
    logger.info("Done. Read %d dev sentences", len(dev))

    helper = ModelHelper.build(train)

    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    return helper, train_data, dev_data, train, dev


def load_and_preprocess_test(helper):
    logger.info("Loading test data...")
    test = read_ngrams('../data/test_for_hyperparameters.txt')
    logger.info("Done. Read %d sentences", len(test))
    test_data = helper.vectorize(test)
    return test_data, test


def load_glove_vectors(file, helper, dimensions=50):
    wordVectors = np.array(np.random.randn(len(helper.tok2id) + 1, dimensions), dtype=np.float32)
    f = open(file, 'r')
    for line in f:
        if not line:
            continue
        row = line.split()
        token = row[0]
        if token not in helper.tok2id:
            continue
        data = [float(x) for x in row[1:]]
        if len(data) != EMBED_SIZE:
            raise RuntimeError("wrong number of dimensions")
        wordVectors[helper.tok2id[token]] = np.asarray(data) #, dtype=np.float32)
    return wordVectors


def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}