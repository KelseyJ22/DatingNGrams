#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Classification
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import time
import logging
from datetime import datetime
import math
import itertools
import numpy as np
import tensorflow as tf
from model import Model

from util import print_sentence, Progbar, minibatches
from data_utils import load_and_preprocess_data, load_embeddings, read_ngrams, ModelHelper, get_chunks
from defs import LBLS
from model import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_loss = list()
dev_loss = list()

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def kl_divergence(P, Q, epsilon=0.01):
    P = P + epsilon
    Q = Q + epsilon
    cross_entropy = -tf.reduce_sum(tf.multiply(tf.log(Q), P), axis=1) #1, m
    entropy = -tf.reduce_sum(tf.multiply(tf.log(P), P), axis=1)
    kl_divergence = tf.reduce_mean(cross_entropy - entropy)
    return kl_divergence

def np_kl_divergence(P, Q, epsilon=0.01):
    P = P + epsilon
    Q = Q + epsilon
    cross_entropy = -np.sum(np.multiply(np.log(Q), P), axis=1)
    entropy = -np.sum(np.multiply(np.log(P), P), axis=1)
    kl_divergence = np.mean(cross_entropy - entropy)
    return kl_divergence

class Config:
    n_word_features = 2 # Number of features for every word in the input.
    n_gram_size = 5
    n_window_features = n_word_features * n_gram_size
    n_classes = len(LBLS) # Why is there a +1 here
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/one_hidden/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_book_predictions.conll"


class NGramModel(Model):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_window_features), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        return feed_dict


    def add_embedding(self):
        L = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        lookups = tf.nn.embedding_lookup(L, self.input_placeholder)
        embeddings = tf.reshape(lookups, [-1, self.config.n_window_features*self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("DATE"):
           
            W  = tf.get_variable("W", shape=(self.config.n_window_features*self.config.embed_size, self.config.hidden_size), initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape=(self.config.hidden_size,), initializer=tf.constant_initializer(value=0.0))
            U  = tf.get_variable("U", shape=(self.config.hidden_size, self.config.n_classes), initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape=(self.config.n_classes,), initializer=tf.constant_initializer(value=0.0))

            h = tf.nn.softmax(tf.matmul(x, W) + b1)
            h_drop = tf.nn.dropout(h, dropout_rate)
            pred = tf.nn.softmax(tf.matmul(h_drop, U) + b2)
        return pred


    def add_loss_op(self, pred):    
        loss = kl_divergence(self.labels_placeholder, pred)
        return loss


    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def preprocess_sequence_data(selsf, examples):
        data = []
        for example, label in examples:
            datum = []
            for word in example:
                datum += word
            data.append((datum, label))
        return data

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        i = 0
        for sentence, label in examples_raw:
            label_ = preds[i]
            i += 1
            ret.append([sentence, label, label_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def visualize_distributions(self, pred_labels, gold_labels, ngrams):
        if pred_labels.shape == gold_labels.shape: 
            for i in range(pred_labels.shape[0]):
                plt.figure()
                plt.title((' ').join(ngrams[i]))
                plt.plot(LBLS, pred_labels[i], 'r', label='predicted distribution')
                plt.plot(LBLS, gold_labels[i], 'b', label='gold_distribution')
                plt.xticks(np.arange(int(LBLS[0]), int(LBLS[len(LBLS)-1]) + 1, 20))
                plt.savefig(self.config.output_path + "fig" + str(i) + ".png")
                plt.clf()
        else:   
            print "Error visualizing distributions"

    def evaluate(self, sess, examples, examples_raw, last_epoch):
        avg_div = 0.0
        seen = 0
        for i, batch in enumerate(minibatches(examples, self.config.batch_size, shuffle=False)):
            
            pred_label = self.predict_on_batch(sess, batch[0])
            gold_label = batch[1]   
            if last_epoch and i == 0:   
                self.visualize_distributions(pred_label[:20], gold_label[:20], [example[0] for example in examples_raw[:20]])
            divergence = np_kl_divergence(gold_label, pred_label)
            avg_div += divergence
            seen += 1
        return avg_div/seen

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw, epoch, last_epoch):

        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        curr_loss = 0.
        num_encountered = 0
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            curr_loss += loss
            num_encountered += 1
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        train_loss.append(curr_loss/num_encountered)
        # print curr_loss/num_encountered
        # epochs.append(epoch+1)
        print("")

        logger.info("Evaluating on development data")
        divergence = self.evaluate(sess, dev_set, dev_set_raw, last_epoch)
        logger.info("KL- divergence: %.2f", divergence)
        dev_loss.append(divergence)

        return divergence



    def fit(self, sess, saver, train_examples_raw, dev_set_raw, train_examples, dev_examples):
        best_score = 0.

        # scores = list()
        epochs = list()
        last_epoch = False
        for epoch in range(self.config.n_epochs):
            if epoch == self.config.n_epochs - 1: 
                last_epoch = True
            epochs.append(epoch + 1) # start at epoch 1, not 0
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_examples, dev_examples, train_examples_raw, dev_set_raw, epoch, last_epoch)
            # scores.append(score)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()

        plt.plot(epochs, train_loss, label='KL Divergence for train')
        plt.ylabel("KL Divergence")
        plt.xlabel("Epoch")
        plt.title("Min Train KL-Divergence: " + "{0:.5f}".format(min(train_loss)))
        plt.legend()
        plt.savefig(self.config.output_path + "KL_train.png")
        plt.clf()
        plt.plot(epochs, dev_loss, label='KL Divergence for dev')
        plt.ylabel("KL Divergence")
        plt.xlabel("Epoch")
        plt.title("Min Dev KL-Divergence: " + "{0:.5f}".format(min(dev_loss)))
        plt.legend()
        plt.savefig(self.config.output_path + "KL_dev.png")
        plt.close('all')
        return best_score

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.helper = helper
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.report = None

        self.build()


def do_train(args):
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = NGramModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train_raw, dev_raw, train, dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an Date Classification model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="../data/train_normalized_overall.txt", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="../data/test_normalized_overall.txt", help="Dev data")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="../data/glove.6B.50d.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
