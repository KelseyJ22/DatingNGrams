#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import sys
import time
import numpy as np
import logging
from datetime import datetime

import tensorflow as tf

from utils.model import Model
from utils.util import print_sentence, Progbar, minibatches
from utils.data_utils import load_and_preprocess_data, load_glove_vectors, read_ngrams, ModelHelper, load_and_preprocess_test
from utils.defs import LBLS
from utils.initialization import xavier_weight_init

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_loss = [10]
pred_latest = None
gold_latest = None
example_latest = None
dev_loss = list()

logger = logging.getLogger('final_project')
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
    n_features = n_word_features * n_gram_size # Number of features for every word in the input.
    n_classes = len(LBLS)
    dropout = 0.4
    embed_size = 50
    batch_size = 200
    n_epochs = 10
    epoch_delta = 0.001
    max_grad_norm = 10.
    lr = 0.01
    hidden_size_1 = 500
    hidden_size_2 = 500
    reg = 0
    rnn_hidden_size = 200

    def __init__(self, output_path=None):
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = "results/two_hidden/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"


class NGramModel(Model):

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_features), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0.5):
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch

        return feed_dict


    def add_embedding(self):
        L = tf.Variable(self.pretrained_embeddings)
        lookups = tf.nn.embedding_lookup(L, self.input_placeholder)
        return lookups


    def add_prediction_op(self):
        x = self.add_embedding()

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.rnn_hidden_size)
        outputs, state = tf.nn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
        x = outputs[:,-1,:]

        xavier_initializer = xavier_weight_init()

        W1 = tf.Variable(xavier_initializer([self.config.rnn_hidden_size, self.config.hidden_size_1]))
        b1 = tf.Variable(tf.zeros([self.config.hidden_size_1], dtype=tf.float32))
        W2 = tf.Variable(xavier_initializer([self.config.hidden_size_1, self.config.hidden_size_2]))
        b2 = tf.Variable(tf.zeros([self.config.hidden_size_2], dtype=tf.float32))
        W3 = tf.Variable(xavier_initializer([self.config.hidden_size_2, self.config.n_classes]))
        b3 = tf.Variable(tf.zeros([self.config.n_classes], dtype=tf.float32))

        layer_1 = tf.add(tf.matmul(x, W1), b1)
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
        layer_2 = tf.nn.relu(layer_2)
        distrib = tf.add(tf.matmul(layer_2, W3), b3) # not actually the probability distrib yet because using softmax_cross_entropy
        pred    = tf.nn.softmax(distrib)

        self.regularization = self.config.reg*tf.nn.l2_loss(W1) + self.config.reg*tf.nn.l2_loss(W2) + self.config.reg*tf.nn.l2_loss(W3)

        return pred


    def add_loss_op(self, pred):
        loss = kl_divergence(self.labels_placeholder, pred) + self.regularization
        return loss


    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op


    def consolidate_predictions(self, examples_raw, examples, preds):
        ret = []
        i = 0
        for sentence, label in examples_raw:
            label_ = preds[i]
            i += 1
            ret.append([sentence, label, label_])
        return ret


    def predict_on_batch(self, sess, inputs_batch):
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
                plt.plot(LBLS, gold_labels[i], 'b', label='gold distribution')
                plt.xticks(np.arange(int(LBLS[0]), int(LBLS[len(LBLS)-1]) + 1, 20))
                plt.savefig(self.config.output_path + "fig" + str(i) + ".png")
                plt.clf()
        else:   
            print "Error visualizing distributions"


    def evaluate(self, sess, examples, examples_raw):
        global pred_latest
        global gold_latest 
        global example_latest
        avg_div = 0.0
        seen = 0
        
        for i, batch in enumerate(minibatches(examples, self.config.batch_size, shuffle=False)):
            pred_label = self.predict_on_batch(sess, batch[0])
            gold_label = batch[1]
            divergence = np_kl_divergence(gold_label, pred_label)
            
            pred_latest = pred_label[:40]
            gold_latest = gold_label[:40]
            example_latest = [example[0] for example in examples_raw[:40]]

            avg_div += divergence
            seen += 1

        return avg_div/seen


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw, epoch):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        curr_loss = 0.
        num_encountered = 0
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            curr_loss += loss
            num_encountered += 1
            if self.report: self.report.log_train_loss(loss)
        train_loss.append(curr_loss/num_encountered)
        
        print("")

        logger.info("Evaluating on development data")
        divergence = self.evaluate(sess, dev_set, dev_set_raw)
        logger.info("KL divergence: %.2f", divergence)

        dev_loss.append(divergence)
        return divergence


    def fit(self, sess, saver, train_examples_raw, dev_set_raw, train_examples, dev_examples, helper):
        global cm_latest
        global train_loss
        best_score = 0.
        epoch = 0
        epochs = list()
        while len(dev_loss) <= 1 or (dev_loss[-2] - dev_loss[-1] > self.config.epoch_delta):
            epochs.append(epoch + 1) # start at epoch 1, not 0
            logger.info("Epoch %d", epoch + 1)
            score = self.run_epoch(sess, train_examples, dev_examples, train_examples_raw, dev_set_raw, epoch)
            if score < best_score: # we want smallest, not largest
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
            epoch += 1

        test_set, test_set_raw = load_and_preprocess_test(helper)
        logger.info("Done training: evaluating on test data")
        divergence = self.evaluate(sess, test_set, test_set_raw)
        logger.info("KL divergence: %.2f", divergence)

        self.visualize_distributions(pred_latest, gold_latest, example_latest) # will be values for test data
        plt.plot(epochs, train_loss[1:], label='KL Divergence for train')
        plt.ylabel("KL Divergence")
        plt.xlabel("Epoch")
        plt.title("Min Train KL Divergence: " + "{0:.5f}".format(min(train_loss)))

        plt.xlim(xmin=1)        
        plt.legend()
        plt.savefig(self.config.output_path + "KL_train.png")
        plt.clf()
        plt.plot(epochs, dev_loss, label='KL Divergence for dev')
        plt.ylabel("KL Divergence")
        plt.xlabel("Epoch")
        plt.title("Min Dev KL Divergence: " + "{0:.5f}".format(min(dev_loss)))  

        plt.xlim(xmin=1)      
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


def classify():
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data()
    embeddings = load_glove_vectors('../data/glove.6B.50d.txt', helper)
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
            model.fit(session, saver, train_raw, dev_raw, train, dev, helper)


classify()