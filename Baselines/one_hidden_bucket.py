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
from utils.model import Model

from utils.util import print_sentence, write_conll, ConfusionMatrix, Progbar, minibatches, write_ngrams
from utils.data_util import load_and_preprocess_data, load_embeddings, read_ngrams, ModelHelper, get_chunks
from utils.defs import LBLS
from utils.model import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

losses = list()
epochs = list()

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        return feed_dict


    def add_embedding(self):
        L = tf.Variable(self.pretrained_embeddings)
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
            pred = tf.matmul(h_drop, U) + b2

        return pred


    def add_loss_op(self, pred):      
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss_vector)

        return loss


    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op


    def preprocess_sequence_data(self, examples):
        data = []
        for example, label in examples:
            datum = []
            for word in example:
                datum += word
            data.append((datum, label))
        return data


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
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def evaluate(self, sess, examples, examples_raw):
        token_cm = ConfusionMatrix(labels=LBLS)
        # Double check how p, r are calculated
        correct_preds, total_correct, total_preds = 0., 0., 0.
        error = 0.0
        for _, label, label_  in self.output(sess, examples_raw, examples):
            token_cm.update(label, label_)
            error = error + ((label - label_)**2)
            correct_preds +=  (label == label_)
        #     correct_preds += label == label_
            total_preds += 1
        #     # total_correct += label != label_

        # p = correct_preds / total_preds if total_preds > 0 else 0
        # r = correct_preds / total_preds if total_preds > 0 else 0
        # f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (error/total_preds, correct_preds/total_preds)


    def outputConfusionMatrix(self, cm):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
        plt.colorbar()
        classes = LBLS
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        flat_cm = [item for sublist in cm for item in sublist]
        thresh = max(flat_cm) / 2.
        for i, j in itertools.product(range(len(cm)), range(len(cm[0]))):
            plt.text(j, i, cm[i][j],
                     horizontalalignment="center",
                     color="white" if cm[i][j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.config.output_path + "ConfusionMatrix.png")
        plt.clf()


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw, epoch, last_epoch):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        curr_loss = 0.
        num_encountered = 0
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            curr_loss += loss
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
            num_encountered +=1
        # print curr_loss/num_encountered
        losses.append(curr_loss/num_encountered)
        epochs.append(epoch+1)
        print("")

        logger.info("Evaluating on development data")
        token_cm, metrics = self.evaluate(sess, dev_set, dev_set_raw)
        if last_epoch: 
            self.outputConfusionMatrix(token_cm.as_data())
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Accuracy: %.2f", metrics[1])
        logger.info("Error: %.2f", metrics[0])

        return metrics[0], metrics[1]


    def output(self, sess, inputs_raw, inputs=None):
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)


    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.
        errors = list()
        accuracies = list()
        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)
        last_epoch = False
        for epoch in range(self.config.n_epochs):
            if epoch == self.config.n_epochs - 1: 
                last_epoch = True
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            error, score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw, epoch, last_epoch)
            errors.append(error)
            accuracies.append(score)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()

        plt.plot(epochs, losses, label='train_loss')
        plt.legend()
        plt.title("Min train loss: " + "{0:.2f}".format(min(losses)))
        plt.savefig(self.config.output_path + "train_loss.png")
        plt.clf()

        plt.plot(epochs, errors, label='squared_error')
        plt.legend()
        plt.title("Min squared distance error: " + "{0:.2f}".format(min(errors)))
        plt.savefig(self.config.output_path + "squared_error.png")
        plt.clf()

        plt.plot(epochs, accuracies, label='accuracies')
        plt.legend()
        plt.title("Max accuracy: " + "{0:.2f}".format(max(accuracies)))
        plt.savefig(self.config.output_path + "accuracies.png")
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
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [preds for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_ngrams(f, output)


def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_ngrams(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = NGramModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an Date Classification model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="../data/train_normalized_overall.txt", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="../data/train_normalized_overall.txt", help="Dev data")

    command_parser.add_argument('-v', '--glove', type=argparse.FileType('r'), default="../data/glove.6B.50d.txt", help="Path to glove_vectore file")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
