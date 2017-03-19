#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import math
import argparse
import sys
import time
import logging
import itertools
import numpy as np
from datetime import datetime

import tensorflow as tf

from utils.model import Model
from utils.util import print_sentence, ConfusionMatrix, Progbar, minibatches, generate_minibatches
from utils.data_util import load_and_preprocess_data, load_glove_vectors, read_ngrams, ModelHelper, load_and_preprocess_test
from utils.defs import LBLS
from utils.initialization import xavier_weight_init

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast


losses = [float('inf')]
cm_latest = None
dev_losses = list()


logger = logging.getLogger('final_project')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def read_book_ngrams(filename):
    for line in open(filename, 'r'):
        return ast.literal_eval(line)


class Config:
    n_word_features = 2 # Number of features for every word in the input.
    n_gram_size = 5
    n_window_features = n_word_features * n_gram_size # Number of features for every word in the input.
    n_classes = len(LBLS)
    dropout = 0.5
    embed_size = 50
    batch_size = 2048
    n_epochs = 10
    epoch_delta = 0.001
    max_grad_norm = 10.
    lr = 0.001
    hidden_size_1 = 300
    hidden_size_2 = 400
    reg = 0

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
        xavier_initializer = xavier_weight_init()

        W = tf.Variable(xavier_initializer([self.config.n_window_features*self.config.embed_size, self.config.hidden_size_1]))
        b1 = tf.Variable(tf.zeros([self.config.hidden_size_1], dtype=tf.float32))
        U = tf.Variable(xavier_initializer([self.config.hidden_size_2, self.config.n_classes]))
        W2 = tf.Variable(xavier_initializer([self.config.hidden_size_1, self.config.hidden_size_2]))
        b2 = tf.Variable(tf.zeros([self.config.hidden_size_2], dtype=tf.float32))
        b3 = tf.Variable(tf.zeros([self.config.n_classes], dtype=tf.float32))


        layer_1 = tf.add(tf.matmul(x, W), b1)
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
        layer_2 = tf.nn.sigmoid(layer_2)
        h_drop = tf.nn.dropout(layer_2, self.dropout_placeholder)
        pred = tf.matmul(h_drop, U) + b3

        self.regularization = self.config.reg*tf.nn.l2_loss(U) + self.config.reg*tf.nn.l2_loss(W) + self.config.reg*tf.nn.l2_loss(W2)

        return pred


    def add_loss_op(self, pred):
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss_vector) + self.regularization
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
        for i in range(0, len(preds)):
            sentence, label = examples_raw[i]
            label_ = preds[i]
            ret.append([sentence, label, label_])
        return ret


    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def predict_on_book_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)

        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def evaluate(self, sess, examples, examples_raw):
        token_cm = ConfusionMatrix(labels=LBLS)
        correct_preds, total_correct, total_preds = 0., 0., 0.
        error = 0.0
        for _, label, label_  in self.output(sess, examples_raw, examples):
            token_cm.update(label, label_)
            error = error + ((label - label_)**2)

            correct_preds +=  (label == label_)
            total_preds += 1

        return token_cm, (error/total_preds, correct_preds/total_preds)


    def output(self, sess, inputs_raw, inputs=None):
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        minibatches = generate_minibatches(inputs, self.config.batch_size, 0.03)
        for i, batch in enumerate(minibatches):
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
           
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)


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


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        global losses
        global cm_latest
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        curr_loss = 0.
        num_encountered = 0

        minibatches = generate_minibatches(train_examples, self.config.batch_size, 0.03)        
        for i, batch in enumerate(minibatches):
            bucket_dist = [0] * self.config.n_classes
            if i < 5: 
                for bucket in batch[1]: 
                    bucket_dist[bucket] += 1
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            curr_loss += loss
            num_encountered += 1
            if self.report: self.report.log_train_loss(loss)
            if i < 5: 
                print ""
                print "Bucket distribution: " + str(bucket_dist) 

        losses.append(curr_loss/num_encountered)

        print("")

        logger.info("Evaluating on development data")
        token_cm, metrics = self.evaluate(sess, dev_set, dev_set_raw)
        cm_latest = token_cm

        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level errors:\n" + token_cm.summary())
        logger.info("Accuracy: %.2f", metrics[1])
        logger.info("Error: %.2f", metrics[0])

        return metrics[0], metrics[1]


    def fit(self, sess, helper, saver, train_examples_raw, dev_set_raw):
        global losses
        global cm_latest
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)
        epoch = 0
        epochs = list()
        errors = list()
        accuracies = list()
        while len(losses) <= 2 or (losses[-1] - losses[-2] > self.config.epoch_delta):
            if len(epochs) > 10: # update as desired
                break
            epochs.append(epoch+1)
            logger.info("Epoch %d", epoch + 1)
            error, score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
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
            epoch += 1

        self.print_results(sess, helper, epochs, losses, cm_latest, errors, accuracies)
        return best_score


    def predict_book_age(self, sess, ngrams, year):
        ngrams_vectoriezed = self.helper.vectorize_ngrams(ngrams)
        predictions = self.predict_on_book_batch(sess, ngrams_vectoriezed)

        cumulative = np.sum(predictions, axis=0)
        for i in xrange(len(predictions)):
            prediction = predictions[i]
            if max(prediction) > 0.50:
                print ngrams[i]
                print year
                print max(prediction) - prediction[5]
                print prediction
                print "\n"
        counts = np.bincount(preds)
        return np.argmax(cumulative)


    def print_results(self, sess, helper, epochs, losses, cm_latest, errors, accuracies):
        logger.info("Done training! Evaluating on test data")
        test_set, test_set_raw = load_and_preprocess_test(helper)
        test = self.preprocess_sequence_data(test_set)
        token_cm, metrics = self.evaluate(sess, test, test_set)
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level errors:\n" + token_cm.summary())
        logger.info("Accuracy: %.2f", metrics[1])
        logger.info("Error: %.2f", metrics[0])

        print 'generating plot with ' + str(epochs) + ' ' + str(losses[1:])
        plt.plot(epochs, losses[1:], label='train_loss')
        plt.legend()
        plt.title("Min train loss: " + "{0:.2f}".format(min(losses)))
        plt.xlim(xmin=1)

        plt.savefig(self.config.output_path + "train_loss.png")
        plt.clf()

        print 'generating plot with ' + str(epochs) + ' ' + str(errors)
        plt.plot(epochs, errors, label='squared_error')
        plt.legend()
        plt.title("Min squared distance error: " + "{0:.2f}".format(min(errors)))
        plt.xlim(xmin=1)

        plt.savefig(self.config.output_path + "squared_error.png")
        plt.clf()

        print 'generating plot with ' + str(epochs) + ' ' + str(accuracies)
        plt.plot(epochs, accuracies, label='accuracies')
        plt.legend()
        plt.title("Max accuracy: " + "{0:.2f}".format(max(accuracies)))
        plt.savefig(self.config.output_path + "accuracies.png")
        plt.close('all')


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
            model.fit(session, helper, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [preds for preds in predictions]
                output = zip(sentences, labels, predictions)


def book_evaluate(model_path):
    config = Config(model_path)
    helper = ModelHelper.load(model_path)
    embeddings = load_glove_vectors("../data/glove.6B.50d.txt", helper)
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
            for i in range(1, 8):
                input_data = read_book_ngrams('../data/' + str(i) +'_5grams.txt')
                result = model.predict_book_age(session, input_data[0], input_data[1])
                print "Predicted", LBLS[result], "Actual", input_data[1]


# book_evaluate(model_path="results/feedforward/latest/")
classify()