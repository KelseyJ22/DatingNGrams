#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import ast
import time
import logging
import itertools
import numpy as np
from datetime import datetime

import tensorflow as tf
from utils.model import Model

from utils.util import print_sentence, ConfusionMatrix, Progbar, minibatches
from utils.data_util import load_and_preprocess_data, load_glove_vectors, read_ngrams, ModelHelper
from utils.defs import LBLS
from utils.model import Model
from utils.initialization import xavier_weight_init
import matplotlib
import matplotlib.pyplot as plt

losses = list()

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def read_book_ngrams(filename):
    for line in open(filename, 'r'):
        return ast.literal_eval(line)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """
    n_word_features = 2 # Number of features for every word in the input.
    n_gram_size = 5
    n_window_features = n_word_features * n_gram_size # Number of features for every word in the input.
    n_classes = len(LBLS)
    dropout = 0.5
    embed_size = 50
    batch_size = 2048
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001
    hidden_size_1 = 200
    hidden_size_2 = 200
    regularization = 0

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/feedforward/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "predictions.conll"


class NGramModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_window_features), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch

        return feed_dict


    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
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

        # self.regularization = self.config.lr*tf.nn.l2_loss(U) + self.config.lr*tf.nn.l2_loss(W)

        return pred


    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """

        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss_vector) #+ self.regularization)
        return loss


    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def preprocess_sequence_data(self, examples):
        """Flattening word vectors for word to represent
            a larger, flatter word vector for n-gram.
        """
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
        #pdb.set_trace()
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
        predictions = sess.run(tf.nn.softmax(self.pred), feed_dict=feed)
        print predictions.shape
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=LBLS)
        correct_preds, total_correct, total_preds = 0., 0., 0.
        error = 0.0
        for _, label, label_  in self.output(sess, examples_raw, examples):
            token_cm.update(label, label_)
            error = error + ((label - label_)**2)
            correct_preds +=  (label == label_)
            total_preds += 1

        return token_cm, (error/total_preds, correct_preds/total_preds)

    def outputConfusionMatrix(self, cm):
        """ Generate a confusion matrix """
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

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw, last_epoch):
        global losses
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        curr_loss = 0.
        num_encountered = 0
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            curr_loss += loss
            num_encountered += 1
            if self.report: self.report.log_train_loss(loss)
        losses.append(curr_loss/num_encountered)

        print("")

        logger.info("Evaluating on development data")
        token_cm, metrics = self.evaluate(sess, dev_set, dev_set_raw)
        if last_epoch: 
            self.outputConfusionMatrix(token_cm.as_data())
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level errors:\n" + token_cm.summary())
        logger.info("Accuracy: %.2f", metrics[1])
        logger.info("Error: %.2f", metrics[0])

        return metrics[0], metrics[1]

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        global losses
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        epochs = list()
        errors = list()
        accuracies = list()
        last_epoch = False
        for epoch in range(self.config.n_epochs):
            if epoch == self.config.n_epochs - 1:
                last_epoch = True
            epochs.append(epoch+1)
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            error, score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw, last_epoch)
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

    def predict_book_age(self, sess, ngrams, year):
        ngrams_vectoriezed = self.helper.vectorize_ngrams(ngrams)
        predictions = self.predict_on_batch(sess, ngrams_vectoriezed)
        # print predictions
        # print year

        cumulative = np.sum(predictions, axis=0)
        # for i in xrange(len(predictions)):
        #     prediction = predictions[i]
        #     if max(prediction) > 0.50:
        #         print ngrams[i]
        #         print "your bucket", prediction[5]
        #         print "max bucket", max(prediction)
        #         print prediction
        #         print "\n"
        # counts = np.bincount(preds)
        # print counts
        print cumulative
        print sum(cumulative[1:6]), sum(cumulative[6:11])
        return np.argmax(cumulative)

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.helper = helper
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.report = None

        self.build()


def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
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
                print LBLS[result], input_data[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="book.txt", help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--glove', type=argparse.FileType('r'), default="../data/glove.6B.50d.txt", help="Path to glove_vectore file")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.set_defaults(func=do_evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)