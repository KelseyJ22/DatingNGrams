from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn_utils import load_dataset, np_kl_divergence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.stats import entropy

count_vect = CountVectorizer()
log_reg = LogisticRegression()
naive_bayes = GaussianNB()

data = load_dataset()
train = data[0]
test = data[1]

train_counts = count_vect.fit_transform(train[0])
transformer = TfidfTransformer(use_idf=False).fit(train_counts)
train_data = transformer.transform(train_counts).toarray()

test_counts = count_vect.transform(test[0])
test_data = transformer.transform(test_counts).toarray()
dense = np.array(train[1])
expected = test[1]

random_baseline = np.array([[0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09] for i in xrange(len(expected))])

print 'read train and test'

log_reg.fit(train_data, dense)
print(log_reg)
log_reg_pred = log_reg.predict_proba(test_data)
print 'Logistic Regression Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print log_reg_pred.shape

kl_logistic = np_kl_divergence(np.array(expected), log_reg_pred)
print kl_logistic

naive_bayes.fit(train_data, dense)
print(naive_bayes)
nb_pred = naive_bayes.predict_proba(test_data)
print 'Naive Bayes Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
kl_nb = np_kl_divergence(np.array(expected), nb_pred)
print kl_nb

kl_random = np_kl_divergence(np.array(expected), random_baseline)
print kl_random

print "LogisticRegression KL-Divergence", kl_logistic
print "NaiveBayes KL-Divergence", kl_nb
print "Random KL-Divergence", kl_random
