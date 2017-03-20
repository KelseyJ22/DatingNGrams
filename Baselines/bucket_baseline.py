from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn_utils_bucket import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

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
expected = np.array(test[1])

print 'read train and test'

log_reg.fit(train_data, dense)
print(log_reg)
log_reg_pred = log_reg.predict(test_data)
print 'Logistic Regression Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(expected, log_reg_pred))
print(metrics.confusion_matrix(expected, log_reg_pred))


naive_bayes.fit(train_data, dense)
print(naive_bayes)
nb_pred = naive_bayes.predict(test_data)
print 'Naive Bayes Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(expected, nb_pred))
print(metrics.confusion_matrix(expected, nb_pred))
