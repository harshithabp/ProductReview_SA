from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

test_csv = pd.read_csv(r'C:\Users\harshitha\Downloads\test_dataset.csv') # path to file
train_csv = pd.read_csv(r'C:\Users\harshitha\Downloads\train_dataset.csv') # path to file

test_csv = test_csv[['clean_reviews','sentiments']]
test_csv

train_csv=train_csv[['clean_reviews','sentiments']]
train_csv

train_X = train_csv['clean_reviews']   # '0' corresponds to Texts/Reviews
train_y = train_csv['sentiments']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X = test_csv['clean_reviews']
test_y = test_csv['sentiments']

# t = time()  # not compulsory

# # loading CountVectorizer
# tf_vectorizer = CountVectorizer() # or term frequency

# X_train_tf = tf_vectorizer.fit_transform(train_X.values.astype('U'))

# duration = time() - t
# print("Time taken to extract features from training data : %f seconds" % (duration))
# print("n_samples: %d, n_features: %d" % X_train_tf.shape)

# t = time()
# X_test_tf = tf_vectorizer.transform(test_X.values.astype('U'))

# duration = time() - t
# print("Time taken to extract features from test data : %f seconds" % (duration))
# print("n_samples: %d, n_features: %d" % X_test_tf.shape)

# # build naive bayes classification model
# t = time()

# naive_bayes_classifier = MultinomialNB()
# naive_bayes_classifier.fit(X_train_tf, train_y)

# training_time = time() - t
# print("train time: %0.3fs" % training_time)

# t = time()
# y_pred = naive_bayes_classifier.predict(X_test_tf)

# test_time = time() - t
# print("test time:  %0.3fs" % test_time)

# score1 = metrics.accuracy_score(test_y, y_pred)
# print("accuracy:   %0.3f" % score1)

# print(metrics.classification_report(test_y, y_pred,target_names=['Positive', 'Negative']))

# print("confusion matrix:")
# print(metrics.confusion_matrix(test_y, y_pred))

# print('------------------------------')


# test=[["bad dont buy"],["Product was dirty like old machine I don't want this I need to return this item and refund my money back"]]
# test_features = tf_vectorizer.transform([r[0] for r in test])
# #y_pred = classifier_linear.predict(test_features)
# y_pred = naive_bayes_classifier.predict(test_features)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression().fit(train_X,train_y)
from sklearn.metrics import r2_score
print(r2_score(regressor.predict(test_X)))