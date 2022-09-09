# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:27:16 2022

@author: omar al akkad
This implementation is for the anything goes part of the NLP challenge.

"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


file = open('train-v2.tsv', 'r', encoding = "UTF-8").readlines()

labels = []
sentences = []
for line in file:
    labels.append(int(line[0]))
    sentences.append(line[2:-1].lower())

X_train, X_dev, Y_train, Y_dev = train_test_split(sentences, labels,
                                                    test_size = 0.2,
                                                    random_state = 42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts).toarray()

# clf = MultinomialNB().fit(X_train_tf, Y_train)
scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
clf = scikit_log_reg.fit(X_train_tf, Y_train)

X_dev_counts = count_vect.transform(X_dev)
X_dev_tfidf = tf_transformer.transform(X_dev_counts).toarray()

predicted = clf.predict(X_train_tf)

encode = LabelEncoder()
encode.fit([0,1])
Y_train_encoded = encode.transform(Y_train)
pred_encoded = encode.transform(predicted)

accuracy = accuracy_score(Y_train_encoded, pred_encoded)
precision = precision_score(Y_train_encoded, pred_encoded)
recall = recall_score(Y_train_encoded, pred_encoded)
f1 = f1_score(Y_train_encoded, pred_encoded)
print("Accuracy on the train set = ", accuracy)
print("Precision of the train set = ", precision)
print("Recall of the train set = ", recall)
print("F1_score of the train set = ", f1)

predicted = clf.predict(X_dev_tfidf)
encode = LabelEncoder()
encode.fit([0,1])
Y_dev_encoded = encode.transform(Y_dev)
pred_encoded = encode.transform(predicted)

accuracy = accuracy_score(Y_dev_encoded, pred_encoded)
precision = precision_score(Y_dev_encoded, pred_encoded)
recall = recall_score(Y_dev_encoded, pred_encoded)
f1 = f1_score(Y_dev_encoded, pred_encoded)
print("Accuracy on the dev set = ", accuracy)
print("Precision of the dev set = ", precision)
print("Recall of the dev set = ", recall)
print("F1_score of the dev set = ", f1)

"""
Checking metrics on new test set provided
"""
#Reading file and separation of tweets and labels
test_file = open('test.tsv', 'r', encoding = "UTF-8").readlines()

Y_test = []
X_test = []
for line in test_file:
    Y_test.append(int(line[0]))
    X_test.append(line[2:-1].lower())

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts).toarray()

predicted = clf.predict(X_test_tfidf)
encode = LabelEncoder()
encode.fit([0,1])
Y_test_encoded = encode.transform(Y_test)
pred_encoded = encode.transform(predicted)

accuracy = accuracy_score(Y_test_encoded, pred_encoded)
precision = precision_score(Y_test_encoded, pred_encoded)
recall = recall_score(Y_test_encoded, pred_encoded)
f1 = f1_score(Y_test_encoded, pred_encoded)
print("Accuracy on the test set = ", accuracy)
print("Precision of the dev set = ", precision)
print("Recall of the dev set = ", recall)
print("F1_score of the dev set = ", f1)

"""
Accuracy on the train set =  0.83690625
Precision of the train set =  0.8333024748503364
Recall of the train set =  0.8428214731585518
F1_score of the train set =  0.8380349439841107
Accuracy on the dev set =  0.7499375
Precision of the dev set =  0.7464684014869889
Recall of the dev set =  0.7549818272966538
F1_score of the dev set =  0.7507009782540969
Accuracy on the test set =  0.7881
Precision of the dev set =  0.7905278531191321
Recall of the dev set =  0.7726345840130505
F1_score of the dev set =  0.7814788078787254
"""