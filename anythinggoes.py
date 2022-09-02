# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:27:16 2022

@author: omar al akkad
This implementation is for the anything goes part of the NLP challenge.

"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


file = open('train-v2.tsv', 'r', encoding = "UTF-8").readlines()

labels = []
sentences = []
for line in file:
    labels.append(int(line[0]))
    sentences.append(line[2:-1].lower())

X_train, X_test, Y_train, Y_test = train_test_split(sentences, labels,
                                                    test_size = 0.2,
                                                    random_state = 42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts).toarray()

clf = MultinomialNB().fit(X_train_tf, Y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts).toarray()

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
print("Precision of the test set = ", precision)
print("Recall of the test set = ", recall)
print("F1_score of the test set = ", f1)

