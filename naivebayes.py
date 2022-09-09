# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:01:41 2022

@author: omar al akkad/ Muhammad Aqeel

This is a from scratch implementation to the sentiment analysis problem using
naive bayes
"""
from math import log
import re

#Reading file and separation of tweets and labels
file = open('train-v2.tsv', 'r', encoding = "UTF-8").readlines()

labels = []
sentences = []
for line in file:
    labels.append(int(line[0]))
    sentences.append(line[2:-1].lower())

# removing unwanted characters from sentences
# removing unwanted characters improved accuracy by 2%
# unwanted_char = ""
unwanted_char = "![]{};\,<>./?@$%^&*_-=+~@"
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        if sentences[i][j] in unwanted_char:
            sentences[i] = sentences[i].replace(sentences[i][j]," ")

#split index for training and dev data (10% is used as dev set)

split_index=int(len(sentences)-(0.1*len(sentences)))

#Preparing training data
train_data=[]
train_labels=[]
for x in range(split_index):
    train_data.append(sentences[x])
    train_labels.append(labels[x])

#Preparing dev set
dev_sentences=[]
dev_labels=[]
for x in range(split_index,len(sentences)):
    dev_sentences.append(sentences[x])
    dev_labels.append(labels[x])


#check split ratio of +ive & -ive tweets
count_1=0
count_0=0
for x in dev_labels:
  if x==1:
    count_1=count_1+1
  else:
    count_0=count_0+1

print("Count of Negative tweets in dev data",count_0, " % of Negative Tweets", (count_0/len(dev_labels))*100)
print("Count of positive tweets in dev data",count_1, " % of Positive Tweets", (count_1/len(dev_labels))*100)
bog = {}

for line in train_data:
    for word in re.findall(r"[\w]+|[^\s\w]", line): #separating punctuation as tokens
        if word not in bog.keys():
            bog[word] = 1
        else:
            bog[word] += 1

bog = {k: v for k, v in sorted(bog.items(), reverse = True, key=lambda item: item[1])}
vocab_size = len(bog.keys())
bog_positive = {}
bog_negative = {}
count_positive = 0
count_negative = 0

#count which words appear in +ive tweet and -ive tweet
for i in range(len(train_data)):
    for word in re.findall(r"[\w]+|[^\s\w]", train_data[i]):
        if labels[i] == 0:
            if word not in bog_negative.keys():
                bog_negative[word] = 1
                count_negative += 1
            else:
                bog_negative[word] += 1
                count_negative += 1
        else:
            if word not in bog_positive.keys():
                bog_positive[word] = 1
                count_positive += 1
            else:
                bog_positive[word] += 1
                count_positive += 1


bog_positive = {k: v for k, v in sorted(bog_positive.items(), reverse = True, key=lambda item: item[1])}
bog_negative = {k: v for k, v in sorted(bog_negative.items(), reverse = True, key=lambda item: item[1])}



#prediction
# we use the naive bayes function provided in the book, we add alpha smoothing
# to increase performance. we also take the log of the probabilities
train_predictions = []
for sentence in train_data:
    prob_positive = 0
    prob_negative = 0
    for word in sentence.split():
        try:
            prob_positive += log((bog_positive[word]+1)/(count_positive+vocab_size))
        except:
            prob_positive += log((0+1)/(count_positive+vocab_size))
        try:
            prob_negative += log((bog_negative[word]+1)/(count_negative+vocab_size))
        except:
            prob_negative += log((0+1)/(count_negative+vocab_size))

    if prob_positive > prob_negative:
        train_predictions.append(1)
    else:
        train_predictions.append(0)

train_total = len(train_predictions)
train_correct = 0
for i in range(train_total):
    if train_predictions[i] == train_labels[i]:
        train_correct += 1

train_accuracy = train_correct/train_total
print("Accuracy on Training set",train_accuracy)

dev_predictions = []
for sentence in dev_sentences:
    prob_positive = 0
    prob_negative = 0
    for word in sentence.split():
        try:
            prob_positive += log((bog_positive[word]+1)/(count_positive+vocab_size))
        except:
            prob_positive += log((0+1)/(count_positive+vocab_size))
        try:
            prob_negative += log((bog_negative[word]+1)/(count_negative+vocab_size))
        except:
            prob_negative += log((0+1)/(count_negative+vocab_size))

    if prob_positive > prob_negative:
        dev_predictions.append(1)
    else:
        dev_predictions.append(0)

dev_total = len(dev_predictions)
dev_correct = 0
for i in range(dev_total):
    if dev_predictions[i] == dev_labels[i]:
        dev_correct += 1

dev_accuracy = dev_correct/dev_total
print("Accuracy on Dev set",dev_accuracy)

"""
Testing performance on test set.
Preprocess test set.
Predict.
Calculate Accuracy.
"""

#Reading file and separation of tweets and labels
test_file = open('test.tsv', 'r', encoding = "UTF-8").readlines()

test_labels = []
test_sentences = []
for line in test_file:
    test_labels.append(int(line[0]))
    test_sentences.append(line[2:-1].lower())

# removing unwanted characters from sentences
# removing unwanted characters improved accuracy by 2%
# unwanted_char = ""
unwanted_char = "![]{};\,<>./?@$%^&*_-=+~@"
for i in range(len(test_sentences)):
    for j in range(len(test_sentences[i])):
        if test_sentences[i][j] in unwanted_char:
            test_sentences[i] = test_sentences[i].replace(test_sentences[i][j]," ")
test_predictions = []
for sentence in test_sentences:
    prob_positive = 0
    prob_negative = 0
    for word in sentence.split():
        try:
            prob_positive += log((bog_positive[word]+1)/(count_positive+vocab_size))
        except:
            prob_positive += log((0+1)/(count_positive+vocab_size))
        try:
            prob_negative += log((bog_negative[word]+1)/(count_negative+vocab_size))
        except:
            prob_negative += log((0+1)/(count_negative+vocab_size))

    if prob_positive > prob_negative:
        test_predictions.append(1)
    else:
        test_predictions.append(0)

test_total = len(test_predictions)
test_correct = 0
for i in range(test_total):
    if test_predictions[i] == test_labels[i]:
        test_correct += 1

test_accuracy = test_correct/test_total
print("Accuracy on Test set",test_accuracy)

"""
Accuracy on Training set 0.817625
Accuracy on Dev set 0.743125
Accuracy on Test set 0.7751
"""