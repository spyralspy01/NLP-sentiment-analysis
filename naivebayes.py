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

#split index for training and test data (10% is used as test data)

split_index=int(len(sentences)-(0.1*len(sentences)))

#Preparing training data
train_data=[]
train_labels=[]
for x in range(split_index):
    train_data.append(sentences[x])
    train_labels.append(labels[x])


#Preparaing test data

test_sentences=[]
test_labels=[]

for x in range(split_index,len(sentences)):
    test_sentences.append(sentences[x])
    test_labels.append(labels[x])

#check split ratio of +ive & -ive tweets 
count_1=0
count_0=0
for x in test_labels:
  if x==1:
    count_1=count_1+1
  else:
    count_0=count_0+1

print("Count of Negative tweets in test data",count_0, " % of Negative Tweets", (count_0/len(test_labels))*100)    
print("Count of positive tweets in test data",count_1, " % of Positive Tweets", (count_1/len(test_labels))*100)    
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
predictions = []
for sentence in test_sentences:
    prob_positive = 1
    prob_negative = 1
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
        predictions.append(1)
    else:
        predictions.append(0)

total = len(predictions)
correct = 0
for i in range(total):
    if predictions[i] == test_labels[i]:
        correct += 1

accuracy = correct/total
print("Accuracy on Test Data",accuracy)



