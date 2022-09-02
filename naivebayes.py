# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:01:41 2022

@author: omar al akkad

This is a from scratch implementation to the sentiment analysis problem using
naive bayes
"""

import string
import pandas as pd

# file = pd.read_csv('train-v2.tsv', sep='\t', header = None)
file = open('train-v2.tsv', 'r', encoding = "UTF-8").readlines()

labels = []
sentences = []
for line in file:
    labels.append(int(line[0]))
    sentences.append(line[2:-1].lower())

unwanted_char = "![]{};\,<>./?@#$%^&*_-=+~@"
bog = {}

for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        if sentences[i][j] in unwanted_char:
            sentences[i] = sentences[i].replace(sentences[i][j]," ")

for line in sentences:
    for word in line.split():
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
for i in range(len(sentences)):
    for word in sentences[i].split():
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

for key in bog.keys():
    if key not in bog_positive.keys():
        bog_positive[key] = 0
    if key not in bog_negative.keys():
        bog_negative[key] = 0

bog_positive = {k: v for k, v in sorted(bog_positive.items(), reverse = True, key=lambda item: item[1])}
bog_negative = {k: v for k, v in sorted(bog_negative.items(), reverse = True, key=lambda item: item[1])}

probabilities_positive = {}
probabilites_negative = {}

for key in bog_positive.keys():
    probabilities_positive[key] = (bog_positive[key]+1)/(count_positive+vocab_size)
for key in bog_negative.keys():
    probabilites_negative[key] = (bog_negative[key]+1)/(count_negative+vocab_size)

predictions = []
for sentence in sentences:
    prob_positive = 1
    prob_negative = 1
    for word in sentence.split():
        prob_positive *= probabilities_positive[word]
        prob_negative *= probabilites_negative[word]
    if prob_positive > prob_negative:
        predictions.append(1)
    else:
        predictions.append(0)

total = len(predictions)
correct = 0
for i in range(total):
    if predictions[i] == labels[i]:
        correct += 1

accuracy = correct/total



