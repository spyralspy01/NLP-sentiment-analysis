# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:01:41 2022

@author: omar al akkad

This is a from scratch implementation to the sentiment analysis problem using
naive bayes
"""
from math import log
import re
file = open('train-v2.tsv', 'r', encoding = "UTF-8").readlines()

labels = []
sentences = []
for line in file:
    labels.append(int(line[0]))
    sentences.append(line[2:-1].lower())


#unwanted_char = ""
# unwanted_char = "![]{};\,<>./?@$%^&*_-=+~@"
bog = {}

# for i in range(len(sentences)):
#     for j in range(len(sentences[i])):
#         if sentences[i][j] in unwanted_char:
#             sentences[i] = sentences[i].replace(sentences[i][j]," ")

for line in sentences:
    for word in re.findall(r"[\w]+|[^\s\w]", line):
        #print(word)
        if word not in bog.keys():
            bog[word] = 1
        else:
            bog[word] += 1

bog = {k: v for k, v in sorted(bog.items(), reverse = True, key=lambda item: item[1])}
vocab_size = len(bog.keys())
#print(bog)
bog_positive = {}
bog_negative = {}
count_positive = 0
count_negative = 0

for i in range(len(sentences)):
    for word in re.findall(r"[\w]+|[^\s\w]", sentences[i]):
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

predictions = []
for sentence in sentences:
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
    if predictions[i] == labels[i]:
        correct += 1

accuracy = correct/total
print(accuracy)



