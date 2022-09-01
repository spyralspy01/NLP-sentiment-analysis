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
    labels.append(line[0])
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




