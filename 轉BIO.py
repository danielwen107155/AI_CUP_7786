# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:14:09 2025

@author: PigFarmer
"""

import re



def simple_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


with open("C:/Users/PigFarmer/Downloads/bio/whisperx/task1_answer22.txt", "r", encoding="utf-8") as f1:
    task1_lines = f1.readlines()

with open("C:/Users/PigFarmer/Downloads/bio/task2_train22.txt", "r", encoding="utf-8") as f2:
    task2_lines = f2.readlines()


label_dict = {}
for line in task2_lines:
    if '\t' not in line:
        continue
    id_str, content = line.strip().split('\t', 1)
    id_num = int(id_str)
    if '\t' in content:
        text, labels_str = content.split('\t', 1)
        labels = [tuple(l.split(":", 1)) for l in labels_str.strip().split("\\n") if ":" in l]
    else:
        text, labels = content, []
    label_dict[id_num] = {"text": text, "labels": labels}


bio_data = []
for line in task1_lines:
    if '\t' not in line:
        continue
    id_str, raw_text = line.strip().split('\t', 1)
    id_num = int(id_str)
    if id_num not in label_dict:
        continue

    tokens = simple_tokenize(raw_text)
    labels = ['O'] * len(tokens)

    for label_type, label_text in label_dict[id_num]["labels"]:
        label_tokens = simple_tokenize(label_text)
        lt_len = len(label_tokens)
        for i in range(len(tokens) - lt_len + 1):
            if tokens[i:i+lt_len] == label_tokens and all(l == 'O' for l in labels[i:i+lt_len]):
                labels[i] = f'B-{label_type}'
                for j in range(1, lt_len):
                    labels[i+j] = f'I-{label_type}'


    if any(label != 'O' for label in labels):
        bio_data.append((id_num, tokens, labels))


with open("C:/Users/PigFarmer/Downloads/bio/bio22.txt", "w", encoding="utf-8") as f:
    for id_num, tokens, labels in bio_data:
        f.write(f"{id_num}\t{tokens}\t{labels}\n")
