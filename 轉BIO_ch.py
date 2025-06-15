# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:29:33 2025

@author: PigFarmer
"""

import re

# 讀取檔案內容
with open('task1_answer.txt', 'r', encoding='utf-8') as f:
    task1_lines = f.readlines()

with open('task2_train.txt', 'r', encoding='utf-8') as f:
    task2_lines = f.readlines()

def parse_annotations(annotation_str):
    annotations = []
    for item in annotation_str.strip().split('\\n'):
        if ':' in item:
            label, phrase = item.split(':', 1)
            annotations.append((label, phrase))
    return annotations

def get_bio_labels(text, annotations):
    bio_labels = ['O'] * len(text)
    for label, phrase in annotations:
        start_idx = text.find(phrase)
        if start_idx == -1:
            continue  # 找不到標記文字
        for i in range(len(phrase)):
            if bio_labels[start_idx + i] != 'O':
                continue  # 避免重複標記
            bio_labels[start_idx + i] = f'I-{label}' if i > 0 else f'B-{label}'
    return bio_labels

# 建立 BIO 格式資料
bio_data = []
for t1_line, t2_line in zip(task1_lines, task2_lines):
    t1_parts = t1_line.strip().split('\t')
    t2_parts = t2_line.strip().split('\t')

    if len(t1_parts) < 2 or len(t2_parts) < 3:
        continue  # 跳過格式不正確的行

    file_id = t1_parts[0]
    text = t1_parts[1]
    annotations = parse_annotations(t2_parts[2])

    chars = list(text)
    bio_labels = get_bio_labels(text, annotations)

    bio_data.append(f"{file_id}\t{chars}\t{bio_labels}")

# 寫入輸出檔
with open('mapped_bio_output.txt', 'w', encoding='utf-8') as f:
    for line in bio_data:
        f.write(line + '\n')
