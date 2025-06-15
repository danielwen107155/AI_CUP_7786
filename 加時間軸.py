# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 00:31:27 2025

@author: PigFarmer
"""

import pandas as pd
import json
import ast


bert_df = pd.read_csv('C:/Users/PigFarmer/Downloads/Private_dataset/task1_bio_output _ch.csv')
with open('C:/Users/PigFarmer/Downloads/Private_dataset/task1_answer_timestamps123.json', 'r', encoding='utf-8') as f:
    task1_data = json.load(f)
# %%



bert_df['tokens'] = bert_df['tokens'].apply(ast.literal_eval)
bert_df['labels'] = bert_df['labels'].apply(ast.literal_eval)


def extract_labeled_tokens(row):
    return [(tok, label) for tok, label in zip(row['tokens'], row['labels']) if label != 'O']

bert_df['labeled_tokens'] = bert_df.apply(extract_labeled_tokens, axis=1)
bert_df_with_labels = bert_df[bert_df['labeled_tokens'].map(len) > 0]


def match_tokens_with_timestamps(token_label_list, segments):
    result_segments = []
    segment_words = [seg['word'].strip(",.?!;:").lower() for seg in segments]
    used_indices = set()

    for token, label in token_label_list:
        token_clean = token.strip(",.?!;:").lower()
        for i, word in enumerate(segment_words):
            if word == token_clean and i not in used_indices:
                matched_segment = segments[i].copy()
                matched_segment['label'] = label
                result_segments.append(matched_segment)
                used_indices.add(i)
                break

    return result_segments


task1_like_output = {}
for _, row in bert_df_with_labels.iterrows():
    sent_id = str(row['id'])
    if sent_id in task1_data:
        segments = task1_data[sent_id]['segments']
        matched = match_tokens_with_timestamps(row['labeled_tokens'], segments)
        if matched:
            task1_like_output[sent_id] = {
                'text': task1_data[sent_id]['text'],
                'segments': matched
            }


def convert_to_task2_format(task1_like_data):
    output_lines = []
    for sent_id, data in task1_like_data.items():
        segments = data['segments']
        merged_entities = []
        current_entity = []
        current_label_type = None
        for seg in segments:
            label = seg['label']
            label_type = label[2:]
            if label.startswith('B-') or (label.startswith('I-') and label_type != current_label_type):
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = [seg]
                current_label_type = label_type
            elif label.startswith('I-') and label_type == current_label_type:
                current_entity.append(seg)
            else:
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = []
                current_label_type = None
        if current_entity:
            merged_entities.append(current_entity)

        for entity in merged_entities:
            start = min(e['start'] for e in entity)
            end = max(e['end'] for e in entity)
            label_type = entity[0]['label'][2:]
            phrase = ' '.join(e['word'].strip(",.?!;:") for e in entity)
            output_lines.append(f"{sent_id}\t{label_type}\t{start:.3f}\t{end:.3f}\t{phrase}")
    return output_lines


output_lines = convert_to_task2_format(task1_like_output)
with open('C:/Users/PigFarmer/Downloads/Private_dataset/task2_answer_ch2.txt', 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')
