# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:09:34 2025

@author: PigFarmer
"""

import pandas as pd
import ast
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, TFBertForTokenClassification
from seqeval.metrics import f1_score
import re

def load_and_parse(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None, names=["id", "tokens", "labels"])
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["labels"] = df["labels"].apply(ast.literal_eval)
    return df

df1 = load_and_parse("C:/Users/PigFarmer/Downloads/bio/bio12.txt")
df2 = load_and_parse("C:/Users/PigFarmer/Downloads/bio/bio22.txt")
df3 = load_and_parse("C:/Users/PigFarmer/Downloads/bio/bio32.txt")
df = pd.concat([df1, df2, df3], ignore_index=True)

sentences = df["tokens"].tolist()
labels = df["labels"].tolist()


unique_tags = sorted(set(tag for seq in labels for tag in seq))
label2id = {label: i for i, label in enumerate(unique_tags)}
id2label = {i: label for label, i in label2id.items()}
labels = [[label2id[tag] for tag in seq] for seq in labels]


tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased)


def tokenize_and_align_labels(texts, labels):
    encodings = tokenizer(texts, is_split_into_words=True, truncation=True,
                          padding="max_length", max_length=128, return_tensors="np")

    encoded_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)

    encodings["labels"] = np.array(encoded_labels)
    return encodings

encodings = tokenize_and_align_labels(sentences, labels)
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
label_ids = encodings["labels"]


input_ids_list = list(input_ids)
attention_mask_list = list(attention_mask)
labels_list = list(label_ids)

X_train_ids, X_temp_ids, X_train_mask, X_temp_mask, y_train, y_temp = train_test_split(
    input_ids_list, attention_mask_list, labels_list, test_size=0.2, random_state=42
)
X_val_ids, X_test_ids, X_val_mask, X_test_mask, y_val, y_test = train_test_split(
    X_temp_ids, X_temp_mask, y_temp, test_size=0.5, random_state=42
)


def to_tf_dataset(input_ids, attention_mask, labels):
    return tf.data.Dataset.from_tensor_slices(({
        "input_ids": tf.convert_to_tensor(input_ids, dtype=tf.int32),
        "attention_mask": tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    }, tf.convert_to_tensor(labels, dtype=tf.int32))).batch(8)

train_dataset = to_tf_dataset(X_train_ids, X_train_mask, y_train)
val_dataset = to_tf_dataset(X_val_ids, X_val_mask, y_val)


model = TFBertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)


class SeqevalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, id2label):
        super().__init__()
        self.val_data = val_data
        self.id2label = id2label

    def on_epoch_end(self, epoch, logs=None):
        true_labels = []
        pred_labels = []

        for batch in self.val_data:
            inputs, labels = batch
            logits = self.model(inputs, training=False).logits
            pred_ids = tf.argmax(logits, axis=-1).numpy()
            labels = labels.numpy()

            for i in range(labels.shape[0]):
                true_seq = []
                pred_seq = []
                for j in range(labels.shape[1]):
                    if labels[i][j] == -100:
                        continue
                    true_seq.append(self.id2label[labels[i][j]])
                    pred_seq.append(self.id2label[pred_ids[i][j]])
                true_labels.append(true_seq)
                pred_labels.append(pred_seq)

        f1 = f1_score(true_labels, pred_labels)
        print(f"\nEpoch {epoch + 1} F1-score: {f1:.4f}")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5))
model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[SeqevalCallback(val_dataset, id2label)])
# %%



df_pred = pd.read_csv("C:/Users/PigFarmer/Downloads/Private_dataset/task1__answer_en.txt", sep="\t", header=None, names=["id", "text"])

def tokenize_text(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

results = []

for idx, text in enumerate(df_pred["text"].tolist()):
    tokens = tokenize_text(text)
    encoding = tokenizer(tokens,
                         is_split_into_words=True,
                         padding="max_length",
                         truncation=True,
                         max_length=128,
                         return_attention_mask=True)

    word_ids = encoding.word_ids()
    input_ids = tf.convert_to_tensor([encoding["input_ids"]], dtype=tf.int32)
    attention_mask = tf.convert_to_tensor([encoding["attention_mask"]], dtype=tf.int32)

    output = model({"input_ids": input_ids, "attention_mask": attention_mask})
    pred_ids = tf.argmax(output.logits, axis=-1).numpy()[0]

    final_labels = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        label_id = int(pred_ids[i])
        label = id2label.get(label_id, "O")
        final_labels.append(label)
        previous_word_idx = word_idx

    results.append({
        "id": df_pred["id"][idx],
        "tokens": tokens,
        "labels": final_labels
    })


for i in range(min(3, len(results))):
    print(f"Sample {i+1}")
    for tok, lab in zip(results[i]["tokens"], results[i]["labels"]):
        print(f"{tok} -> {lab}")
    print("-" * 50)


output_df = pd.DataFrame(results)
output_df.to_csv("C:/Users/PigFarmer/Downloads/Private_dataset/task2222.csv", index=False)
print("✅ 預測結果已輸出")
