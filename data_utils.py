import pandas as pd
import numpy as np
import tensorflow as tf
import json

def load_data(file_path):
    """Função para carregar o dataset"""
    return pd.read_csv(file_path, encoding= 'utf-8', index_col=0)

def to_tuples(data):
    """Converte os dados para formato de tuplas"""
    iterator = zip(data["token"].values.tolist(), data["iob_label"].values.tolist())
    return [(token, iob_label) for token, iob_label in iterator]

def build_vocab(data):
    """Cria dicionários de índice para palavras e tags"""
    all_words = list(set(data["token"].values))
    all_tags = list(set(data["iob_label"].values))

    word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
    word2index["--UNKNOWN_WORD--"] = 0
    word2index["--PADDING--"] = 1

    index2word = {idx: word for word, idx in word2index.items()}
    tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}

    tag2index["--PADDING--"] = 0

    index2tag = {idx: word for word, idx in tag2index.items()}

    return word2index, index2word, tag2index, index2tag

def tokenize(reports, word2index, tag2index, max_sentence_size = 512):
    contents = []
    labels = []
    for report in reports:
        content = []
        label = []
    
        for i in range(len(report)):
            token, iob_tag = report[i]
            word_idx = word2index.get(token, 0)
            tag_idx = tag2index.get(iob_tag, 0)
            content.append(word_idx)
            label.append(tag_idx)

        contents.append(content)
        labels.append(label)

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, maxlen=max_sentence_size, padding='post', value=1)
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_sentence_size, padding='post')

    tag_size = len(tag2index)

    labels_categorical = [tf.keras.utils.to_categorical(i, num_classes=tag_size) for i in labels]
    labels_categorical = np.asarray(labels_categorical)

    return contents, labels, labels_categorical

def save_vocab_separately(word2index, index2word, tag2index, index2tag, directory):
    """Salva cada dicionário em um arquivo JSON separado"""
    with open(f'{directory}/word2index.json', 'w', encoding='utf-8') as f:
        json.dump(word2index, f, ensure_ascii=False, indent=4)

    with open(f'{directory}/index2word.json', 'w', encoding='utf-8') as f:
        json.dump(index2word, f, ensure_ascii=False, indent=4)

    with open(f'{directory}/tag2index.json', 'w', encoding='utf-8') as f:
        json.dump(tag2index, f, ensure_ascii=False, indent=4)

    with open(f'{directory}/index2tag.json', 'w', encoding='utf-8') as f:
        json.dump(index2tag, f, ensure_ascii=False, indent=4)

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    return vocab

