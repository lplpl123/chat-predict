import copy

import jieba
from models import word_to_vec
import torch


def load_data():
    data = []
    with open('./data/train_data.txt',
              encoding="utf-8") as file:
        while True:
            string = file.readline()
            if string:
                data.append(string.strip())
            if not string:
                break
    return data

def split_words(data):
    split_data = []
    for seq in data:
        split_seq = jieba.cut(seq)
        split_data.append(list(split_seq))
    return split_data

def encoding_words(split_data, volcabulary):
    encoding_data = []
    for sentence in split_data:
        encodings = []
        for word in sentence:
            encodings.append(volcabulary.index(word))
        encoding_data.append(encodings)
    return encoding_data

def rearray_data(data):
    rearray_lst = []
    train_data = []
    for seq in data:
        for index in seq:
            rearray_lst.append(index)
    for i in range(len(rearray_lst) - 2):
        vec_data = [0] * 2
        vec_data[0] = torch.tensor([rearray_lst[i], rearray_lst[i + 1]])
        vec_data[1] = torch.tensor(rearray_lst[i + 2])
        train_data.append(copy.copy(vec_data))
    return train_data

def preprocess(type):
    if type == "train":
        data = load_data()
        split_data_words = split_words(data)
        word2vec = word_to_vec.word_vec(split_data_words)
        volcabulary = word2vec.wv.index_to_key
        split_data_indexs = encoding_words(split_data_words, volcabulary)
        train_data = rearray_data(split_data_indexs)
        return train_data, word2vec, volcabulary
    elif type == "infer":
        # 准备model
        data = load_data()
        split_data_words = split_words(data)
        word2vec = word_to_vec.word_vec(split_data_words)
        volcabulary = word2vec.wv.index_to_key
        # 推理数据预处理
        my_words = input("请输入你想说的话：")
        split_my_words = [word for word in jieba.cut(my_words)]
        # encoding
        split_words_indexs = []
        for word in split_my_words:
            split_words_indexs.append(volcabulary.index(word))
        split_words_indexs = torch.tensor(split_words_indexs)
        return split_words_indexs