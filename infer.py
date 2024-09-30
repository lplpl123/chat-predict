import torch
import numpy as np
from tools import preprocess
from models import model


def infer(data):
    data = data.unsqueeze(0)
    y = rnn.forward(data)
    result = torch.argmax(y, dim=0)
    return result

if __name__ == '__main__':

    # words编码
    encoding_words = preprocess.preprocess("infer")
    train_data, word2vec, volcabulary = preprocess.preprocess("train")
    rnn = torch.load('./saved_models/rnn.pth')
    while True:
        import time
        time.sleep(1)
        # tensor([11]) tensor([[ 11, 202]])
        predict_index = infer(encoding_words)
        # decoding
        predict_word = volcabulary[predict_index]
        print(predict_word, end="") # 综上所述
        # next
        encoding_words = list(encoding_words)
        encoding_words.pop(0)
        encoding_words.append(predict_index.item())
        encoding_words = torch.tensor(encoding_words)