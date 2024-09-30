from torch import nn
import torch


class RNN(nn.Module):
    def __init__(self, vectors, volcabulary_length):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(volcabulary_length, 300)
        self.rnn = nn.RNN(300, 1024)
        self.linear = nn.Linear(1024, volcabulary_length) # 输入应该和输出是一样的

    def forward(self, x):
        x = self.embedding(x)
        x, h = self.rnn(x) # x: torch.Size([1, 2, 8])
        # x = self.linear(x) # x: [1, 2, 203]
        x = self.linear(x[0, -1])
        return x