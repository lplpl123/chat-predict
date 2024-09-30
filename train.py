import torch
from tools import preprocess
from models import model


def train():
    for epoch in range(25):
        for x, y_label in train_data:
            x = x.unsqueeze(0)
            y = rnn.forward(x) # torch.Size([128])
            loss = loss_function(y, y_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("完成第{}轮训练，loss为{}".format(epoch, loss))


if __name__ == "__main__":
    train_data, word2vec, volcabulary = preprocess.preprocess("train")
    # init
    rnn = model.RNN(word2vec.wv.vectors, len(volcabulary))
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
    # train
    train()
    torch.save(rnn, "saved_models/rnn.pth")
    torch.save(rnn.state_dict(), 'saved_models/rnn.params')