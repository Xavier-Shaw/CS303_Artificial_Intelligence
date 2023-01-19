import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class NetModel(torch.nn.Module):
    def __init__(self, n_features, n_output):
        super(NetModel, self).__init__()
        self.hiddenLayer1 = torch.nn.Linear(n_features, 100)
        self.hiddenLayer2 = torch.nn.Linear(100, 50)
        self.hiddenLayer3 = torch.nn.Linear(50, 20)
        self.predictLayer = torch.nn.Linear(20, n_output)

    # 搭建神经网络， 输入data: x
    def forward(self, x):
        # 使用隐藏层加工x，用激励函数激活隐藏层输出的信息
        x = F.relu(self.hiddenLayer1(x))
        # 使用预测层预测
        x = self.hiddenLayer2(x)
        x = self.hiddenLayer3(x)
        x = self.predictLayer(x)
        return x


if __name__ == "__main__":
    information = torch.load("./data.pth")
    data = information["feature"]
    target = information["label"]

    train_set, test_set, train_target, test_target = train_test_split(data, target, test_size=0.2)
    # 处理labels
    train_label = torch.zeros(len(train_target), 10, dtype=torch.float32)
    test_label = torch.zeros(len(test_target), 10, dtype=torch.float32)
    for i in range(len(train_target)):
        train_label[i][train_target[i]] = 1.
    for i in range(len(test_target)):
        test_label[i][test_target[i]] = 1.

    # initialize model
    net = NetModel(n_features=256, n_output=10)   # 总共10类

    # learning parameters
    lr = 0.01
    epochs = 200
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()    #crossentropy为啥不行？

    # train
    scheduler.step()
    net.train()
    round = 0
    for epoch in range(epochs):
        round += 1
        optimizer.zero_grad()
        output = net(train_set)
        loss = criterion(output, train_label)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        if round % 10 == 0:
            model_pred = torch.tensor([item.cpu().detach().numpy() for item in output])
            idx = np.argmax(model_pred, axis=1)
            output_all = []
            targets_all = []
            for index in idx:
                output_all.append(index.item())
            for label in train_target:
                targets_all.append(label.item())    # target:不需要向量，数字即可
            print(f'training loss: {loss}')
            print(classification_report(targets_all, output_all))

    # test
    test_output = net(test_set)
    model_pred = torch.tensor([item.cpu().detach().numpy() for item in test_output])
    idx = np.argmax(model_pred, axis=1)
    output_all = []
    targets_all = []
    for index in idx:
        output_all.append(index.item())
    for label in test_target:
        targets_all.append(label.item())    # target:不需要向量，数字即可
    print(classification_report(targets_all, output_all))

    torch.save(net.state_dict(), './model1.pth')
