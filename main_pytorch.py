import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_input = nn.Linear(3, 32)
        self.fc2_hidden = nn.Linear(32, 32)
        self.fc3_output = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1_input(x))
        x = F.relu(self.fc2_hidden(x))
        x = self.fc3_output(x)

        return F.log_softmax(x, dim=1)


def squaredHingeLoss(t, y):
    dot_pro = torch.tensordot(a=t, b=y)
    right_side = 1 - dot_pro
    # print(right_side)
    zero = torch.tensor(0.0, dtype=torch.float32)
    # print(zero)
    hinge_loss = torch.maximum(input=zero, other=right_side)
    # print(hinge_loss)
    square_hinge_loss = torch.pow(input=hinge_loss, exponent=2)
    # print(square_hinge_loss)
    return square_hinge_loss


if __name__ == "__main__":
    X_train = torch.from_numpy(np.load("dataset/X_train.npy"))
    X_test = torch.from_numpy(np.load("dataset/X_test.npy"))
    y_train = torch.from_numpy(np.load("dataset/y_train.npy"))
    y_test = torch.from_numpy(np.load("dataset/y_test.npy"))

    X_train = X_train.to(dtype=torch.float32)
    X_test = X_test.to(dtype=torch.float32)
    y_train = y_train.to(dtype=torch.float32)
    y_test = y_test.to(dtype=torch.float32)

    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)

    trainset = torch.utils.data.DataLoader(train, batch_size=32)
    testset = torch.utils.data.DataLoader(test, batch_size=32)

    net = Net()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 50
    loss = 0

    for epochs in range(EPOCHS):
        for data in trainset:
            X, y = data
            optimizer.zero_grad()
            # net.zero_grad()
            output = net(X.view(-1, 3))
            loss = squaredHingeLoss(t=output, y=y)
            # loss = F.mse_loss(input=y, target=output)
            loss.backward()
            optimizer.step()
        print(f"{epochs + 1}) loss: {loss}")
