import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_input = nn.Linear(3, 64)

        self.fc2_hidden = nn.Linear(64, 128)
        self.fc3_hidden = nn.Linear(128, 128)
        self.fc4_hidden = nn.Linear(128, 128)
        self.fc5_hidden = nn.Linear(128, 128)
        self.fc6_hidden = nn.Linear(128, 128)
        self.fc7_hidden = nn.Linear(128, 128)
        self.fc8_hidden = nn.Linear(128, 128)
        self.fc9_hidden = nn.Linear(128, 64)

        self.fc10_output = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1_input(x))

        x = F.relu(self.fc2_hidden(x))
        x = F.relu(self.fc3_hidden(x))
        x = F.relu(self.fc4_hidden(x))
        x = F.relu(self.fc5_hidden(x))
        x = F.relu(self.fc6_hidden(x))
        x = F.relu(self.fc7_hidden(x))
        x = F.relu(self.fc8_hidden(x))
        x = F.relu(self.fc9_hidden(x))
        x = self.dropout(x)

        x = self.fc10_output(x)

        return x

'''
def squaredHingeLoss(t, y):
    dot_pro = torch.tensordot(a=t, b=y)
    right_side = 1 - dot_pro
    zero = torch.tensor(0.0, dtype=torch.float32)
    hinge_loss = torch.maximum(input=zero, other=right_side)
    square_hinge_loss = torch.pow(input=hinge_loss, exponent=2)
    return square_hinge_loss
'''

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

    trainset = torch.utils.data.DataLoader(train, batch_size=64)
    testset = torch.utils.data.DataLoader(test, batch_size=64)

    net = Net()
    print(net)

    optimizer = optim.RMSprop(net.parameters(), lr=0.000001, momentum=0.9)

    EPOCHS = 10
    loss = None

    for epochs in range(EPOCHS):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 3))
            loss = F.mse_loss(input=y, target=output)
            loss.backward()
            optimizer.step()
        print(f"{epochs + 1}) loss: {loss}")
