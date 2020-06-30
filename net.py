import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionNet(nn.Module):
    def __init__(self,
                 input=(30 * 30 + 30 * 30 + 1),
                 output=30,
                 noise=False):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output)

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        self.sfm = nn.Softmax(dim=1)

        self.noise = None
        if noise:
            self.noise = torch.rand
        self.noise_strength = 0.01

    def forward(self, x):
        output = x
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        if not self.noise is None:
            n = self.noise(*output.shape) * self.noise_strength
            output = output + n
        # 连接关系掩码
        cm = x[:, :900]
        cm = cm.view(-1, 30, 30)
        lp = x[:, 1800].long()
        mask = [cm[i, lp[i]].unsqueeze(0) for i in range(cm.shape[0])]
        mask = torch.cat(tuple(mask), dim=0)
        output = output * mask
        output = F.softmax(output,-1)
        # output = self.sfm(output)
        return output


class ActorNetLoss(nn.Module):
    def __init__(self):
        super(ActorNetLoss, self).__init__()

    def forward(self, critic_q):
        total_loss = -torch.mean(critic_q)
        return total_loss


class CriticNet(nn.Module):
    def __init__(self,
                 input=(30 * 30 + 30 * 30 + 1 + 30),
                 output=1):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output)

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticNetLoss(nn.Module):
    def __init__(self):
        super(CriticNetLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        total_loss = self.loss(pred, target)
        return total_loss


if __name__ == "__main__":
    action_net = ActionNet()
    x = torch.rand(64, 1801)
    x = action_net(x)
    print(x.shape)
