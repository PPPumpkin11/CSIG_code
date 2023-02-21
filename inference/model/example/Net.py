import torch.nn as nn
import torch

# 固定检测60帧内的动作,这里只是给了一个简单的神经网络，可以自行用更高效的模型替代它
class Net(nn.Module):
    def __init__(self, ActionLength):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(ActionLength * 24 * 6, 5120, bias=False)
        self.output = nn.Linear(5120, 10, bias=False)
        self.ActionLength = ActionLength

    def forward(self, x):
        x = x.view(-1, self.ActionLength * 24 * 6)
        x = self.hidden1(x)
        y = self.output(x)

        return y