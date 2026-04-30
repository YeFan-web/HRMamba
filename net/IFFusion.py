import math
import torch
from torch import nn
from mamba_ssm import Mamba
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out





class IFFusion(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(IFFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.mamba = Mamba(
            d_model=channel,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv1d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input2,input):
        input = input.permute(0, 3, 1, 2)
        input2 = input2.permute(0, 3, 1, 2)
        x = self.avg_pool(input)
        x1 = self.fc(x.squeeze(-1))
        x = x.squeeze(-1).transpose(-1, -2)
        x2 = self.mamba(x)#.transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        out = (input2*out).permute(0, 2, 3, 1)
        return out

if __name__ == '__main__':
    input = torch.rand(1,256,256,64).to('cuda')
    input2 = torch.rand(1, 256, 256,64).to('cuda')
    A = IFFusion(channel=64).to('cuda')
    y = A(input,input2)
    print(y.size())


