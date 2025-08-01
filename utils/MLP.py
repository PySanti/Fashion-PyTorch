import torch
class MLP(torch.nn.Module):
    def __init__(self, input_shape=(28,28)):
        super(MLP, self).__init__()
        self.hl1 = torch.nn.Linear(input_shape[0]*input_shape[1], 120)
        self.hl2 = torch.nn.Linear(120, 72)
        self.hl3 = torch.nn.Linear(72, 48)
        self.out_layer = torch.nn.Linear(48, 10)

    def forward(self, x):
        out = self.hl1(x)
        out = torch.nn.ReLU(out)
        out = self.hl2(out)
        out = torch.nn.ReLU(out)
        out = self.hl3(out)
        out = torch.nn.ReLU(out)
        out = self.out_layer(out)
        out = torch.nn.Softmax(out)
        return out



