import torch
class MLP(torch.nn.Module):
    def __init__(self, input_shape=(28,28)):
        super(MLP, self).__init__()

        self.act = torch.nn.Mish()

        self.flat_layer = torch.nn.Flatten()

        self.hl1 = torch.nn.Linear(input_shape[0]*input_shape[1], 120)
        self.hl2 = torch.nn.Linear(120, 72)
        self.hl3 = torch.nn.Linear(72, 48)
        self.out_layer = torch.nn.Linear(48, 10)

        self.drop1 = torch.nn.Dropout(p=0.3)
        self.drop2 = torch.nn.Dropout(p=0)
        self.drop3 = torch.nn.Dropout(p=0)

        self.bn1 = torch.nn.BatchNorm1d(self.hl1.out_features)
        self.bn2 = torch.nn.BatchNorm1d(self.hl2.out_features)
        self.bn3 = torch.nn.BatchNorm1d(self.hl3.out_features)

        # inicializacion de pesos
        self._init_weigths()

    def forward(self, x):
        out = self.flat_layer(x)
        out = self.drop1(self.act(self.bn1(self.hl1(out))))
        out = self.drop2(self.act(self.bn2(self.hl2(out))))
        out = self.drop3(self.act(self.bn3(self.hl3(out))))
        out = self.out_layer(out)
        return out
    
    def _init_weigths(self):
        torch.nn.init.kaiming_normal_(self.hl1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.hl2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.hl3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.out_layer.weight)

