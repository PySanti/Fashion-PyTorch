import torch
class MLP(torch.nn.Module):
    def __init__(self, input_shape=(28,28), hiden_sizes=[250, 150, 100], dropout_rate=0.2):
        super(MLP, self).__init__()
        self.flat_layer = torch.nn.Flatten()
        self.layers = torch.nn.ModuleList()
        current_input_size = input_shape[0]*input_shape[1]

        for size in hiden_sizes:
            self.layers.append(torch.nn.Linear(current_input_size, size))
            self.layers.append(torch.nn.BatchNorm1d(size))
            self.layers.append(torch.nn.Mish())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))
            current_input_size = size
        self.out_layer = torch.nn.Linear(current_input_size, 10)
        # inicializacion de pesos
        self._initialize_weights()

    def forward(self, x):
        out = self.flat_layer(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_layer(out)
        return out
    
    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight,nonlinearity='leaky_relu')

