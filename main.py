import numpy as np
from torch.utils.data import TensorDataset
from sklearn.externals.array_api_compat.numpy import test
import torch
from utils.MLP import MLP
from utils.accuracy import accuracy
from utils.plot_loss import plot_loss
from utils.data_loading import *
from utils.train_model import train_model


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')


mlp = MLP().to('cuda')
train_losses, val_losses = train_model(mlp, base_lr=1e-4, l2_rate=1e-2)

# test accuracy
test_acc = np.array([])
for (X_test_batch, Y_test_batch) in test_loader:
    outputs = mlp(X_test_batch)
    test_acc = np.append(test_acc, accuracy(outputs, Y_test_batch))
print(f"Test acc : {test_acc.mean()}")
plot_loss(train_losses, val_losses)
