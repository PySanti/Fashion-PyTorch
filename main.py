import numpy as np
from torch.utils.data import TensorDataset
from pyarrow import Tensor, output_stream
from sklearn.externals.array_api_compat.numpy import test
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
from utils.accuracy import accuracy
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split
from utils.plot_loss import plot_loss


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
)

# Cargar el dataset de prueba
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
)

X_train,Y_train = convert_dataset(train_dataset)
X_test,Y_test = convert_dataset(test_dataset)

X_train, X_val,Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')


BATCH_SIZE  = 128
EPOCHS      = 40

# loaders
train_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True)
val_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(X_val, Y_val),
        batch_size=128,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dataset=TensorDataset(X_test, Y_test),
        batch_size=128,
        shuffle=True)



mlp = MLP().to('cuda')
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001, weight_decay=1e-2)


val_losses = np.array([])
train_losses = np.array([])

for ep in range(EPOCHS+1):

    # entrenamiento

    batches_train_loss = np.array([])
    mlp.train()

    for (X_train_batch, Y_train_batch) in train_loader:
        X_train_batch = X_train_batch.float()
        outputs = mlp(X_train_batch)
        batch_loss = loss(outputs, Y_train_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batches_train_loss = np.append(batches_train_loss, batch_loss.item())


    # validacion
    mlp.eval()
    batches_val_loss = np.array([])
    correct_val_samples = np.array([])

    with torch.no_grad():
        for (X_val_batch, Y_val_batch) in val_loader:
            X_val_batch = X_val_batch.float()
            outputs = mlp(X_val_batch)
            batches_val_loss = np.append(batches_val_loss, loss(outputs, Y_val_batch).item())
            correct_val_samples = np.append(correct_val_samples, accuracy(outputs, Y_val_batch))

    print(f'Epoca actual : {ep}/{EPOCHS}')
    print(f"\tTrain batches : {len(batches_train_loss)}")
    print(f'\tTrain loss : {batches_train_loss.mean():.2f}')
    print(f"\tVal acc: {correct_val_samples.mean():.2f}")
    print(f'\tVal loss : {batches_val_loss.mean():.2f}')
    print(f"\tDiff: {((batches_val_loss.mean())*100)/(batches_train_loss.mean())-100:.2f}")

    val_losses = np.append(val_losses, batches_val_loss.mean())
    train_losses = np.append(train_losses, batches_train_loss.mean())


# test accuracy
test_acc = np.array([])
for (X_test_batch, Y_test_batch) in test_loader:
    outputs = mlp(X_test_batch)
    test_acc = np.append(test_acc, accuracy(outputs, Y_test_batch))
print(f"Test acc : {test_acc.mean()}")
plot_loss(train_losses, val_losses)

