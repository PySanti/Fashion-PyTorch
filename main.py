import numpy as np
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


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

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, stratify=Y_test)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')


batch_size = 128
dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True)

mlp = MLP().to('cuda')
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.RMSprop(mlp.parameters(), lr=0.0001, momentum=0.9)
epochs = 100
batches_count = len(X_train)//batch_size + 1 if len(X_train)%batch_size !=0 else 0

for ep in range(epochs):
    train_loss = 0
    val_loss = 0
    for (X_train_batch, Y_train_batch) in dataloader:
        X_train_batch = X_train_batch.float()
        Y_train_batch = Y_train_batch
        outputs = mlp(X_train_batch)
        batch_loss = loss(outputs, Y_train_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()
        val_loss += loss(mlp(X_val), Y_val)

    mean_val_loss = val_loss/batches_count
    mean_train_loss = train_loss/batches_count
    print(f'Epoca actual : {ep}/{epochs}')
    print(f"\tBatches : {batches_count}")
    print(f'\tTrain loss : {mean_train_loss}')
    print(f'\tVal loss : {mean_val_loss}')
    print(f"\tDiff: {100-((1/val_loss)*100/(1/train_loss))}")





