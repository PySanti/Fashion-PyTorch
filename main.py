import numpy as np
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split


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


BATCH_SIZE  = 128
EPOCHS      = 100

# loaders
train_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True)
val_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
        shuffle=True)

mlp = MLP().to('cuda')
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.RMSprop(mlp.parameters(), lr=0.0001, momentum=0.9)

for ep in range(EPOCHS):

    # entrenamiento

    train_loss = np.array([])
    mlp.train()

    for (X_train_batch, Y_train_batch) in train_loader:
        X_train_batch = X_train_batch.float()
        outputs = mlp(X_train_batch)
        batch_loss = loss(outputs, Y_train_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss = np.append(train_loss, batch_loss.item())


    # validacion
    mlp.eval()
    val_loss = np.array([])

    with torch.no_grad():
        for (X_val_batch, Y_val_batch) in val_loader:
            X_val_batch = X_val_batch.float()
            val_loss = np.append(val_loss, loss(mlp(X_val_batch), Y_val_batch).item())

    print(f'Epoca actual : {ep}/{EPOCHS}')
    print(f"\tTrain batches : {len(train_loss)}")
    print(f'\tTrain loss : {train_loss.mean()}')
    print(f'\tVal loss : {val_loss.mean()}')
    print(f"\tDiff: {((val_loss.mean())*100)/(train_loss.mean())-100}")

