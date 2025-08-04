import numpy as np
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
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

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, stratify=Y_test)

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
        dataset=MyDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True)
val_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
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

    with torch.no_grad():
        for (X_val_batch, Y_val_batch) in val_loader:
            X_val_batch = X_val_batch.float()
            batches_val_loss = np.append(batches_val_loss, loss(mlp(X_val_batch), Y_val_batch).item())


    print(f'Epoca actual : {ep}/{EPOCHS}')
    print(f"\tTrain batches : {len(batches_train_loss)}")
    print(f'\tTrain loss : {batches_train_loss.mean()}')
    print(f'\tVal loss : {batches_val_loss.mean()}')
    print(f"\tDiff: {((batches_val_loss.mean())*100)/(batches_train_loss.mean())-100}")

    val_losses = np.append(val_losses, batches_val_loss.mean())
    train_losses = np.append(train_losses, batches_train_loss.mean())

plot_loss(train_losses, val_losses)
