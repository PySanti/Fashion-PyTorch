from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
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


train_set = MyDataset(X_train, Y_train)
dataloader = torch.utils.data.DataLoader(dataset=train_set,batch_size=64,shuffle=True)

for (X_train_batch, Y_train_batch) in dataloader:
    print("Mostrando shape de batches cargados por dataloader")
    print(X_train_batch.shape, X_train_batch.type())
    print(Y_train_batch.shape, Y_train_batch.type())

