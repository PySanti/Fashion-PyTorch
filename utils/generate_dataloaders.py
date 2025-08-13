from torch.utils.data import TensorDataset
import torch
import torchvision
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split


train_dataset = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,)
test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,)

X_train,Y_train = convert_dataset(train_dataset)
X_test,Y_test = convert_dataset(test_dataset)

X_train, X_val,Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')


def generate_dataloaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(X_test, Y_test),
            batch_size=batch_size,
            shuffle=True)
    return (train_loader, val_loader, test_loader)



