import torchvision
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split


def load_base_datasets():
    train_dataset = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,)

    X_train,Y_train = convert_dataset(train_dataset)
    X_test,Y_test = convert_dataset(test_dataset)

    X_train, X_val,Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


