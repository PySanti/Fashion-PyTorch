from torch.utils.data import TensorDataset
import torch

def generate_dataloaders(batch_size, train_dataset, val_dataset, test_dataset=None):
    test_loader = None
    X_train,Y_train = train_dataset
    X_val, Y_val = val_dataset
    train_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            dataset=TensorDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=True)
    if test_dataset:
        X_test, Y_test = test_dataset
        test_loader = torch.utils.data.DataLoader(
                dataset=TensorDataset(X_test, Y_test),
                batch_size=batch_size,
                shuffle=True)
    return (train_loader, val_loader, test_loader)



