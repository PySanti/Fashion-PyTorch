import numpy as np
from sklearn.externals.array_api_compat.numpy import test
import torch
from utils.accuracy import accuracy
from utils.data_loading import *

def train_model(mlp,  base_lr, l2_rate, EPOCHS=40):
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(mlp.parameters(), lr=base_lr, weight_decay=l2_rate)
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

    return train_losses, val_losses



