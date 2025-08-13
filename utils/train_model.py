import numpy as np
import torch
from utils.accuracy import accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.MLP import MLP
from ray import tune

from utils.generate_dataloaders import generate_dataloaders


def train_model(config):

    mlp = MLP(
            hidden_sizes=[config["l1_size"], config["l2_size"], config["l3_size"]],
            dropout_rates=[config["l1_drop"], config["l2_drop"], config["l3_drop"]]
            ).to('cuda')

    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(
            mlp.parameters(), 
            lr=config['base_lr'], 
            weight_decay=config['l2_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['t_max'])   

    (train_loader, val_loader, _) = generate_dataloaders(config["batch_size"])


    for ep in range(config['max_epochs']+1):

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

        scheduler.step()



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



        tune.report({"accuracy": correct_val_samples.mean()})


#        print(f'Epoca actual : {ep}/{config["max_epochs"]}')
#        print(f"\tTrain batches : {len(batches_train_loss)}")
#        print(f'\tTrain loss : {batches_train_loss.mean():.2f}')
#        print(f"\tVal acc: {correct_val_samples.mean():.2f}")
#        print(f'\tVal loss : {batches_val_loss.mean():.2f}')


