import os
import numpy as np
import torch
from utils.accuracy import accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.MLP import MLP
from ray import tune
import os
import tempfile
from pathlib import Path
import torch
from ray import tune
from ray import train
from ray.tune import Checkpoint
import ray.cloudpickle as pickle

from utils.generate_dataloaders import generate_dataloaders

def train_model(config, train_dataset, val_dataset):
    mlp = MLP(
            hidden_sizes=[config["l1_size"], config["l2_size"], config["l3_size"]],
            dropout_rates=[config["l1_drop"], config["l2_drop"], config["l3_drop"]]
            ).to('cuda')
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(mlp.parameters(),lr=config['base_lr'],weight_decay=config['l2_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['t_max'])   
    (train_loader, val_loader, _) = generate_dataloaders(config["batch_size"], train_dataset, val_dataset)
    for ep in range(config['max_epochs']+1):
        # entrenamiento
        batches_train_loss = []
        mlp.train()
        for (X_train_batch, Y_train_batch) in train_loader:
            X_train_batch = X_train_batch.float().to('cuda')
            Y_train_batch = Y_train_batch.to('cuda')
            outputs = mlp(X_train_batch)
            batch_loss = loss(outputs, Y_train_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batches_train_loss.append(batch_loss.item())
        scheduler.step()
        # validacion
        mlp.eval()
        batches_val_loss = []
        correct_val_samples = []

        with torch.no_grad():
            for (X_val_batch, Y_val_batch) in val_loader:
                X_val_batch = X_val_batch.float().to('cuda')
                Y_val_batch = Y_val_batch.to('cuda')
                outputs = mlp(X_val_batch)
                batches_val_loss.append(loss(outputs, Y_val_batch).item())
                correct_val_samples.append(accuracy(outputs, Y_val_batch))
    # se almacena un checkpoint (estado del modelo) al final de cada trial
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        torch.save(
            mlp.state_dict(),
            os.path.join(temp_checkpoint_dir, "model.pt"),
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        tune.report({"accuracy": np.mean(correct_val_samples)}, checkpoint=checkpoint)
