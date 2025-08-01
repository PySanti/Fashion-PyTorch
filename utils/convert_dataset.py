import numpy as np
import torch
def convert_dataset(set_):
    X_data = []
    Y_data = []
    for (img, tag) in set_:
        X_data.append(np.array(img))
        Y_data.append(tag)
    return torch.tensor(np.array(X_data)), torch.tensor(np.array(Y_data)).unsqueeze(1)
