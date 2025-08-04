import torch
def accuracy(predictions, targets):
    """
        Calcula el accuracy entre outputs y labels
    """

    _, predicted = torch.max(predictions, 1)
    return ((predicted == targets).sum().item()/targets.shape[0])*100
