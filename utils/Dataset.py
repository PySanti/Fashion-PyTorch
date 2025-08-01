from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __getitem__(self,index):
        # Get one item from the dataset
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
