import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),      
    transforms.Normalize((0.5,), (0.5,)) # range [-1, -1]
])
class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        '''
        sfilename : sprites file
        lfilename : label file 
        null_context : do we need context for conditional generation 
        '''
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabels_shape = self.slabels.shape
        
    def __len__(self):
        return len(self.sprites)
    
    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                # set null value of no null_context
                label = torch.tensor(0).to(torch.int64)
            else:
                # or set the available label
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        
        return (image, label)
    
    def getshapes(self):
        return self.sprites_shape, self.slabels_shape



