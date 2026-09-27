import numpy as np
import torch
from torch.utils.data import Dataset

class TrafficWindowDataset(Dataset):
    def __init__(self, data, history=12, horizon=3, mask_rate=0.3, seed=42, stride=1, max_samples=None):
        self.data=data.astype(np.float32); self.history=history; self.horizon=horizon
        self.indices=np.arange(0, len(data)-history-horizon+1, stride)
        if max_samples and len(self.indices)>max_samples: self.indices=self.indices[:max_samples]
        self.mask_rate=mask_rate; self.seed=seed
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        start=int(self.indices[i]); x=self.data[start:start+self.history]; y=self.data[start+self.history:start+self.history+self.horizon]
        rng=np.random.default_rng(self.seed+i)
        mask=(rng.random(x.shape)>=self.mask_rate).astype(np.float32)
        xm=x*mask
        return {'x':torch.from_numpy(xm), 'mask':torch.from_numpy(mask), 'y':torch.from_numpy(y), 'original':torch.from_numpy(x)}
