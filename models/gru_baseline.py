import torch
from torch import nn

class GRUBaseline(nn.Module):
    """Temporal-only baseline using one shared GRU across sensor sequences."""
    def __init__(self, num_nodes, num_features, hidden_dim=16, horizon=3):
        super().__init__()
        self.num_nodes=num_nodes; self.num_features=num_features; self.horizon=horizon
        self.gru=nn.GRU(num_features, hidden_dim, batch_first=True)
        self.head=nn.Linear(hidden_dim,horizon*num_features)
    def forward(self,x,mask=None,adjacency=None):
        b,t,n,f=x.shape
        seq=x.permute(0,2,1,3).reshape(b*n,t,f)
        h,_=self.gru(seq)
        out=self.head(h[:,-1]).reshape(b,n,self.horizon,f).permute(0,2,1,3)
        return {'forecast':out}
