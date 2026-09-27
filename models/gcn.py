import torch
from torch import nn

class GCNLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__(); self.linear=nn.Linear(in_features,out_features)
    def forward(self,x,adj):
        # x [B,T,N,F], adj [N,N]
        b,t,n,f=x.shape
        h=x.reshape(b*t,n,f)
        h=torch.matmul(adj.unsqueeze(0),h)
        h=self.linear(h)
        return h.reshape(b,t,n,-1)
