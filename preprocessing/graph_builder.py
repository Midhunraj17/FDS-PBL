import numpy as np
import torch

def build_normalized_adjacency(edges, num_nodes, self_loops=True):
    a=np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for s,d in edges.astype(int): a[s,d]=1; a[d,s]=1
    if self_loops: np.fill_diagonal(a,1)
    deg=a.sum(1); inv=np.power(deg,-0.5,where=deg>0,out=np.zeros_like(deg)); A=(inv[:,None]*a)*inv[None,:]
    return torch.tensor(A,dtype=torch.float32)
