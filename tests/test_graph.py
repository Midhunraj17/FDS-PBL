import numpy as np, torch
from preprocessing.graph_builder import build_normalized_adjacency

def test_graph_shape_and_symmetry():
    a=build_normalized_adjacency(np.array([[0,1],[1,2]]),3); assert a.shape==(3,3); assert torch.allclose(a,a.T); assert torch.diag(a).min()>0
