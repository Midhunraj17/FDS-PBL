import numpy as np, torch
from preprocessing.windowing import TrafficWindowDataset
from preprocessing.graph_builder import build_normalized_adjacency
from models.gcn_gru import GCNGRU

def test_synthetic_pipeline_forward_and_backward():
    data=np.random.default_rng(1).random((40,5,3)).astype('float32'); ds=TrafficWindowDataset(data,4,2,.3,1,1,4); b=ds[0]; m=GCNGRU(5,3,8,2); a=build_normalized_adjacency(np.array([[0,1],[1,2],[2,3],[3,4]]),5); o=m(b['x'][None],b['mask'][None],a); loss=(o['forecast']-b['y'][None]).abs().mean(); loss.backward(); assert np.isfinite(loss.item())
