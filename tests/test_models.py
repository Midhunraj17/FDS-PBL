import torch
from models.gcn_gru import GCNGRU

def test_gcn_gru_shapes():
    m=GCNGRU(5,3,8,2); x=torch.randn(2,4,5,3); mask=torch.ones_like(x); a=torch.eye(5); o=m(x,mask,a); assert o['forecast'].shape==(2,2,5,3); assert o['imputation'].shape==x.shape
