import numpy as np
from preprocessing.masking import make_mask,apply_mask

def test_mask_is_deterministic_and_approximately_requested():
    a=make_mask((100,10,3),.3,seed=7); b=make_mask((100,10,3),.3,seed=7); assert np.array_equal(a,b); assert abs((a==0).mean()-.3)<.05

def test_apply_mask():
    x=np.ones((2,2,1)); m=np.array([[[1],[0]],[[1],[1]]]); assert apply_mask(x,m)[0,1,0]==0
