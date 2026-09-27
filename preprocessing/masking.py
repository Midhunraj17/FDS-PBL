import numpy as np

def make_mask(shape, missing_rate=0.3, mode='random', seed=42):
    if not 0 <= missing_rate < 1: raise ValueError('missing_rate must be in [0,1).')
    rng=np.random.default_rng(seed)
    if mode=='random':
        return (rng.random(shape) >= missing_rate).astype(np.float32)
    if mode=='block':
        t,n,f=shape; m=np.ones(shape,dtype=np.float32); target=int(t*n*missing_rate)
        while (m==0).sum()/f < target:
            node=int(rng.integers(n)); start=int(rng.integers(max(1,t-12))); length=int(rng.integers(1,13)); m[start:min(t,start+length),node,:]=0
        return m
    raise ValueError(f'Unknown missingness mode: {mode}')

def apply_mask(x, mask): return x*mask
