import numpy as np

def clean_traffic(data):
    x=np.asarray(data,dtype=np.float32).copy()
    if not np.isfinite(x).all():
        raise ValueError('Non-finite values detected; refusing silent cleanup.')
    x[x < 0] = 0
    return x
