from pathlib import Path
import numpy as np
import pandas as pd
from .validator import validate_traffic_array, validate_edges

def load_pems04(root='data/raw'):
    base=Path(root)/'PEMS04'
    npz=base/'PEMS04.npz'; csv=base/'PEMS04.csv'
    if not npz.exists() or not csv.exists():
        raise FileNotFoundError(f'PEMS04 files not found under {base}')
    data=np.load(npz)['data'].astype(np.float32)
    edges=pd.read_csv(csv)
    edge_array=edges.iloc[:,:2].to_numpy(dtype=np.int64)
    info=validate_traffic_array(data,307,3)
    validate_edges(edge_array, data.shape[1])
    return data, edge_array, info
