from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from preprocessing.loader import load_pems04
from preprocessing.cleaner import clean_traffic
from preprocessing.normalization import FeatureScaler
from preprocessing.graph_builder import build_normalized_adjacency
from utils.config import load_config
from utils.io import save_json
import numpy as np
cfg=load_config(); data,edges,info=load_pems04(cfg['dataset']['root']); data=clean_traffic(data)
scaler=FeatureScaler().fit(data[:int(len(data)*cfg['split']['train'])]); scaler.save(ROOT/'outputs/artifacts/scaler.pkl')
np.save(ROOT/'outputs/artifacts/graph_adjacency.npy',build_normalized_adjacency(edges,data.shape[1]).numpy())
save_json(info,ROOT/'outputs/artifacts/dataset_info.json'); print('[INFO] Prepared PEMS04:',info)
