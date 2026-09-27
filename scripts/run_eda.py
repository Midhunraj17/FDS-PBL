from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from preprocessing.loader import load_pems04
from preprocessing.cleaner import clean_traffic
from visualization.eda import generate_eda
from visualization.network_plot import save_network
from utils.config import load_config
cfg=load_config(); data,edges,_=load_pems04(cfg['dataset']['root']); data=clean_traffic(data)
generate_eda(data,ROOT/'outputs/figures'); save_network(edges,data.shape[1],ROOT/'outputs/figures/network_graph.png'); print('[INFO] EDA figures generated.')
