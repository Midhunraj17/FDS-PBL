from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from preprocessing.loader import load_pems04
from preprocessing.masking import make_mask
from visualization.missingness_plot import save_missingness
from utils.config import load_config
cfg=load_config(); data,_,_=load_pems04(cfg['dataset']['root']); mask=make_mask((min(200,len(data)),data.shape[1],data.shape[2]),cfg['missingness']['default_rate'],seed=cfg['project']['seed']); save_missingness(mask,ROOT/'outputs/figures/missingness_heatmap.png'); print('[INFO] Demo outputs generated.')
