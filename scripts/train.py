from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from preprocessing.loader import load_pems04
from preprocessing.cleaner import clean_traffic
from preprocessing.splitter import chronological_split
from preprocessing.windowing import TrafficWindowDataset
from preprocessing.graph_builder import build_normalized_adjacency
from preprocessing.normalization import FeatureScaler
from models.gcn_gru import GCNGRU
from training.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device
from utils.io import save_json
from visualization.traffic_plots import save_loss

cfg=load_config(); set_seed(cfg['project']['seed']); device=get_device(); print('[INFO] Device:',device)
data,edges,info=load_pems04(cfg['dataset']['root']); data=clean_traffic(data)
tr,va,te=chronological_split(data,cfg['split']['train'],cfg['split']['validation'])
scaler=FeatureScaler().fit(tr); tr=scaler.transform(tr); va=scaler.transform(va); te=scaler.transform(te); scaler.save(ROOT/'outputs/artifacts/scaler.pkl')
np_adj=build_normalized_adjacency(edges,data.shape[1]); torch.save(np_adj,ROOT/'outputs/artifacts/graph_adjacency.pt')
h=cfg['dataset']['history']; p=cfg['dataset']['horizon']; stride=cfg['dataset']['stride']; tc=cfg['training']
train_ds=TrafficWindowDataset(tr,h,p,cfg['missingness']['default_rate'],cfg['project']['seed'],stride,tc['max_train_samples']); val_ds=TrafficWindowDataset(va,h,p,cfg['missingness']['default_rate'],cfg['project']['seed']+10000,1,tc['max_val_samples'])
train_dl=DataLoader(train_ds,batch_size=tc['batch_size'],shuffle=True); val_dl=DataLoader(val_ds,batch_size=tc['batch_size'],shuffle=False)
model=GCNGRU(data.shape[1],data.shape[2],cfg['model']['hidden_dim'],p,cfg['model']['dropout']); opt=torch.optim.Adam(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
trainer=Trainer(model,opt,device,{**tc,'epochs':tc['epochs'],'patience':tc['patience']}); hist=trainer.fit(train_dl,val_dl,np_adj,'outputs/models/best_model.pt'); save_json(hist,ROOT/'outputs/metrics/training_history.json'); save_loss(hist,ROOT/'outputs/figures/training_loss.png')
print('[INFO] Training complete. Best checkpoint saved.')
