from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from preprocessing.loader import load_pems04
from preprocessing.cleaner import clean_traffic
from preprocessing.splitter import chronological_split
from preprocessing.windowing import TrafficWindowDataset
from preprocessing.graph_builder import build_normalized_adjacency
from preprocessing.normalization import FeatureScaler
from models.gcn_gru import GCNGRU
from models.gru_baseline import GRUBaseline
from training.checkpoint import load_checkpoint
from evaluation.metrics import regression_metrics
from evaluation.evaluate_imputation import evaluate_imputation
from congestion.detector import speed_threshold,detect_congestion
from congestion.propagation import propagation_events
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device
from utils.io import save_json
from visualization.congestion_plot import save_congestion

cfg=load_config(); set_seed(cfg['project']['seed']); device=get_device()
data,edges,_=load_pems04(cfg['dataset']['root']); data=clean_traffic(data)
tr,va,te=chronological_split(data,cfg['split']['train'],cfg['split']['validation'])
scaler=FeatureScaler().fit(tr); tr_s=scaler.transform(tr); te_s=scaler.transform(te)
adj=build_normalized_adjacency(edges,data.shape[1]).to(device)
h=cfg['dataset']['history']; p=cfg['dataset']['horizon']; n=data.shape[1]; f=data.shape[2]

# The final model is trained with the default 30% missingness and evaluated at all requested rates.
model=GCNGRU(n,f,cfg['model']['hidden_dim'],p,cfg['model']['dropout']).to(device)
load_checkpoint(model,ROOT/'outputs/models/best_model.pt',device); model.eval()
rows=[]; imputation_rows=[]
for rate in cfg['evaluation']['missing_rates']:
    ds=TrafficWindowDataset(te_s,h,p,rate,cfg['project']['seed']+20000+int(rate*1000),1,cfg['training']['max_test_samples'])
    dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False)
    preds=[]; ys=[]; imps=[]; originals=[]; masks=[]
    with torch.no_grad():
        for b in dl:
            o=model(b['x'].to(device),b['mask'].to(device),adj)
            preds.append(o['forecast'].cpu().numpy()); ys.append(b['y'].numpy()); imps.append(o['imputation'].cpu().numpy()); originals.append(b['original'].numpy()); masks.append(b['mask'].numpy())
    pred_s=np.concatenate(preds); y_s=np.concatenate(ys); imp_s=np.concatenate(imps); orig_s=np.concatenate(originals); masks=np.concatenate(masks)
    pred=scaler.inverse_transform(pred_s); y=scaler.inverse_transform(y_s); imp=scaler.inverse_transform(imp_s); orig=scaler.inverse_transform(orig_s)
    rows.append({'model':'GCN-GRU','missing_rate':rate,**regression_metrics(y[:,:,:,0],pred[:,:,:,0])})
    imputation_rows.append({'model':'GCN-GRU','missing_rate':rate,**evaluate_imputation(orig,imp,masks)})
    if rate == cfg['missingness']['default_rate']:
        np.savez_compressed(ROOT/'outputs/predictions/test_predictions.npz',y=y,pred=pred,imputation=imp,mask=masks)

# Fair persistence baseline, evaluated on each masking condition.
for rate in cfg['evaluation']['missing_rates']:
    ds=TrafficWindowDataset(te_s,h,p,rate,cfg['project']['seed']+20000+int(rate*1000),1,cfg['training']['max_test_samples'])
    latest=ds.data[[int(i)+h-1 for i in ds.indices]][:,None,:,:]
    pers_s=np.repeat(latest,p,axis=1); y_s=np.stack([ds[i]['y'].numpy() for i in range(len(ds))])
    y=scaler.inverse_transform(y_s); pers=scaler.inverse_transform(pers_s)
    rows.append({'model':'Persistence','missing_rate':rate,**regression_metrics(y[:,:,:,0],pers[:,:,:,0])})

# Temporal baseline trained once at default missingness, then evaluated at all rates.
gru=GRUBaseline(n,f,cfg['model']['hidden_dim'],p).to(device); opt=torch.optim.Adam(gru.parameters(),lr=cfg['training']['learning_rate']); default_rate=cfg['missingness']['default_rate']
gru_ds=TrafficWindowDataset(te_s,h,p,default_rate,cfg['project']['seed']+20000+int(default_rate*1000),1,cfg['training']['max_test_samples']); gru_dl=DataLoader(gru_ds,batch_size=cfg['training']['batch_size'],shuffle=True)
gru.train()
for _ in range(1):
    for b in gru_dl:
        opt.zero_grad(); out=gru(b['x'].to(device)); loss=torch.mean(torch.abs(out['forecast']-b['y'].to(device))); loss.backward(); opt.step()
gru.eval()
for rate in cfg['evaluation']['missing_rates']:
    ds=TrafficWindowDataset(te_s,h,p,rate,cfg['project']['seed']+20000+int(rate*1000),1,cfg['training']['max_test_samples']); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False)
    gp=[]; ys=[]
    with torch.no_grad():
        for b in dl: gp.append(gru(b['x'].to(device))['forecast'].cpu().numpy()); ys.append(b['y'].numpy())
    gp=scaler.inverse_transform(np.concatenate(gp)); yy=scaler.inverse_transform(np.concatenate(ys))
    rows.append({'model':'GRU','missing_rate':rate,**regression_metrics(yy[:,:,:,0],gp[:,:,:,0])})

comparison=pd.DataFrame(rows); comparison.to_csv(ROOT/'outputs/metrics/comparison.csv',index=False)
imputation=pd.DataFrame(imputation_rows); imputation.to_csv(ROOT/'outputs/metrics/imputation_comparison.csv',index=False)
# Congestion is defined on raw speed values from the training period.
threshold=speed_threshold(tr[:,:,1],cfg['congestion']['speed_percentile']); cong=detect_congestion(te[:min(500,len(te)),:,1],threshold); save_congestion(cong,ROOT/'outputs/figures/congestion_heatmap.png'); events=propagation_events(cong,edges); save_json({'speed_threshold':threshold,'events':events[:100]},ROOT/'outputs/metrics/propagation.json')

summary={'dataset':'PEMS04','seed':cfg['project']['seed'],'split':cfg['split'],'history_steps':h,'forecast_horizon':p,'missing_rates':cfg['evaluation']['missing_rates'],'forecast_comparison':rows,'imputation':imputation_rows,'device':str(device),'speed_congestion_threshold':threshold}
save_json(summary,ROOT/'outputs/metrics/experiment_summary.json'); save_json({'forecast_comparison':rows,'imputation':imputation_rows},ROOT/'outputs/metrics/metrics.json')
print(json.dumps(summary,indent=2))
