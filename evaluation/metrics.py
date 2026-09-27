import numpy as np

def mae(y_true,y_pred): return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))
def rmse(y_true,y_pred): return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))
def mape(y_true,y_pred,eps=1e-3):
    yt=np.asarray(y_true,dtype=float); yp=np.asarray(y_pred,dtype=float)
    threshold=max(eps,0.01*float(np.median(np.abs(yt))))
    valid=np.abs(yt)>threshold
    if not valid.any(): return float('nan')
    return float(np.mean(np.abs((yt[valid]-yp[valid])/np.abs(yt[valid])))*100)
def smape(y_true,y_pred,eps=1e-5):
    yt=np.asarray(y_true,dtype=float); yp=np.asarray(y_pred,dtype=float)
    denom=np.maximum(np.abs(yt)+np.abs(yp),eps)
    return float(np.mean(2*np.abs(yt-yp)/denom)*100)
def regression_metrics(y_true,y_pred):
    return {'MAE':mae(y_true,y_pred),'RMSE':rmse(y_true,y_pred),'MAPE_nonzero':mape(y_true,y_pred),'SMAPE':smape(y_true,y_pred)}
