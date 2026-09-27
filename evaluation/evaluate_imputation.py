import numpy as np
from .metrics import regression_metrics

def evaluate_imputation(original,reconstructed,mask):
    missing=(np.asarray(mask)==0)
    if not missing.any(): return {'MAE':0.0,'RMSE':0.0,'MAPE':0.0}
    return regression_metrics(np.asarray(original)[missing],np.asarray(reconstructed)[missing])
