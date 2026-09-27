import numpy as np
from evaluation.metrics import mae,rmse,mape

def test_metrics_known_values():
    y=np.array([1.,2.,3.]); p=np.array([1.,3.,2.]); assert mae(y,p)==2/3; assert round(rmse(y,p),6)==round((2/3)**0.5,6)
def test_mape_handles_zero(): assert np.isfinite(mape(np.array([0.,1.]),np.array([0.1,1.])))
