from .metrics import regression_metrics

def evaluate_forecast(y_true,y_pred): return regression_metrics(y_true,y_pred)
