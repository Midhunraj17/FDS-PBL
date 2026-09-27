import torch

def masked_l1(pred,target,mask=None):
    if mask is None: return torch.mean(torch.abs(pred-target))
    valid=mask.float(); return (torch.abs(pred-target)*valid).sum()/valid.sum().clamp_min(1.0)

def multitask_loss(outputs,y,original,missing_mask,lambda_imputation=1.0):
    forecast_loss=torch.mean(torch.abs(outputs['forecast']-y))
    missing=(1.0-missing_mask)
    imputation_loss=masked_l1(outputs['imputation'],original,missing)
    return forecast_loss+lambda_imputation*imputation_loss, {'forecast_loss':float(forecast_loss.detach()),'imputation_loss':float(imputation_loss.detach())}
