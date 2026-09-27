import torch

def predict(model,loader,adjacency,device):
    model.eval(); forecasts=[]; imputations=[]
    with torch.no_grad():
        for b in loader:
            out=model(b['x'].to(device),b['mask'].to(device),adjacency.to(device)); forecasts.append(out['forecast'].cpu());
            if 'imputation' in out: imputations.append(out['imputation'].cpu())
    return torch.cat(forecasts).numpy(), (torch.cat(imputations).numpy() if imputations else None)
