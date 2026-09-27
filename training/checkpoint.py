from pathlib import Path
import torch

def save_checkpoint(model,optimizer,epoch,metrics,path,config):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict() if optimizer else None,'epoch':epoch,'metrics':metrics,'config':config},path)

def load_checkpoint(model,path,device='cpu',optimizer=None):
    ckpt=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and ckpt.get('optimizer_state'): optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt
