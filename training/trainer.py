import torch
from pathlib import Path
from .checkpoint import save_checkpoint
from models.losses import multitask_loss

class Trainer:
    def __init__(self,model,optimizer,device,config): self.model=model; self.optimizer=optimizer; self.device=device; self.config=config
    def fit(self,train_loader,val_loader,adjacency,checkpoint_path):
        best=float('inf'); patience=0; history=[]
        self.model.to(self.device); adjacency=adjacency.to(self.device)
        for epoch in range(1,self.config['epochs']+1):
            self.model.train(); train_loss=0.0
            for batch in train_loader:
                x=batch['x'].to(self.device); m=batch['mask'].to(self.device); y=batch['y'].to(self.device); orig=batch['original'].to(self.device)
                self.optimizer.zero_grad(); out=self.model(x,m,adjacency); loss,parts=multitask_loss(out,y,orig,m); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0); self.optimizer.step(); train_loss+=loss.item()
            train_loss/=max(1,len(train_loader)); val_loss=self.validate(val_loader,adjacency); history.append({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss})
            if val_loss<best:
                best=val_loss; patience=0; save_checkpoint(self.model,self.optimizer,epoch,{'val_loss':val_loss},checkpoint_path,self.config)
            else: patience+=1
            if patience>=self.config['patience']: break
        return history
    def validate(self,loader,adjacency):
        self.model.eval(); total=0.0
        with torch.no_grad():
            for batch in loader:
                x=batch['x'].to(self.device); m=batch['mask'].to(self.device); y=batch['y'].to(self.device); orig=batch['original'].to(self.device)
                out=self.model(x,m,adjacency); loss,_=multitask_loss(out,y,orig,m); total+=loss.item()
        return total/max(1,len(loader))
