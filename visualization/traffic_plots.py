import matplotlib.pyplot as plt
import numpy as np

def save_actual_predicted(actual,pred,path,title='Actual vs Predicted'):
    plt.figure(figsize=(10,4)); plt.plot(actual,label='Actual'); plt.plot(pred,label='Predicted'); plt.title(title); plt.xlabel('Forecast step'); plt.ylabel('Traffic value'); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()

def save_loss(history,path):
    plt.figure(figsize=(8,4)); plt.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Train'); plt.plot([x['epoch'] for x in history],[x['val_loss'] for x in history],label='Validation'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training History'); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
