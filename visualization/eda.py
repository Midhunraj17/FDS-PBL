from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_eda(data,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    labels=['Flow','Speed','Occupancy']
    fig,axs=plt.subplots(3,1,figsize=(10,8))
    sample=data[:min(len(data),288),:5]
    for f,ax in enumerate(axs): ax.plot(sample[:,:,f].mean(axis=1)); ax.set_title(labels[f]); ax.set_xlabel('5-minute step')
    fig.tight_layout(); fig.savefig(out/'traffic_over_time.png',dpi=140); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3))
    for f,ax in enumerate(axs): ax.hist(data[:,:,f].ravel(),bins=40); ax.set_title(labels[f])
    fig.tight_layout(); fig.savefig(out/'feature_distributions.png',dpi=140); plt.close(fig)
    flat=data.reshape(-1,3); corr=np.corrcoef(flat.T)
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(corr,vmin=-1,vmax=1); ax.set_xticks(range(3),labels); ax.set_yticks(range(3),labels); ax.set_title('Feature Correlation'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'correlation_matrix.png',dpi=140); plt.close(fig)
