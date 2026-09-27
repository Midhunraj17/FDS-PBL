import matplotlib.pyplot as plt

def save_missingness(mask,path):
    plt.figure(figsize=(10,4)); plt.imshow(mask[:min(200,len(mask)),:,0].T,aspect='auto',interpolation='nearest'); plt.title('Simulated Missingness (1=Observed, 0=Missing)'); plt.xlabel('Time'); plt.ylabel('Sensor'); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
