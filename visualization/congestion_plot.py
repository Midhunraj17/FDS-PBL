import matplotlib.pyplot as plt
import numpy as np

def save_congestion(congestion,path):
    plt.figure(figsize=(10,4)); plt.imshow(np.asarray(congestion).T,aspect='auto',interpolation='nearest'); plt.title('Congestion State by Sensor and Time'); plt.xlabel('Time'); plt.ylabel('Sensor'); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
