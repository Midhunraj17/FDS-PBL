import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class FeatureScaler:
    def __init__(self): self.scaler=StandardScaler()
    def fit(self, x):
        self.scaler.fit(x.reshape(-1,x.shape[-1])); return self
    def transform(self, x):
        s=x.shape; return self.scaler.transform(x.reshape(-1,s[-1])).reshape(s).astype(np.float32)
    def inverse_transform(self, x):
        s=x.shape; return self.scaler.inverse_transform(x.reshape(-1,s[-1])).reshape(s).astype(np.float32)
    def save(self,path): joblib.dump(self.scaler,path)
    def load(self,path): self.scaler=joblib.load(path); return self
