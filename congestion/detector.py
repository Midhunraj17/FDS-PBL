import numpy as np

def speed_threshold(training_speed,percentile=25): return float(np.percentile(training_speed,percentile))
def detect_congestion(speed,threshold): return (np.asarray(speed)<threshold).astype(np.int8)
