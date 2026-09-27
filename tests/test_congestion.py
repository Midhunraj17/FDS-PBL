import numpy as np
from congestion.detector import speed_threshold,detect_congestion

def test_congestion_detector():
    t=speed_threshold(np.array([10,20,30,40]),50); c=detect_congestion(np.array([5,60]),t); assert c.tolist()==[1,0]
