import numpy as np

def traffic_activity_proxy(flow,speed,occupancy):
    def norm(x):
        x=np.asarray(x,float); lo,hi=np.nanpercentile(x,5),np.nanpercentile(x,95); return np.clip((x-lo)/(hi-lo+1e-8),0,1)
    # Proxy only: combines observable traffic activity signals; not OD demand.
    return 0.5*norm(flow)+0.25*norm(occupancy)+0.25*(1-norm(speed))
