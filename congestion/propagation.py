import numpy as np

def propagation_events(congestion,edges,max_events=200):
    c=np.asarray(congestion); events=[]
    edge_set=[(int(a),int(b)) for a,b in edges]
    for t in range(1,c.shape[0]):
        newly=np.where((c[t]==1)&(c[t-1]==0))[0]
        for node in newly:
            neighbors=[b for a,b in edge_set if a==node]+[a for a,b in edge_set if b==node]
            for nb in neighbors:
                if c[t-1,nb]==0 and c[t,nb]==1:
                    events.append({'time_index':t,'source':int(node),'neighbor':int(nb)})
                    if len(events)>=max_events:return events
    return events
