import networkx as nx
import matplotlib.pyplot as plt

def save_network(edges,nodes,path,highlight=None):
    G=nx.Graph(); G.add_nodes_from(range(nodes)); G.add_edges_from([(int(a),int(b)) for a,b in edges]); pos=nx.spring_layout(G,seed=42,k=0.15,iterations=30)
    plt.figure(figsize=(9,7)); nx.draw_networkx_edges(G,pos,alpha=.15,width=.4); nx.draw_networkx_nodes(G,pos,node_size=12,node_color='steelblue')
    if highlight is not None and highlight in G: nx.draw_networkx_nodes(G,pos,nodelist=[highlight],node_size=80,node_color='crimson')
    plt.title('PEMS04 Sensor Graph'); plt.axis('off'); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
