from preprocessing.loader import load_pems04

def test_pems04_loader():
    data,edges,info=load_pems04('data/raw'); assert data.shape==(16992,307,3); assert edges.shape[0]==340
