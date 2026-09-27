import pandas as pd

def save_comparison(rows,path):
    df=pd.DataFrame(rows); df.to_csv(path,index=False); return df
