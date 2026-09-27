from pathlib import Path
import json, numpy as np, pandas as pd
import streamlit as st
from preprocessing.normalization import FeatureScaler
ROOT=Path(__file__).resolve().parents[1]

def load_json(path): return json.loads(path.read_text()) if path.exists() else None

def load_predictions():
    p=ROOT/'outputs/predictions/test_predictions.npz'; return np.load(p) if p.exists() else None

st.set_page_config(page_title='TrafficScope-AI',page_icon='🚦',layout='wide')
st.title('TrafficScope-AI')
st.caption('Missingness-aware traffic forecasting, reconstruction and congestion analysis')
metrics=load_json(ROOT/'outputs/metrics/experiment_summary.json'); info=load_json(ROOT/'outputs/artifacts/dataset_info.json')
if metrics is None:
    st.warning('Model artifacts are not available. Run `python run.py` from the project root first.'); st.stop()
info=info or {}; cols=st.columns(4); cols[0].metric('Dataset','PEMS04'); cols[1].metric('Sensors',info.get('nodes','—')); cols[2].metric('Time steps',info.get('time_steps','—')); cols[3].metric('Default missingness',f"{metrics.get('missing_rates',[0.3])[1]*100:.0f}%")
TABS=st.tabs(['Overview','EDA','Missing Data','Forecast','Network & Congestion','Evaluation','About'])
with TABS[0]:
 st.subheader('System workflow'); st.code('PEMS04 → validation → EDA → missingness mask → traffic graph → GCN + GRU → imputation + forecasting → evaluation → congestion analysis',language='text'); st.write('All displayed numerical results are loaded from the executed experiment artifacts.')
with TABS[1]:
 for f in ['traffic_over_time.png','feature_distributions.png','correlation_matrix.png']:
  p=ROOT/'outputs/figures'/f
  if p.exists(): st.image(str(p),caption=f.replace('_',' ').replace('.png','').title())
with TABS[2]:
 p=ROOT/'outputs/figures/missingness_heatmap.png'
 if p.exists(): st.image(str(p),caption='Simulated missingness')
 st.dataframe(pd.DataFrame(metrics.get('imputation',[])),use_container_width=True)
with TABS[3]:
 z=load_predictions()
 if z is not None:
  scaler=FeatureScaler().load(ROOT/'outputs/artifacts/scaler.pkl'); sensor=st.slider('Sensor',0,306,0); step=st.slider('Forecast step',1,z['pred'].shape[1],1)-1; feat=st.selectbox('Feature',['Flow','Speed','Occupancy']); fi=['Flow','Speed','Occupancy'].index(feat)
  actual=scaler.inverse_transform(z['y'])[:,step,sensor,fi]; pred=scaler.inverse_transform(z['pred'])[:,step,sensor,fi]
  st.line_chart(pd.DataFrame({'Actual':actual[:100],'Predicted':pred[:100]}))
 else: st.info('Prediction artifact unavailable.')
with TABS[4]:
 for f in ['network_graph.png','congestion_heatmap.png']:
  p=ROOT/'outputs/figures'/f
  if p.exists(): st.image(str(p),caption=f.replace('_',' ').replace('.png','').title())
 prop=load_json(ROOT/'outputs/metrics/propagation.json')
 if prop: st.metric('Congestion speed threshold',f"{prop['speed_threshold']:.3f}"); st.write(f"Recorded network-based propagation events: {len(prop.get('events',[]))}")
with TABS[5]:
 comp=ROOT/'outputs/metrics/comparison.csv'
 if comp.exists(): st.dataframe(pd.read_csv(comp),use_container_width=True)
 st.dataframe(pd.DataFrame(metrics.get('imputation',[])),use_container_width=True)
with TABS[6]:
 st.markdown('### Methodology'); st.write('GCN models spatial relationships among traffic sensors while GRU models temporal evolution. The multi-task model reconstructs masked historical states and forecasts future states. Congestion is inferred from a configurable speed percentile and propagation is treated as network-based analysis, not causal proof.')
 st.markdown('### Limitations'); st.write('PEMS04 is a benchmark dataset; simulated missingness differs from real sensor failures; congestion labels are inferred; propagation is not causal; demand analytics are a proxy rather than OD estimation.')
