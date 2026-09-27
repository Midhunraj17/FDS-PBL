from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parent

def run(script):
    return subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True).returncode

p=argparse.ArgumentParser(description='TrafficScope-AI pipeline')
p.add_argument('--prepare',action='store_true'); p.add_argument('--eda',action='store_true'); p.add_argument('--train',action='store_true'); p.add_argument('--evaluate',action='store_true'); p.add_argument('--all',action='store_true')
a=p.parse_args(); selected=any([a.prepare,a.eda,a.train,a.evaluate,a.all])
if not selected or a.prepare or a.all: run('prepare_data.py')
if not selected or a.eda or a.all: run('run_eda.py'); run('generate_demo_outputs.py')
if not selected or a.train or a.all: run('train.py')
if not selected or a.evaluate or a.all: run('evaluate.py')
print('[INFO] TrafficScope-AI pipeline complete.')
