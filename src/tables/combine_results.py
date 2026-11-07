"""Merge every *results.csv under results/ into one table: dataset x method, "mean ± std".

    python src/tables/combine_results.py                 # prints the table and writes results/combined_table.csv
"""
import glob
import sys

import pandas as pd

METHODS = ['paper', 'rf', 'logreg', 'head', 'mixer', 'pool', 'pool_mixer', 'lora']
folder = sys.argv[1] if len(sys.argv) > 1 else 'results'
files = [f for f in glob.glob(f'{folder}/**/*results.csv', recursive=True) if 'method' in pd.read_csv(f, nrows=0).columns
         and 'paper' in pd.read_csv(f, nrows=0).columns]
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(['dataset', 'method'], keep='last')
cell = df.assign(v=df['mean'].map('{:.4f}'.format) + ' ± ' + df['std'].map('{:.4f}'.format))
table = cell.pivot(index='dataset', columns='method', values='v')
table.insert(0, 'paper', df.groupby('dataset')['paper'].first())
table = table[[m for m in METHODS if m in table.columns]]
table.to_csv(f'{folder}/combined_table.csv', encoding='utf-8-sig')
print(f'{len(files)} files, {table.shape[0]} datasets')
print(table.to_string())
