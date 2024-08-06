#!/usr/bin/env python
"""psd_input_analysis.py: Early visualization plots for analyzing microphysical observations."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_corr(df, size=12):
    df = df[['Log10_Nt', 'Log10_Sr', 'Fs', 'Log10_n0', 'Log10_Ed', 'Log10_lambda']]
    corr = df.corr()
    corr_values = corr.unstack().drop_duplicates().dropna()
    corr_values = corr_values[corr_values.index.get_level_values(0) != corr_values.index.get_level_values(1)]
    sorted_corr = corr_values.abs().sort_values(ascending=False)
    sorted_labels = list(sorted_corr.index)
    sorted_columns = []
    for label_pair in sorted_labels:
        if label_pair[0] not in sorted_columns:
            sorted_columns.append(label_pair[0])
        if label_pair[1] not in sorted_columns:
            sorted_columns.append(label_pair[1])
            
    sorted_corr = corr.loc[sorted_columns, sorted_columns]
    fig, ax = plt.subplots(figsize=(size, size))
    plt.title("PSD Variable Correlation Matrix")
    h = ax.matshow(sorted_corr, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(h, ax=ax, label='Correlation')
    plt.xticks(range(len(sorted_corr.columns)), sorted_corr.columns, rotation=90)
    plt.yticks(range(len(sorted_corr.columns)), sorted_corr.columns)
    plt.tight_layout()
    plt.savefig('../images/sorted_corr.png')

df = pd.read_csv('../data/pca_inputs/all_sites_pip.csv')
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.drop(['D0'], axis=1, inplace=True)
df['phase'] = np.where(df['Rho'] <= 0.4, 1, 0)

df = df[(df['Ed'] >= 0) & (df['Ed'] <= 4)]
df['Log10_n0'] = df['n0'].apply(np.log)
df['Log10_lambda'] = df['lambda'].apply(np.log)
df['Log10_Ed'] = df['Ed'].apply(np.log)
df['Log10_Sr'] = df['Sr'].apply(np.log)
df['Log10_Nt'] = df['Nt'].apply(np.log)
df = df[(df['Log10_n0'] >= 0)]
df = df[(df['Log10_lambda'] <= 1)]
df.drop(columns=['n0', 'lambda', 'Ed', 'Sr', 'Nt', 'Rho'], inplace=True)
print(df)
plot_corr(df)

sns_plot = sns.pairplot(df, kind='kde', hue='phase', height=3, palette={0: 'red', 1: 'blue'}, corner=True)
# sns_plot.map_upper(sns.kdeplot, levels=4, color=".2")
sns_plot.savefig('../images/output_kde.png')
