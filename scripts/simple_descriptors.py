#!/usr/bin/env python
"""simple_descriptors.py: Data descriptor helper functions."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/pca_inputs/all_sites_pip.csv')
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.drop(['D0'], axis=1, inplace=True)
df = df[(df['Ed'] >= 0) & (df['Ed'] <= 4)]

# Summary statistics
print(df.describe())

df['Log10_n0'] = df['n0'].apply(np.log)
df['Log10_lambda'] = df['lambda'].apply(np.log)
df['Log10_Ed'] = df['Ed'].apply(np.log)
df['Log10_Sr'] = df['Sr'].apply(np.log)
df['Log10_Nt'] = df['Nt'].apply(np.log)
df = df[(df['Log10_n0'] >= 0)]
df = df[(df['Log10_lambda'] <= 1)]
df.drop(columns=['n0', 'lambda', 'Ed', 'Sr', 'Nt', 'Rho'], inplace=True)

print(df.skew())
print(df.kurtosis())

def plot_site_counts(df):
    site_counts = df['site'].value_counts()

    plt.bar(site_counts.index, site_counts.values)
    plt.xlabel('Site')
    plt.ylabel('Counts')
    plt.title('Total counts for each site')
    plt.xticks(ticks=range(len(site_counts.index)), labels=site_counts.index)
    plt.tight_layout()
    plt.savefig('../images/counts.png')

plot_site_counts(df)