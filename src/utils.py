import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def download_kaggle_dataset(dataset_name, path='data', file_name=None):
    """Download a Kaggle dataset to a specified path."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    # No need to manually authenticate since they've been set as environment variables
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=path, unzip=True)
    df = pd.read_csv(f'{path}/{file_name}')
    return df

@st.cache_data
def create_wins_cross_table(df):
    """Create a cross table of the first hand color and the winning hand from a dataset of games played in BGA"""
    # First hand colors
    colors = {'y': 'Yellow', 'r': 'Red'}
    df['first_hand_color'] = df['record'].apply(lambda x: colors[x[0]])
    # First hand wins
    df['winning_hand'] = df.apply(lambda row: 'First hand' if row['winner'] == row['first_hand'] else 'Second hand', axis=1)

    # Cross table
    cross_table = pd.crosstab(df['first_hand_color'], df['winning_hand'])
    cross_table.index.name = None

    # Add total sum row and column
    cross_table['Total'] = cross_table.sum(axis=1)
    cross_table.loc['Total'] = cross_table.sum()

    return cross_table

@st.cache_data
def table_to_percentage(table):
    """Convert a table to percentage"""
    res = table / table.loc['Total', 'Total'] * 100
    return res.apply(lambda col: col.apply(lambda x: f"{x:.2f}%"))

def total_color(column):
    color_total = 'rgba(150,150,150, 0.5)'
    color_r = 'rgba(255,0,0, 0.25)'
    color_y = 'rgba(255,255,0, 0.25)'
    color_hand = 'rgba(200,200,200, 0.5)'
    return ['background-color: %s' % (color_total if (column.name=='Total' and index=='Total') else (color_hand if (index=='Total') else (color_r if (column.name=='Total' and index=='Red') else (color_y if (column.name=='Total' and index=='Yellow') else '')))) for index, x in column.items()]

@st.cache_data
def plot_hist(df, bin_width=1, title='Histogram'):
    from scipy.stats import skewnorm
    fig, ax = plt.subplots(1, 1)
    # Histogram
    bin_width = bin_width
    bins = [i for i in range(df.min(), df.max(), bin_width)]
    df.hist(bins=bins, edgecolor='white', grid=False, figsize=(20,8), alpha=0.5, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    # Distribution data
    a, loc, scale = skewnorm.fit(df)
    x = np.linspace(df.min(), df.max(), 100)
    prop = len(df)*bin_width
    ax.plot(x, prop*skewnorm.pdf(x, a, loc, scale), 'r-', lw=1, alpha=1, label=f'distribution approximation')
    ax.axvline(x=df.mean(), color='g', label=f'mean: {df.mean():.2f}')
    ax.axvline(x=df.median(), color='g', linestyle='--', label=f'median: {df.median():.2f}')
    ax.legend()
    ax.set_xlabel('Number of hands/moves per game', fontsize=16)
    ax.set_ylabel('Number of games', fontsize=16)
    return ax

def code_to_old(code):
    a = list(code)
    for i in range(len(a) - 1):
        if a[i] == 'y':
            a[i] = 'b'
        elif a[i] == 'r':
            a[i] = 'l'
            a[i+1] = str(6 - int(a[i+1]))
    code = ''.join(a)
    return code