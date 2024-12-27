import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


#@st.cache_data # If this function is cached the data doesn't get reloaded when the date changes
def download_kaggle_dataset(dataset_name, path='data', file_name=None):
    """Download a Kaggle dataset to a specified path."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    import logging
    # No need to manually authenticate since they've been set as environment variables
    api = KaggleApi()
    api.authenticate()
    logging.info('Downloading dataset...')
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
def plotly_histogram(df, bin_width=1, title='Histogram'):
    from scipy.stats import skewnorm
    bins = list(range(df["moves"].min(), df["moves"].max() + bin_width, bin_width))

    fig = go.Figure()
    hist_color = "blue"
    fig.add_trace(
        go.Histogram(
            x=df["moves"],
            xbins=dict(start=bins[0], end=bins[-1], size=bin_width),  # Force bin width
            marker=dict(color=hist_color, line=dict(width=0.1, color="white")),
            opacity=0.5,
            name="Games by game length",
            hovertemplate="Games of length %{x}: " + f"<span style='color:{hist_color};'>" + "%{y}</span><extra></extra>",  # Custom tooltip
        )
    )
    # Fit skewnorm distribution to the data
    a, loc, scale = skewnorm.fit(df["moves"])
    x = np.linspace(df["moves"].min(), df["moves"].max(), 100)
    prop = len(df["moves"]) * bin_width
    skew_y = prop * skewnorm.pdf(x, a, loc, scale)
    # Add skewnorm line as a secondary trace
    dist_color = "red"
    dist_label = "Distribution approximation"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=skew_y,
            mode='lines',
            line=dict(color=dist_color, width=2),
            name=dist_label,
            hoverinfo="skip",  # Skip the hover information
        )
    )
    # Add mean and median lines
    mean_value = df["moves"].mean()
    median_value = df["moves"].median()
    max_bin_height = df["moves"].value_counts(bins=len(range(df["moves"].min(), df["moves"].max() + 1, bin_width))).max()
    fig.add_trace(
        go.Scatter(
            x=[mean_value, mean_value],
            y=[0, max_bin_height],
            mode='lines',
            line=dict(color='green', width=2),
            name=f"Mean: {mean_value:.2f}",
            hoverinfo="skip",  # Skip the hover information
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[median_value, median_value],
            y=[0, max_bin_height],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name=f"Median: {median_value:.2f}",
            hoverinfo="skip",  # Skip the hover information
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial", color="black")),
        xaxis_title="Number of hands/moves per game",
        yaxis_title="Number of games",
        bargap=0.1,
    )
    return fig

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

def get_number_of_wins_per_player(df, game_selection):
    """Get the number of wins per player"""
    player1 = df.loc[game_selection]['winner']
    player2 = df.loc[game_selection]['loser']
    wins = df[(df['winner'] == player1) & (df['loser'] == player2)].shape[0]
    losses = df[(df['winner'] == player2) & (df['loser'] == player1)].shape[0]
    return wins, losses