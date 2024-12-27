import src.utils as utils
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os, hashlib, glob
from datetime import datetime
import plotly.express as px
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

st.set_page_config(
    page_title="Squadro",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get help': "https://dirdam.github.io/contact.html",
        'About': f"""This app was proudly developed by [Adrián Jiménez Pascual](https://dirdam.github.io/)."""
    })

# Set Kaggle environment variables
os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle_username"]
os.environ["KAGGLE_KEY"] = st.secrets["kaggle_key"]

# Specify the Kaggle dataset
dataset_name = 'dirdam/squadro-games-played-in-bga' # Kaggle dataset URL
data_path = 'data' # Local path where the dataset will be downloaded
file_name = 'Squadro_BGA_history.csv' # Actual file name in the dataset
# Registry of last download date
last_date_file = 'last_download_date.txt'

# Download dataset if today is not the last download date
today = datetime.now().date().strftime('%Y%m%d')
last_date = open(f'{data_path}/{last_date_file}', 'r').read() if os.path.exists(f'{data_path}/{last_date_file}') else None
if today != last_date or not os.path.exists(f'{data_path}/{file_name}'): # If date file does not exist or today is not the last download date
    # Remove the old file if it exists
    if os.path.exists(f'{data_path}/{file_name}'):
        os.remove(f'{data_path}/{file_name}')
    # Remove images. Since data changes images need to be updated too
    image_files = glob.glob(f'{data_path}/*.png')
    for file in image_files:
        os.remove(file)
    df = utils.download_kaggle_dataset(dataset_name, data_path, file_name)
    with open(f'{data_path}/{last_date_file}', 'w') as f:
        f.write(today)
else:
    df = pd.read_csv(f'{data_path}/{file_name}')

# Add column
df['moves'] = df['record'].str.len()//2

# Sidebar content
st.markdown( # Center the sidebar content
    """
    <style>
        [data-testid=stSidebar] {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
with st.sidebar:
    st.markdown('''# By and for [Squadristas](https://boardgamearena.com/gamepanel?game=squadro)''')
    st.image('https://dirdam.github.io/images/logo_squadro_transparent.png', use_container_width=True)

# Main content
st.title('Squadro Statistics Visualizer')
st.markdown('All the statistics shown here come from the latest version of the [Squadro games played in BGA](https://www.kaggle.com/datasets/dirdam/squadro-games-played-in-bga/data) dataset periodically uploaded to Kaggle.')
st.markdown(f'''As explained in the dataset description, the dataset contains information about most Squadro games played in [Board Game Arena](https://boardgamearena.com/gamepanel?game=squadro).
- Data coverage range: **`14-12-2020`** through **`{pd.to_datetime(df['date']).max().date().strftime('%d-%m-%Y')}`**.
- Number of games registered: **`{len(df):,.0f}`** games.''')

# Tabs
tab1, tab2 = st.tabs(['📈 Games statistics', '▶️ Replay specific game'])
# Cross tables
with tab1:
    st.markdown('## Restrict by ELO')
    threshold = st.slider('Restrict the games to those where both players had ELO greater than:', 0, 500, value=100, step=50)
    if threshold < 100:
        st.warning('⚠️ Only games where at least one player had ELO greater than 100 are taken into account.')
    st.markdown('## Games won by hand and color')
    st.markdown('The tables show the _**victories**_ per _color_ and _hand_ combination.')
    temp = df[(df['elo_winner'] >= threshold) & (df['elo_loser'] >= threshold)].copy()

    cross_table = utils.create_wins_cross_table(temp)
    percentage_table = utils.table_to_percentage(cross_table)
    # Show cross tables
    ct_col, per_col = st.columns(2)
    with ct_col:
        st.markdown('### Total number')
        st.write(cross_table.style.apply(utils.total_color).format("{:,.0f}"))
    with per_col:
        st.write('### Percentage')
        st.write(percentage_table.style.apply(utils.total_color))

    # Games length statistics
    st.markdown('## Games length histogram')
    st.markdown(f'''Visualization of how long games take to play when both players have ELO greater than **`{threshold}`**.
- **x-axis**: number of hands/moves (_plies_) per game.
- **y-axis**: number of games.''')

    bin_width = st.select_slider('Select bin width for the histogram:', options=[1, 5, 10, 20, 50], value=1)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(temp['moves']).values).hexdigest()
    fig = utils.plotly_histogram(temp, bin_width=bin_width, title='Squadro moves histogram')
    st.plotly_chart(fig, use_container_width=True)

# Replay
with tab2:
    st.markdown('## Replay specific game')
    st.markdown('Filter and choose which game you would like to replay.')
    df_replay_all = df[['date', 'winner', 'loser', 'elo_winner', 'elo_loser', 'moves']].copy()
    df_replay = df_replay_all.copy()
    cols = st.columns(3)
    with cols[0]:
        date = st.date_input('Select date (optional):', value=None, min_value=pd.to_datetime(df_replay['date'].min()), max_value=pd.to_datetime(df_replay['date'].max()))
        if date: # If date is selected, filter any game played on that date, regardless the hour
            df_replay = df_replay[pd.to_datetime(df_replay['date']).dt.date == date]
    with cols[1]:
        players = df_replay[['winner', 'loser']].melt()['value'].value_counts().index
        player1 = st.selectbox('Select player:', players, index=None, placeholder="Choose or type name")
        if player1:
            df_replay = df_replay[(df_replay['winner'] == player1) | (df_replay['loser'] == player1)]
    with cols[2]:
        rivals_list = df_replay[['winner', 'loser']].melt()['value'].value_counts().index
        player2 = st.selectbox('Select rival:', [r for r in rivals_list if r != player1], index=None, placeholder="Choose or type name", disabled=player1 is None)
        if player2:
            df_replay = df_replay[(df_replay['winner'] == player2) | (df_replay['loser'] == player2)]

    max_cap = min(50, len(df_replay)) # Limit the number of games to show
    show_top = st.checkbox(f'Show only top **`{max_cap}`** results from a total of **`{len(df_replay):,.0f}`** games.', value=True)

    row_height = 35
    st.dataframe(df_replay.head(max_cap) if show_top else df_replay, height=6*row_height) # Limits the height of the table to 6 rows

    red_rgb = 'rgb(140,0,0)'
    yellow_rgb = 'rgb(255,160,0)'
    color_css = f"""
    <style>
    .red {{
        color: {red_rgb};
    }}
    .yellow {{
        color: {yellow_rgb};
    }}
    </style>
    """
    st.markdown(color_css, unsafe_allow_html=True)

    # Game replay visualization
    game_selection = st.selectbox('Select a game to visualize from the upper table:', (df_replay.head(max_cap) if show_top else df_replay).index, index=None, placeholder="Choose or type row number", format_func=lambda x: f'{x} - "{df_replay.loc[x]["winner"]}" vs "{df_replay.loc[x]["loser"]}"')
    if game_selection is not None:
        game_record = df.loc[game_selection]['record']
        winner = df.loc[game_selection]['winner']
        loser = df.loc[game_selection]['loser']
        winner_is_first_hand = winner == df.loc[game_selection]['first_hand']
        winner_color = 'red' if ((game_record[0] == 'r' and winner_is_first_hand) or (game_record[0] == 'y' and not winner_is_first_hand)) else 'yellow'
        loser_color = 'yellow' if winner_color == 'red' else 'red'
        head_to_head_wins, head_to_head_losses = utils.get_number_of_wins_per_player(df, game_selection)
        st.markdown(f'### Replay of <span class="{winner_color}">_{df_replay.loc[game_selection]["winner"]}_</span> vs <span class="{loser_color}">_{df_replay.loc[game_selection]["loser"]}_</span>', unsafe_allow_html=True)
        st.markdown(f'''- Winner: _**{winner}**_.
    - Total head-to-head wins: _**{winner}**_ **`{head_to_head_wins}`** - **`{head_to_head_losses}`** _**{loser}**_.
- Replay buttons:
    - Click `▶️` to advance and `◀️` to go backwards. (`▶️▶️` and `◀️◀️` jump faster)
    - Click the top right corner of the board to see the game statistics for `Moves to finish`.''', unsafe_allow_html=True)
        game_record = utils.code_to_old(game_record)
        components.iframe(f'https://dirdam.github.io/games/squadro_replay.html?code={game_record}', height=600)

        # Players ELO progression
        st.markdown('### Players ELO evolution')
        # Restrict df to the players in the replay table
        players_df = df[(df['winner'] == winner) | (df['loser'] == winner) | (df['winner'] == loser) | (df['loser'] == loser)].copy()
        # Get ELO and date data for each player
        players_df = players_df[['winner', 'loser', 'elo_winner', 'elo_loser', 'date']]
        winners = players_df[['winner', 'elo_winner', 'date']].copy()
        winners.columns = ['player', 'elo', 'date']
        losers = players_df[['loser', 'elo_loser', 'date']].copy()
        losers.columns = ['player', 'elo', 'date']
        elo_df = pd.concat([winners, losers], ignore_index=True) # Concatenating the DataFrames
        elo_df = elo_df[(elo_df['player'] == winner) | (elo_df['player'] == loser)] # Restrict to winner and loser
        elo_df = elo_df.sort_values(by='date')

        color_map = {
            winner: red_rgb if winner_color == 'red' else yellow_rgb,
            loser: red_rgb if loser_color == 'red' else yellow_rgb
        }
        fig = px.line(elo_df, x='date', y='elo', color='player', markers=True, color_discrete_map=color_map)
        fig.update_layout(
            title='Historical ELO progression by player',
            xaxis_title='Date',
            yaxis_title='ELO'
        )

        # Shows a line chart with a line for each of the players' ELO progression
        st.plotly_chart(fig)