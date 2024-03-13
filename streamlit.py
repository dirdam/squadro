import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os, hashlib
import src.utils as utils
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
download_path = 'data' # Local path where the dataset will be downloaded
file_name = 'Squadro_BGA_history.csv' # Actual file name in the dataset

# Download dataset
today = datetime.now().date()
if 'last_date' not in st.session_state or today > st.session_state.last_date:
    logging.info('Downloading dataset...')
    df = utils.download_kaggle_dataset(dataset_name, download_path, file_name)
    st.session_state.last_date = today
else:
    df = pd.read_csv(f'{download_path}/{file_name}')

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
    st.markdown('''# For and by [Squadristas](https://boardgamearena.com/gamepanel?game=squadro)''')
    st.image('https://dirdam.github.io/images/logo_squadro_transparent.png', use_column_width=True)

# Main content
st.title('Squadro Statistics Visualizer')
st.markdown('All the statistics shown here come from the latest version of the [Squadro games played in BGA](https://www.kaggle.com/datasets/dirdam/squadro-games-played-in-bga/data) dataset periodically uploaded to Kaggle.')
st.markdown(f'''As explained in the dataset description, the dataset contains information about most Squadro games played in [Board Game Arena](https://boardgamearena.com/).
- Data coverage range: **`14-12-2020`** through **`{pd.to_datetime(df['date']).max().date().strftime('%d-%m-%Y')}`**.
- Number of games registered: **`{len(df):,.0f}`** games.''')

# Cross tables
st.markdown('---')
st.markdown('## Games won by hand and color')
st.markdown('The tables show the _victories_ per _color_ and _hand_ combination.')
threshold = st.slider('Restrict the tables values to games where both players had ELO greater than:', 0, 500, value=100, step=50)
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
ax = utils.plot_hist(temp['moves'], bin_width=1, title='Squadro moves histogram')
data_hash = hashlib.sha256(pd.util.hash_pandas_object(temp['moves']).values).hexdigest()
if not os.path.exists(f'{download_path}/{data_hash}.png'): # If the image does not exist, save it
    st.pyplot(ax.figure)
    ax.figure.savefig(f'{download_path}/{data_hash}.png')
else: # If the image exists, show it
    st.image(f'{download_path}/{data_hash}.png')

# Replay
st.markdown('---')
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

st.dataframe(df_replay.head(max_cap) if show_top else df_replay)

color_css = """
<style>
.red {
    color: rgb(140,0,0);
}
.yellow {
    color: rgb(255,160,0);
}
</style>
"""
st.markdown(color_css, unsafe_allow_html=True)

# Game replay visualization
game_selection = st.selectbox('Select a game to visualize from the upper table:', (df_replay.head(max_cap) if show_top else df_replay).index, index=None, placeholder="Choose or type row number", format_func=lambda x: f'{x} - "{df_replay.loc[x]["winner"]}" vs "{df_replay.loc[x]["loser"]}"')
if game_selection is not None:
    game_record = df.loc[game_selection]['record']
    winner_is_first_hand = df.loc[game_selection]['winner'] == df.loc[game_selection]['first_hand']
    winner_color = 'red' if ((game_record[0] == 'r' and winner_is_first_hand) or (game_record[0] == 'y' and not winner_is_first_hand)) else 'yellow'
    loser_color = 'yellow' if winner_color == 'red' else 'red'
    head_to_head_wins, head_to_head_losses = utils.get_number_of_wins_per_player(df, game_selection)
    st.markdown(f'### Replay of <span class="{winner_color}">_{df_replay.loc[game_selection]["winner"]}_</span> vs <span class="{loser_color}">_{df_replay.loc[game_selection]["loser"]}_</span>', unsafe_allow_html=True)
    st.markdown(f'''- Winner: _**{df.loc[game_selection]['winner']}**_.
    - Total head-to-head wins: _**{df.loc[game_selection]["winner"]}**_ **`{head_to_head_wins}`** - **`{head_to_head_losses}`** _**{df.loc[game_selection]["loser"]}**_.
- Replay buttons:
    - Click `▶️` to advance and `◀️` to go backwards. (`▶️▶️` and `◀️◀️` jump faster)
    - Click the top right corner of the board to see the game statistics for `Moves to finish`.''', unsafe_allow_html=True)
    game_record = utils.code_to_old(game_record)
    components.iframe(f'https://dirdam.github.io/games/squadro_replay.html?code={game_record}', height=600)