import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import src.utils as utils

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
utils.download_kaggle_dataset(dataset_name, download_path)

# Load the dataset
df = pd.read_csv(f'{download_path}/{file_name}')
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
st.markdown('As explained in the dataset description, the dataset contains information about most Squadro games played in [Board Game Arena](https://boardgamearena.com/) since it was made available on `14-12-2020`.')

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
st.markdown(f'''Visualization of how long games take to play when both players have ELO greater than {threshold}.
- **x-axis**: number of hands/moves (_plies_) per game.
- **y-axis**: number of games.''')
ax = utils.plot_hist(temp['moves'], bin_width=1, title='Squadro moves histogram')
st.pyplot(ax.figure)

# Replay
st.markdown('---')
st.markdown('## Replay specific game')
df_replay_all = df[['date', 'winner', 'loser', 'elo_winner', 'elo_loser', 'moves']].copy()
df_replay = df_replay_all.copy()
cols = st.columns(3)
with cols[0]:
    date = st.date_input('Select date:', value=None, min_value=pd.to_datetime(df_replay['date'].min()), max_value=pd.to_datetime(df_replay['date'].max()))
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

st.dataframe(df_replay)

game_selection = st.selectbox('Select a game to visualize:', df_replay.index, index=None, placeholder="Choose or type row number", format_func=lambda x: f'{x} - "{df_replay.loc[x]["winner"]}" vs "{df_replay.loc[x]["loser"]}"')
if game_selection is not None:
    game_record = df.loc[game_selection]['record']
    winner_color = 'red' if game_record[0] == 'r' else 'yellow'
    st.markdown(f'### Replay of _{df_replay.loc[game_selection]["winner"]}_ vs _{df_replay.loc[game_selection]["loser"]}_')
    st.markdown(f'''- Click `▶️` to advance and `◀️` to go backwards. (`▶️▶️` and `◀️◀️` jump faster)
- Click the top right corner of the board to see the game statistics for `Moves to finish`.
- Result: _**{df.loc[game_selection]['winner']}**_ (_{winner_color}_) wins.''')
    game_record = utils.code_to_old(game_record)
    components.iframe(f'https://dirdam.github.io/games/squadro_replay.html?code={game_record}', height=600)