from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from definitions import DOCUMENTS_PATH, RNG
from utils.dataset import get_summary_of_games
from utils.swiss_system import generate_next_round


# Read team names
with open(DOCUMENTS_PATH / "team_names.txt", "r") as f:
    team_names = f.read().split("\n")
n_teams = len(team_names)
n_games_per_round = n_teams // 2

# Generate Round 1
game_ids_1 = [f"G-1-{i + 1}" for i in range(n_games_per_round)]
games_0 = pd.DataFrame(columns=["Round", "Team_1", "Team_2", "Score_1", "Score_2"])
ratings_0 = pd.Series(0, index=team_names)
games_1 = generate_next_round(ratings_0, games_0, game_ids_1, 2)

# Fill Round 1 with artificial results
games_1["Score_1"] = [15, 15, 15, 10, 12, 7, 15, 2, 8, 9, 15, 15]
games_1["Score_2"] = [6, 14, 10, 15, 15, 15, 13, 15, 15, 15, 10, 14]
games_1.to_csv(DOCUMENTS_PATH / "games_round_1.csv")
games_1["Rating"] = games_1["Score_1"] - games_1["Score_2"]
ratings_1 = pd.concat(
    [games_1[["Team_1", "Rating"]].set_index("Team_1"), games_1[["Team_2", "Rating"]].set_index("Team_2")]
).squeeze()
ratings_1.loc[ratings_1.index.isin(games_1["Team_2"])] *= -1
ratings_1 = ratings_1.sort_values(ascending=False)
summary_1 = get_summary_of_games(games_1, ratings_1)
summary_1.to_csv(DOCUMENTS_PATH / "summary_round_1.csv")

# Generate Round 2
game_ids_2 = [f"G-2-{i + 1}" for i in range(n_games_per_round)]
games_2 = generate_next_round(ratings_1, games_1, game_ids_2, 2)
games_2.to_csv(DOCUMENTS_PATH / "games_round_2.csv")

# Generate Round 2 as if for playoffs
game_ids_2_pf = [f"P-18-{i + 1}" for i in range(4)] + [f"G-2-{i + 1}" for i in range(n_games_per_round - 4)]
playoff_teams = ratings_1.index[: 8]
pairs_playoffs = [(playoff_teams[i], playoff_teams[7 - i]) for i in range(4)]
games_2_pf = generate_next_round(ratings_1.iloc[8 :], games_1, game_ids_2_pf, 2, pairs_playoffs)
games_2_pf.to_csv(DOCUMENTS_PATH / "games_round_2_pf.csv")

