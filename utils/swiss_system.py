import typing as t

import networkx as nx
import numpy as np
import pandas as pd


def generate_next_round(
    ratings: pd.Series,
    games: pd.DataFrame,
    game_ids: t.List[str],
    round_num: int,
    pairs_playoffs: t.List[t.Tuple[str, str]] = [],
):
    """
    Generate the next group round based on the team ratings.

    :param ratings: Ratings of non-playoff teams
    :param games: Games table, DataFrame with columns Round", "Team_1", "Team_2", "Score_1", "Score_2"
    :param game_ids: Game IDs, acts like the index of the Games tables
    :param round_num: Round number
    :param pairs_playoffs: Already determined playoff pairs
    :return: Games table, DataFrame with columns Round", "Team_1", "Team_2", "Score_1", "Score_2" (scores are empty)
    """
    games_next = pd.DataFrame(index=game_ids, columns=["Round", "Team_1", "Team_2", "Score_1", "Score_2"])
    games_next["Round"] = round_num
    n_games_playoffs = len(pairs_playoffs)
    if pairs_playoffs:
        games_next.loc[game_ids[: n_games_playoffs] , ["Team_1", "Team_2"]] = pairs_playoffs
    games_next.loc[game_ids[n_games_playoffs :], ["Team_1", "Team_2"]] = get_minimal_pairing(ratings, games)
    return games_next


def get_minimal_pairing(ratings: pd.Series, games: pd.DataFrame) -> t.List[t.Tuple[str, str]]:
    """
    Generate minimal pairing of the teams based on their ratings

    :param ratings: Ratings
    :param games: Games table, DataFrame with columns Round", "Team_1", "Team_2", "Score_1", "Score_2"
    :return: Pairs of next-round opponents
    """
    ratings_np = ratings.values
    teams = ratings.index.tolist()
    pairing_costs = pd.DataFrame((ratings_np[:, None] - ratings_np[None, :]) ** 2, index=teams, columns=teams)
    for team in teams:
        pairing_costs.loc[team, team] = np.nan
    for _, rw in games.iterrows():
        if rw["Team_1"] in teams and rw["Team_2"] in teams:
            pairing_costs.loc[rw["Team_1"], rw["Team_2"]] = np.nan
            pairing_costs.loc[rw["Team_2"], rw["Team_1"]] = np.nan
    pairing_costs = pairing_costs.stack().reset_index()
    pairing_costs.columns = ["source", "target", "weight"]
    pairing_costs["weight"] = 1 + pairing_costs["weight"].max() - pairing_costs["weight"]
    # pairing_costs = pairing_costs.loc[pairing_costs["source"] < pairing_costs["target"]]
    g = nx.from_pandas_edgelist(pairing_costs, edge_attr="weight")
    pairs = [p if ratings[p[0]] >= ratings[p[1]] else p[::-1] for p in nx.max_weight_matching(g)]
    return sorted(pairs, key=lambda x: ratings[x[0]] + ratings[x[1]])[::-1]
