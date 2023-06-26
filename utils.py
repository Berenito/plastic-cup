import typing as t

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from definitions import MAX_SCORE_DIFF


def generate_swiss_round(
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
    ratings = ratings.loc[~ratings.index.isin([team for pair in pairs_playoffs for team in pair])]
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
    pairing_costs = pd.DataFrame(
        (ratings_np[:, None] - ratings_np[None, :]) ** 2, index=teams, columns=teams, dtype=float
    )
    if len(teams) % 2 == 1:
        pairing_costs["pickup"] = 0
        pairing_costs.loc["pickup"] = 0
        pairing_costs.loc["pickup", "pickup"] = np.nan
        ratings = ratings.copy()
        ratings["pickup"] = -np.inf  # For ordering purposes
    for team in teams:
        pairing_costs.loc[team, team] = np.nan
    for _, rw in games.iterrows():
        if rw["Team_1"] in teams + ["pickup"] and rw["Team_2"] in teams + ["pickup"]:
            pairing_costs.loc[rw["Team_1"], rw["Team_2"]] = np.nan
            pairing_costs.loc[rw["Team_2"], rw["Team_1"]] = np.nan
    pairing_costs = pairing_costs.stack().reset_index()
    pairing_costs.columns = ["source", "target", "weight"]
    pairing_costs["weight"] = 1 + pairing_costs["weight"].max() - pairing_costs["weight"]
    g = nx.from_pandas_edgelist(pairing_costs, edge_attr="weight")
    pairs = [p if ratings[p[0]] >= ratings[p[1]] else p[::-1] for p in nx.max_weight_matching(g)]
    return sorted(pairs, key=lambda x: ratings[x[0]] + ratings[x[1]])[::-1]


def calculate_windmill_ratings(
    games: pd.DataFrame, teams: pd.Series, mean_rating: int = 0, n_round: int = 2
) -> pd.Series:
    """
    Calculate Windmill ratings (linear regression fit of the score differences) based on the given games.

    :param games: Games table, DataFrame with columns Round", "Team_1", "Team_2", "Score_1", "Score_2"
    :param teams: Teams
    :param mean_rating: Mean rating
    :param n_round: Number of decimals to round
    :return:
    """
    games = games.loc[~(games[["Team_1", "Team_2"]] == "pickup").any(axis=1)]
    games = reorder_games_by_win(games.copy())
    games["Game_Rank_Diff"] = (games["Score_1"] - games["Score_2"]).clip(upper=MAX_SCORE_DIFF)
    components = get_graph_components(get_games_graph(games), teams)

    coefficients = pd.Series(0, index=teams, dtype="float64")
    for i_comp, comp in components.reset_index().groupby("Component"):
        teams_comp = comp["Team"]
        df_comp = games.loc[games["Team_1"].isin(teams_comp) | games["Team_2"].isin(teams_comp)].reset_index()
        df_comp["const"] = 1
        df_plus = df_comp.pivot(index="index", columns="Team_1", values="const").fillna(0).loc[df_comp["index"]]
        df_minus = df_comp.pivot(index="index", columns="Team_2", values="const").fillna(0).loc[df_comp["index"]]
        x = df_plus.subtract(df_minus, fill_value=0)[teams_comp]
        y = df_comp["Game_Rank_Diff"]
        lr = LinearRegression(fit_intercept=False).fit(x, y)
        coefficients[teams_comp] = (lr.coef_ - lr.coef_.mean() + mean_rating).round(n_round)
    return coefficients.sort_values(ascending=False)


def get_summary_of_games(games: pd.DataFrame, ratings: pd.Series, teams: pd.Series) -> pd.DataFrame:
    """
    Calculate summary statistics from the Games Table.

    :param games: Games Table
    :param ratings: Ratings
    :param teams: Teams
    :return: Summary DataFrame with columns Rating, Opponent_Rating, Record, Score
    """
    games = games.loc[~(games[["Team_1", "Team_2"]] == "pickup").any(axis=1)]
    games = reorder_games_by_win(games.copy())
    games_dupl = duplicate_games(games)
    summary = pd.DataFrame(index=teams)
    summary["Wins"] = games.groupby("Team_1")["Score_1"].count().reindex(teams).fillna(0).astype(int)
    summary["Losses"] = games.groupby("Team_2")["Score_2"].count().reindex(teams).fillna(0).astype(int)
    summary["Goals_for"] = games_dupl.groupby("Team_1")["Score_1"].sum().reindex(teams).fillna(0).astype(int)
    summary["Goals_against"] = games_dupl.groupby("Team_1")["Score_2"].sum().reindex(teams).fillna(0).astype(int)
    summary["Record"] = summary.apply(lambda x: f"{x['Wins']}-{x['Losses']}", axis=1)
    summary["Rating"] = ratings
    games_dupl["Opponent_Rating"] = summary["Rating"].reindex(games_dupl["Team_2"]).values
    summary["Opponent_Rating"] = games_dupl.groupby("Team_1")["Opponent_Rating"].mean().reindex(teams).fillna(0)
    summary["Opponent_Rating"] = summary["Opponent_Rating"].round(2)
    summary["Score"] = summary.apply(lambda x: f"{x['Goals_for']}-{x['Goals_against']}", axis=1)
    summary = summary[["Rating", "Opponent_Rating", "Record", "Score"]].sort_values(by="Rating", ascending=False)
    return summary


def duplicate_games(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add duplicates of the games with Team_1 <-> Team_2 and Score_1 <-> Score_2; i.e., each game will be twice
    in the returned Games Table (some functions are easier to apply on the Games Table in this format).

    :param df_games: Games Table
    :return: Duplicated Games Table
    """
    df_games_reversed = df_games.rename(
        columns={"Team_1": "Team_2", "Team_2": "Team_1", "Score_1": "Score_2", "Score_2": "Score_1"}
    )
    # If Team_Rank_Diff_{...} or Game_Rank_Diff_{...} is present, change the sign in the reversed table
    df_games_reversed.loc[:, df_games_reversed.columns.str.contains("Rank_Diff")] *= -1
    return pd.concat([df_games, df_games_reversed]).reset_index(drop=True)


def get_games_graph(games: pd.DataFrame) -> nx.Graph:
    """
    Get graph representation of the connected teams (with played game) from the Games Table using networkx library.

    :param games: Games Table
    :return: Graph with connections between the teams that played together
    """
    teams = get_teams_in_games(games)
    df_connected_init = pd.DataFrame(0, index=teams, columns=teams)  # To ensure that all the teams are in the DataFrame
    df_connected = duplicate_games(games).groupby(["Team_1", "Team_2"])["Team_1"].count().rename("N_Games")
    df_connected = df_connected.reset_index().pivot(index="Team_1", columns="Team_2", values="N_Games").fillna(0)
    df_connected = df_connected_init.add(df_connected, fill_value=0).astype("int").clip(upper=1)
    return nx.from_pandas_adjacency(df_connected)


def get_graph_components(graph_connections: nx.Graph, teams: t.Union[list, pd.Series]) -> pd.Series:
    """
    Return the graph component label for each team in the graph. Numbering is ordered based on the number
    of the teams in the component (bigger components go first).

    :param graph_connections: Graph with connections between the teams that played together
    :param teams: A list of teams we are interested in
    :return: Graph component label for each team
    """
    components_raw = list(nx.algorithms.connected_components(graph_connections))
    components_raw = sorted(components_raw, key=len, reverse=True)  # Sort list of lists by length in descending order
    dict_components = dict(zip(range(1, len(components_raw) + 1), components_raw))
    components = pd.Series(index=teams, name="Component", dtype=float)
    for team in teams:
        component_key = [k for k, v in dict_components.items() if team in v]
        components[team] = component_key[0] if component_key else np.nan
    return components


def get_teams_in_games(games: pd.DataFrame) -> pd.Series:
    """
    Get all teams present in the Games Table.

    :param games: Games Table
    :return: Series of the teams
    """
    games_dupl = duplicate_games(games)
    return pd.Series(games_dupl["Team_1"].unique()).rename("Team").sort_values()


def get_ranking_metrics(games: pd.DataFrame, ratings: pd.Series) -> t.Tuple[float, float]:
    """
    Get quality metrics for given ratings.

    :param games: Games Table
    :param ratings: Ratings
    :return: RMSE; max avg of teams' residuals
    """
    games = reorder_games_by_win(games.copy())
    games["Game_Rank_Diff"] = (games["Score_1"] - games["Score_2"]).clip(upper=MAX_SCORE_DIFF)
    games["Team_Rank_Diff"] = ratings.reindex(games["Team_1"]).values - ratings.reindex(games["Team_2"]).values
    games["Resid_1"] = games[f"Game_Rank_Diff"] - games[f"Team_Rank_Diff"]
    games["Resid_2"] = -games["Resid_1"]
    rmse = np.sqrt(np.mean(games["Resid_1"] ** 2))
    games_extended = pd.concat(
        [
            games.rename(columns={"Team_1": "Team", "Resid_1": "Resid"}),
            games.rename(columns={"Team_2": "Team", "Resid_2": "Resid"})
        ]
    )
    sum_resid_teams = games_extended.groupby("Team")["Resid"].sum()
    max_sum_resid_team = sum_resid_teams.abs().max()
    return rmse, max_sum_resid_team


def reorder_games_by_win(games: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder Games Table such that Team_1 is always a winner.

    :param games: Games Table
    :return: Games Table with Team_1 as winner
    """
    idx_bad_w_l = games["Score_1"] < games["Score_2"]
    if idx_bad_w_l.any():
        games.loc[idx_bad_w_l, ["Team_1", "Team_2", "Score_1", "Score_2"]] = games.loc[
            idx_bad_w_l, ["Team_2", "Team_1", "Score_2", "Score_1"]
        ].values
    return games
