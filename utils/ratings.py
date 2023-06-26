import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.dataset import get_games_graph, get_graph_components, get_teams_in_games, reorder_games_by_win


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
    games["Game_Rank_Diff"] = games["Score_1"] - games["Score_2"]
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
