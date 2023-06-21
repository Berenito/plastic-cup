import typing as t
import pandas as pd


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


def get_summary_of_games(games: pd.DataFrame, ratings: pd.Series) -> pd.DataFrame:
    """
    Calculate summary statistics from the Games Table.

    :param games: Games Table
    :param ratings: Ratings
    :return: Summary DataFrame with columns Rating, Opponent_Rating, Record, Score
    """
    # Reorder such that Team_1 is the winner
    idx_bad_w_l = games["Score_1"] < games["Score_2"]
    games = games.copy()
    games.loc[idx_bad_w_l, ["Team_1", "Team_2", "Score_1", "Score_2"]] = games.loc[
        idx_bad_w_l, ["Team_2", "Team_1", "Score_2", "Score_1"]
    ].values
    team_names = ratings.index
    games_dupl = duplicate_games(games)
    summary = pd.DataFrame(index=team_names)
    summary["Wins"] = games.groupby("Team_1")["Score_1"].count().reindex(team_names).fillna(0).astype(int)
    summary["Losses"] = games.groupby("Team_2")["Score_2"].count().reindex(team_names).fillna(0).astype(int)
    summary["Goals_for"] = games_dupl.groupby("Team_1")["Score_1"].sum().reindex(team_names).fillna(0)
    summary["Goals_against"] = games_dupl.groupby("Team_1")["Score_2"].sum().reindex(team_names).fillna(0)
    summary["Record"] = summary.apply(lambda x: f"{x['Wins']}-{x['Losses']}", axis=1)
    summary["Rating"] = ratings
    games_dupl["Opponent_Rating"] = summary["Rating"].reindex(games_dupl["Team_2"]).values
    summary["Opponent_Rating"] = games_dupl.groupby("Team_1")["Opponent_Rating"].mean().reindex(team_names).fillna(0)
    summary["Score"] = summary.apply(lambda x: f"{x['Goals_for']}-{x['Goals_against']}", axis=1)
    summary = summary[["Rating", "Opponent_Rating", "Record", "Score"]].sort_values(by="Rating", ascending=False)
    return summary

