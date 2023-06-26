import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from definitions import N_ROUNDS_GROUP, N_ROUNDS_PLAYOFF
from utils import (
    calculate_windmill_ratings,
    generate_swiss_round,
    get_summary_of_games,
    get_ranking_metrics,
    get_teams_in_games,
    reorder_games_by_win,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def main():
    """
    Calculate the ratings and summary for the previous round (unless the first round is specified).
    Ratings are calculated as the linear regression of the score differences capped at 15.
    Generate the games for the next round according to the minimum-cost pairing of the squares of rating differences
    (unless the last round plus one is specified).

    Run with arguments:
    * --path - Path where to store the documents
    * --round - Round number (1-9), games for round 9 will not be generated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path, help="Path where to store the documents")
    parser.add_argument("--round", required=True, type=int, help="Round number (1-9)")
    args = parser.parse_args()

    # Check if round number is valid
    if not 0 < args.round <= N_ROUNDS_GROUP + N_ROUNDS_PLAYOFF + 1:
        raise ValueError("Invalid round number.")

    # Check if output files do not exist already (otherwise they can be overwritten by mistake)
    if (
        (args.path / f"games_round_{args.round}.csv").exists()
        or (args.path / f"summary_round_{args.round - 1}.csv").exists()
    ):
        raise RuntimeError("Results for this round are already present.")

    # Check if results from all previous rounds are present
    for i in range(1, args.round):
        if (
            not (args.path / f"games_round_{i}.csv").exists()
            or (i > 1 and not (args.path / f"summary_round_{i - 1}.csv").exists())
        ):
            raise RuntimeError("Results for previous rounds are missing.")

    with open(args.path / "team_names.txt", "r") as f:
        team_names = f.read().split("\n")
    if len(set([t.lower() for t in team_names])) != len(team_names):
        raise ValueError("All team names must be unique.")
    if "pickup" in [t.lower() for t in team_names]:
        raise ValueError("'Pickup' is a reserved team name and cannot be used.")
    n_teams = len(team_names)
    n_games_per_round = int(np.ceil(n_teams / 2))
    team_names = pd.Series(team_names, name="Team")
    n_playoff_teams = 2 ** N_ROUNDS_PLAYOFF
    n_games_playoffs = n_playoff_teams // 2

    if args.round == 1:
        games = pd.DataFrame(columns=["Round", "Team_1", "Team_2", "Score_1", "Score_2"])
        ratings = pd.Series(0, index=team_names)
    else:
        # Load all the games from previous rounds
        games_list = [pd.read_csv(args.path / f"games_round_{i + 1}.csv", index_col=0) for i in range(args.round - 1)]
        games = pd.concat(games_list)
        if (games[["Score_1", "Score_2"]].dtypes == "object").any() or games.isna().any().any():
            raise ValueError("Games contain invalid entries.")
        if (games["Score_1"] == games["Score_2"]).any():
            raise ValueError("Draws are not allowed.")
        # Calculate summary of the previous round
        games_swiss = games.loc[games.index.str.startswith("G")]
        games_playoffs = games.loc[games.index.str.startswith("P")]
        ratings = calculate_windmill_ratings(games_swiss, team_names)
        summary = get_summary_of_games(games_swiss, ratings, team_names)
        rmse, max_resid = get_ranking_metrics(games_swiss, ratings)
        print(f"RMSE: {rmse:.4f}, Max Resid: {max_resid:.4f}")
        if max_resid > 0.1:
            raise ValueError("Ratings were not calculated correctly.")
        if args.round > N_ROUNDS_GROUP + 1:
            summary = summary.loc[~summary.index.isin(get_teams_in_games(games_playoffs))]
        summary.to_csv(args.path / f"summary_round_{args.round - 1}.csv")

    if args.round <= N_ROUNDS_GROUP + N_ROUNDS_PLAYOFF:
        if args.round <= N_ROUNDS_GROUP:
            game_ids = [f"G-{args.round}-{i + 1}" for i in range(n_games_per_round)]
            pairs_playoffs = []
        else:
            n_remain = N_ROUNDS_GROUP + N_ROUNDS_PLAYOFF - args.round + 1
            idx_better = [2 ** n_remain * (i // (2 ** (n_remain - 1))) + 1 for i in range(n_games_playoffs)]
            idx_game = [i % (2 ** (n_remain - 1)) + 1 for i in range(n_games_playoffs)]
            game_ids_playoffs = [
                f"P-{ib}{ib + 2 ** n_remain - 1}-{ig}" for ib, ig in zip(idx_better, idx_game)
            ]
            game_ids_swiss = [f"G-{args.round}-{i + 1}" for i in range(n_games_per_round - n_games_playoffs)]
            game_ids = game_ids_playoffs + game_ids_swiss
            if args.round == N_ROUNDS_GROUP + 1:
                # Generate first playoff pairs based on the ratings
                playoff_teams = ratings.index[: n_playoff_teams].tolist()
            else:
                # Generate next playoff pairs based on the results
                games_playoffs_use = reorder_games_by_win(games_playoffs.loc[games_playoffs["Round"] == args.round - 1])
                places = games_playoffs_use.index.map(lambda x: x.split("-")[1]).values
                games_playoffs_use["Places"] = places
                playoff_teams = [
                    team
                    for _, g in games_playoffs_use.groupby("Places")
                    for team in g["Team_1"].tolist() + g["Team_2"].tolist()
                ]
            pairs_playoffs = [
                (playoff_teams[(ib - 1) + (ig - 1)], playoff_teams[ib + 2 ** n_remain - 1 - ig])
                for ib, ig in zip(idx_better, idx_game)
            ]

        games_next = generate_swiss_round(ratings, games, game_ids, args.round, pairs_playoffs)
        games_next.to_csv(args.path / f"games_round_{args.round}.csv")


if __name__ == "__main__":
    main()
