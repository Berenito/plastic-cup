import argparse
from pathlib import Path

import pandas as pd

from definitions import N_ROUNDS_GROUP, N_ROUNDS_PLAYOFF
from utils.dataset import get_summary_of_games, get_ranking_metrics
from utils.ratings import calculate_windmill_ratings
from utils.swiss_system import generate_next_round


def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path, help="Path where to store the documents")
    parser.add_argument("--round", required=True, type=int, help="Round number (1-8)")
    args = parser.parse_args()

    # Check if round number is valid
    if not 0 < args.round <= N_ROUNDS_GROUP + N_ROUNDS_PLAYOFF:
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
    n_teams = len(team_names)
    n_games_per_round = n_teams // 2

    if args.round == 1:
        games = pd.DataFrame(columns=["Round", "Team_1", "Team_2", "Score_1", "Score_2"])
        ratings = pd.Series(0, index=team_names)
    else:
        # Load all the games from previous rounds
        games_list = [pd.read_csv(args.path / f"games_round_{i + 1}.csv", index_col=0) for i in range(args.round - 1)]
        games = pd.concat(games_list)
        if games.isna().any().any():
            raise ValueError("Games contain invalid entries.")
        # Calculate summary of the previous round
        ratings = calculate_windmill_ratings(games)
        summary = get_summary_of_games(games, ratings)
        summary.to_csv(args.path / f"summary_round_{args.round - 1}.csv")
        rmse, max_resid = get_ranking_metrics(games, ratings)
        print(f"RMSE: {rmse:.4f}, Max Resid: {max_resid:.4f}")

    game_ids = [f"G-{args.round}-{i + 1}" for i in range(n_games_per_round)]
    games_next = generate_next_round(ratings, games, game_ids, args.round)
    games_next.to_csv(args.path / f"games_round_{args.round}.csv")


if __name__ == "__main__":
    main()
