import argparse
from pathlib import Path

import pandas as pd

from definitions import DOCUMENTS_PATH, N_ROUNDS_GROUP, N_ROUNDS_PLAYOFF
from utils.dataset import get_summary_of_games
from utils.swiss_system import generate_next_round


def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, type=int, help="Round number (1-8)")
    args = parser.parse_args()

    # Check if round number is valid
    if not 0 < args.round <= N_ROUNDS_GROUP + N_ROUNDS_PLAYOFF:
        raise ValueError("Invalid round number.")

    # Check if output files do not exist already (otherwise they can be overwritten by mistake)
    if (
        (DOCUMENTS_PATH / f"games_round_{args.round}").exists()
        or (DOCUMENTS_PATH / f"summary_round_{args.round - 1}").exists()
    ):
        raise RuntimeError("Results for this round are already present.")

    # Check if results from all previous rounds are present
    for i in range(1, args.round):
        if (
            not (DOCUMENTS_PATH / f"games_round_{i}").exists()
            or (i > 1 and not (DOCUMENTS_PATH / f"summary_round_{i - 1}").exists())
        ):
            raise RuntimeError("Results for previous rounds are missing.")

    with open(DOCUMENTS_PATH / "team_names.txt", "r") as f:
        team_names = f.read().split("\n")
    n_teams = len(team_names)
    n_games_per_round = n_teams // 2

    if args.round == 1:
        game_ids = [f"G-1-{i + 1}" for i in range(n_games_per_round)]
        games = pd.DataFrame(columns=["Round", "Team_1", "Team_2", "Score_1", "Score_2"])
        ratings = pd.Series(0, index=team_names)
    else:
        # Load all the games from previous rounds
        pass

    games_1 = generate_next_round(ratings, games, game_ids, args.round)
    games_1.to_csv(DOCUMENTS_PATH / f"games_round_{args.round}.csv")


if __name__ == "__main__":
    main()
