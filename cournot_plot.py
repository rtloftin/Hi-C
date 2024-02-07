"""Standalone script to generates plots for experiments in the Cournot game."""
import argparse
from collections import defaultdict
import pathlib

from hi_c import line_plot, load_experiment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("path", type=str,
                        help="directory containing experimental results")
    parser.add_argument("title", type=str,
                        help="the title to be used for the plots")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = load_experiment(args.path)

    x_axis = data[0]["global/total_steps"]
    series = defaultdict(list)

    for seed in data:
        series["Leader"].append(seed["global/payoff_0"])
        series["Follower"].append(seed["global/payoff_1"])

    path = pathlib.PurePath(args.path)
    payoff_path = path.parent / (str(path.name) + "_payoffs.png")

    line_plot(series,
              x_axis,
              title=args.title,
              x_label="time steps",
              y_label="payoff",
              errors="range",
              image_path=payoff_path)
