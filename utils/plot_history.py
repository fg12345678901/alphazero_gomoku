import argparse
import csv
from datetime import datetime

import matplotlib.pyplot as plt


def main(csv_path: str, out_img: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        print("No data in", csv_path)
        return

    steps = list(range(len(rows)))
    elo = [float(r["elo"]) for r in rows]
    win = [float(r["win_rate"]) for r in rows]
    times = [datetime.fromtimestamp(int(r["timestamp"])) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ----- step vs. Elo -----
    color = "tab:blue"
    ax1.set_xlabel("evaluation step")
    ax1.set_ylabel("Elo", color=color)
    ax1.plot(steps, elo, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    ax1_twin = ax1.twinx()
    color = "tab:green"
    ax1_twin.set_ylabel("Win rate", color=color)
    ax1_twin.plot(steps, win, color=color, linestyle="--")
    ax1_twin.tick_params(axis="y", labelcolor=color)

    # ----- time vs. Elo -----
    color = "tab:blue"
    ax2.set_xlabel("time")
    ax2.set_ylabel("Elo", color=color)
    ax2.plot(times, elo, color=color, marker="o")
    ax2.tick_params(axis="y", labelcolor=color)
    fig.autofmt_xdate(rotation=45)

    ax2_twin = ax2.twinx()
    color = "tab:green"
    ax2_twin.set_ylabel("Win rate", color=color)
    ax2_twin.plot(times, win, color=color, linestyle="--")
    ax2_twin.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(out_img)
    print("Saved figure to", out_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Elo history")
    parser.add_argument("--csv", default="logs/elo_history.csv")
    parser.add_argument("--out", default="elo_history.png")
    args = parser.parse_args()
    main(args.csv, args.out)
