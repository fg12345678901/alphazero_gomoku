import csv
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def main(csv_path: str, out_img: str):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print('No data in', csv_path)
        return
    steps = list(range(len(rows)))
    elo = [float(r['elo']) for r in rows]
    win = [float(r['win_rate']) for r in rows]
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('evaluation step')
    ax1.set_ylabel('Elo', color=color)
    ax1.plot(steps, elo, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Win rate', color=color)
    ax2.plot(steps, win, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.savefig(out_img)
    print('Saved figure to', out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Elo history')
    parser.add_argument('--csv', default='logs/elo_history.csv')
    parser.add_argument('--out', default='elo_history.png')
    args = parser.parse_args()
    main(args.csv, args.out)
