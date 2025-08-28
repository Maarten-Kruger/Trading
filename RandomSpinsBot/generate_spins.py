import csv
import os
import random
import argparse
from datetime import datetime


"""
Random Spins Generator

Generates a CSV with n spins for each of x buttons (modes), where each
button has its own configurable win probability.

Quick edit: modify BUTTONS and DEFAULT_SPINS below, or use CLI flags.

CSV columns:
- button_id: index of the button (0-based)
- button_name: name of the button/mode
- spin_index: index of spin within this button (1..n)
- win: 1 if win, 0 if loss
- outcome: "win" or "loss"
- rand: random number drawn (0..1)
- run_id: timestamp for this generation (ISO format)
"""


# ----- Easy-to-edit section -----
# Define your buttons (modes) here. You can change names and win_prob values.
BUTTONS = [
    {"name": "Mode 1", "win_prob": 0.50},
    {"name": "Mode 2", "win_prob": 0.50},
    {"name": "Mode 3", "win_prob": 0.52},
    {"name": "Mode 4", "win_prob": 0.50},
    {"name": "Mode 5", "win_prob": 0.50},
]

# Default number of spins per button (can be overridden via CLI)
DEFAULT_SPINS = 20

# Default output path (can be overridden via CLI)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "output", "spins.csv")
# --------------------------------


def validate_buttons(buttons):
    if not buttons:
        raise ValueError("BUTTONS list is empty.")
    for i, b in enumerate(buttons):
        if "name" not in b or "win_prob" not in b:
            raise ValueError(f"Button at index {i} missing 'name' or 'win_prob'.")
        p = b["win_prob"]
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"win_prob for '{b['name']}' must be in [0,1], got {p}.")


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def generate_spins(buttons, n_spins, out_path, seed=None):
    validate_buttons(buttons)
    if seed is not None:
        random.seed(seed)

    ensure_dir(out_path)
    run_id = datetime.utcnow().isoformat()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["button_id", "button_name", "spin_index", "win", "outcome", "rand", "run_id"]) 

        for idx, button in enumerate(buttons):
            name = button["name"]
            p = button["win_prob"]
            for s in range(1, n_spins + 1):
                r = random.random()
                win = 1 if r < p else 0
                outcome = "win" if win == 1 else "loss"
                writer.writerow([idx, name, s, win, outcome, f"{r:.10f}", run_id])

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random spins CSV for multiple slot modes.")
    parser.add_argument("--n", type=int, default=DEFAULT_SPINS, help="Number of spins per button (default: %(default)s)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output CSV file path (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = generate_spins(BUTTONS, args.n, args.out, args.seed)
    print(f"Wrote spins to: {out_path}")


if __name__ == "__main__":
    main()

