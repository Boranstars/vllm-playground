import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    raise SystemExit("Usage: python experiments/plot_output_length.py results/output_length-<timestamp>.csv")

csv_path = Path(sys.argv[1])
raw_completion_tokens = []
raw_e2e_ms = []
raw_tpot_ms = []
groups = defaultdict(lambda: {"completion_tokens": [], "e2e_ms": [], "tpot_ms": []})

with csv_path.open(newline="", encoding="utf-8") as csv_file:
    for row in csv.DictReader(csv_file):
        completion_tokens = int(row["completion_tokens"])
        e2e_ms = float(row["e2e_s"]) * 1000
        tpot_ms = (
            None
            if not row["tpot_s_per_token"]
            else float(row["tpot_s_per_token"]) * 1000
        )
        target = row["target_max_tokens"]

        raw_completion_tokens.append(completion_tokens)
        raw_e2e_ms.append(e2e_ms)
        if tpot_ms is not None:
            raw_tpot_ms.append((completion_tokens, tpot_ms))
        groups[target]["completion_tokens"].append(completion_tokens)
        groups[target]["e2e_ms"].append(e2e_ms)
        if tpot_ms is not None:
            groups[target]["tpot_ms"].append(tpot_ms)

median_completion_tokens = []
median_e2e_ms = []
median_tpot_ms = []
for values in groups.values():
    median_completion_tokens.append(statistics.median(values["completion_tokens"]))
    median_e2e_ms.append(statistics.median(values["e2e_ms"]))
    median_tpot_ms.append(
        statistics.median(values["tpot_ms"]) if values["tpot_ms"] else None
    )

median_groups = sorted(
    zip(median_completion_tokens, median_e2e_ms, median_tpot_ms),
    key=lambda point: point[0],
)
median_completion_tokens = [point[0] for point in median_groups]
median_e2e_ms = [point[1] for point in median_groups]
median_tpot_ms = [point[2] for point in median_groups]

e2e_fig, e2e_ax = plt.subplots()
e2e_ax.scatter(raw_completion_tokens, raw_e2e_ms, label="Raw samples")
e2e_ax.plot(
    median_completion_tokens,
    median_e2e_ms,
    marker="o",
    label="Group medians",
)
e2e_ax.set_xlabel("Actual completion tokens")
e2e_ax.set_ylabel("End-to-end latency (ms)")
e2e_ax.set_title("Output length vs end-to-end latency")
e2e_ax.legend()
e2e_fig.tight_layout()
e2e_path = csv_path.with_name(f"{csv_path.stem}-e2e.png")
e2e_fig.savefig(e2e_path)
plt.close(e2e_fig)

tpot_fig, tpot_ax = plt.subplots()
tpot_x = [point[0] for point in raw_tpot_ms]
tpot_y = [point[1] for point in raw_tpot_ms]
tpot_ax.scatter(tpot_x, tpot_y, label="Raw samples")
tpot_ax.plot(
    median_completion_tokens,
    median_tpot_ms,
    marker="o",
    label="Group medians",
)
tpot_ax.set_xlabel("Actual completion tokens")
tpot_ax.set_ylabel("TPOT (ms/token)")
tpot_ax.set_title("Output length vs TPOT")
tpot_ax.legend()
tpot_fig.tight_layout()
tpot_path = csv_path.with_name(f"{csv_path.stem}-tpot.png")
tpot_fig.savefig(tpot_path)
plt.close(tpot_fig)

print(f"Saved: {e2e_path}")
print(f"Saved: {tpot_path}")
