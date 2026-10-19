import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    raise SystemExit("Usage: python experiments/plot_prompt_length.py results/prompt_length-<timestamp>.csv")

csv_path = Path(sys.argv[1])
raw_prompt_tokens = []
raw_ttft_ms = []
raw_tpot_ms = []
groups = defaultdict(lambda: {"prompt_tokens": [], "ttft_ms": [], "tpot_ms": []})

with csv_path.open(newline="", encoding="utf-8") as csv_file:
    for row in csv.DictReader(csv_file):
        prompt_tokens = int(row["actual_prompt_tokens"])
        ttft_ms = float(row["ttft_s"]) * 1000
        tpot_ms = float(row["tpot_s_per_token"]) * 1000
        target_size = row["target_size"]

        raw_prompt_tokens.append(prompt_tokens)
        raw_ttft_ms.append(ttft_ms)
        raw_tpot_ms.append(tpot_ms)
        groups[target_size]["prompt_tokens"].append(prompt_tokens)
        groups[target_size]["ttft_ms"].append(ttft_ms)
        groups[target_size]["tpot_ms"].append(tpot_ms)

median_prompt_tokens = []
median_ttft_ms = []
median_tpot_ms = []
for values in groups.values():
    median_prompt_tokens.append(statistics.median(values["prompt_tokens"]))
    median_ttft_ms.append(statistics.median(values["ttft_ms"]))
    median_tpot_ms.append(statistics.median(values["tpot_ms"]))

median_groups = sorted(
    zip(median_prompt_tokens, median_ttft_ms, median_tpot_ms),
    key=lambda point: point[0],
)
median_prompt_tokens = [point[0] for point in median_groups]
median_ttft_ms = [point[1] for point in median_groups]
median_tpot_ms = [point[2] for point in median_groups]

ttft_fig, ttft_ax = plt.subplots()
ttft_ax.scatter(raw_prompt_tokens, raw_ttft_ms, label="Raw samples")
ttft_ax.plot(
    median_prompt_tokens,
    median_ttft_ms,
    marker="o",
    label="Group medians",
)
ttft_ax.set_xlabel("Actual prompt tokens")
ttft_ax.set_ylabel("TTFT (ms)")
ttft_ax.set_title("Prompt length vs TTFT")
ttft_ax.legend()
ttft_fig.tight_layout()
ttft_path = csv_path.with_name(f"{csv_path.stem}-ttft.png")
ttft_fig.savefig(ttft_path)
plt.close(ttft_fig)

tpot_fig, tpot_ax = plt.subplots()
tpot_ax.scatter(raw_prompt_tokens, raw_tpot_ms, label="Raw samples")
tpot_ax.plot(
    median_prompt_tokens,
    median_tpot_ms,
    marker="o",
    label="Group medians",
)
tpot_ax.set_xlabel("Actual prompt tokens")
tpot_ax.set_ylabel("TPOT (ms/token)")
tpot_ax.set_title("Prompt length vs TPOT")
tpot_ax.legend()
tpot_fig.tight_layout()
tpot_path = csv_path.with_name(f"{csv_path.stem}-tpot.png")
tpot_fig.savefig(tpot_path)
plt.close(tpot_fig)

print(f"Saved: {ttft_path}")
print(f"Saved: {tpot_path}")
