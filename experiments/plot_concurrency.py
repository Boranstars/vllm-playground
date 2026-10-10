"""Plot three primary views and three post-hoc diagnostics from a levels CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("levels_csv", type=Path)
    path = parser.parse_args().levels_csv
    with path.open(newline="", encoding="utf-8") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["concurrency"]))
    if not rows:
        raise SystemExit("No completed levels in CSV")
    levels = [int(row["concurrency"]) for row in rows]
    if len(set(levels)) != len(levels):
        raise SystemExit("Expected one row per concurrency level")
    positions = list(range(len(levels)))
    stem = path.stem.removesuffix("-levels")
    plots = (
        # Primary plots specified before the experiment.
        ("throughput", "Generation throughput", "Generation tokens/s",
         [("generation_tokens_per_s", "Server Counter delta / time")]),
        ("ttft", "Time to first token", "TTFT (s)",
         [("ttft_p50_s", "p50"), ("ttft_p95_s", "p95")]),
        ("e2e", "End-to-end latency", "E2E (s)",
         [("e2e_p50_s", "p50"), ("e2e_p95_s", "p95")]),
        # Post-hoc diagnostic / learning plots of already collected fields.
        ("tpot", "Diagnostic: time per output token", "TPOT (ms/token)",
         [("tpot_p50_s_per_token", "p50"), ("tpot_p95_s_per_token", "p95")]),
        ("kv-usage", "Diagnostic: KV cache usage", "KV block-pool usage (%)",
         [("kv_cache_usage_perc_mean", "Mean"), ("kv_cache_usage_perc_peak", "Peak")]),
        ("running", "Diagnostic: running requests", "Running requests",
         [("running_mean", "Mean"), ("running_peak", "Peak")]),
    )
    for suffix, title, ylabel, series in plots:
        fig, ax = plt.subplots()
        scale = {"tpot": 1000, "kv-usage": 100}.get(suffix, 1)
        for key, label in series:
            values = [float(row[key]) * scale if row[key] else float("nan") for row in rows]
            ax.plot(positions, values, marker="o", label=label)
        if suffix == "running":
            ax.plot(positions, levels, linestyle="--", color="gray",
                    label="y = client concurrency (reference)")
            ax.set_ylim(bottom=0)
        elif suffix == "kv-usage":
            ax.set_ylim(0, 100)
        ax.set_xticks(positions, labels=[str(level) for level in levels])
        ax.set_xlabel("Client concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        output = path.with_name(f"{stem}-{suffix}.png")
        fig.savefig(output)
        plt.close(fig)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
