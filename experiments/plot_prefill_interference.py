"""Offline analysis of client-observed token-bearing events; never sends inference."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

CONDITIONS = ("baseline", "short", "long")
STALL_FACTOR = 2.0
ANALYSIS_COLUMNS = [
    "run_id", "block_index", "trial_index", "condition", "pre_median_gap_ms",
    "post_peak_gap_ms", "trial_stall_ratio", "token_bearing_events",
    "singleton_event_ratio", "two_token_event_ratio", "ge3_token_event_ratio",
    "multi_token_event_ratio", "max_delta_token_count", "singleton_pair_count",
    "foreground_ttft_s", "analysis_valid", "invalid_reason", "request_window_summaries",
]


def event_gaps(events):
    """Pair receipts; the later event still determines analysis-window membership."""
    bearing = sorted((e for e in events if e["delta_token_count"] > 0),
                     key=lambda e: (e["received_ns"], e["event_index"]))
    return [
        {"start_relative_ms": previous["relative_to_alignment_ms"],
         "relative_ms": current["relative_to_alignment_ms"],
         "gap_ms": (current["received_ns"] - previous["received_ns"]) / 1e6,
         "singleton_pair": previous["delta_token_count"] == current["delta_token_count"] == 1}
        for previous, current in zip(bearing, bearing[1:])
    ]


def analyze_trial(trial, events):
    errors = []
    if trial["status"] != "success":
        errors.append("measurement failed: " + trial.get("error", ""))
    if not trial.get("alignment_ns"):
        errors.append("alignment missing")
    grouped = defaultdict(list)
    for event in events:
        if event["role"] == "background":
            grouped[event["background_id"]].append(event)
    if set(grouped) != set(range(8)):
        errors.append("expected all 8 backgrounds")
    distributions = [e["delta_token_count"] for group in grouped.values()
                     for e in group if e["delta_token_count"] > 0]
    n = len(distributions)
    gaps, summaries = {}, []
    for background_id, group in sorted(grouped.items()):
        if any(e["relative_to_alignment_ms"] is None for e in group):
            errors.append(f"bg{background_id}: alignment missing")
            continue
        gaps[background_id] = event_gaps(group)
        pre = [g["gap_ms"] for g in gaps[background_id] if -500 <= g["relative_ms"] < 0]
        post = [g["gap_ms"] for g in gaps[background_id] if 0 <= g["relative_ms"] <= 500]
        # Two or more gaps in each window is the predeclared minimum; never fill gaps.
        valid = len(pre) >= 2 and len(post) >= 2 and statistics.median(pre) > 0
        if not valid:
            errors.append(f"bg{background_id}: insufficient/zero gaps (pre={len(pre)}, post={len(post)})")
        pre_median = statistics.median(pre) if pre else None
        post_peak = max(post) if post else None
        summaries.append(dict(background_id=background_id, pre_gap_count=len(pre),
                              post_gap_count=len(post), pre_median_gap_ms=pre_median,
                              post_peak_gap_ms=post_peak,
                              request_stall_ratio=post_peak / pre_median if valid else None))
    valid = not errors and len(summaries) == 8
    row = {k: trial.get(k, "") for k in ("run_id", "block_index", "trial_index", "condition", "foreground_ttft_s")}
    row.update(
        pre_median_gap_ms=statistics.median(s["pre_median_gap_ms"] for s in summaries) if valid else None,
        post_peak_gap_ms=statistics.median(s["post_peak_gap_ms"] for s in summaries) if valid else None,
        trial_stall_ratio=statistics.median(s["request_stall_ratio"] for s in summaries) if valid else None,
        token_bearing_events=n,
        singleton_event_ratio=sum(d == 1 for d in distributions) / n if n else None,
        two_token_event_ratio=sum(d == 2 for d in distributions) / n if n else None,
        ge3_token_event_ratio=sum(d >= 3 for d in distributions) / n if n else None,
        multi_token_event_ratio=sum(d > 1 for d in distributions) / n if n else None,
        max_delta_token_count=max(distributions, default=0),
        singleton_pair_count=sum(g["singleton_pair"] for group in gaps.values() for g in group),
        analysis_valid=valid, invalid_reason="; ".join(errors),
        request_window_summaries=json.dumps(summaries),
    )
    return row, gaps


def timeline_bins(gaps, bin_ms, left_ms, right_ms):
    """Legacy endpoint-bin diagnostic; the interval timeline does not use this."""
    # Preserve a brief spike: take each request's peak per bin, then median across
    # requests present in that bin. Missing requests/bins are not filled with zero.
    per_request = defaultdict(lambda: defaultdict(list))
    for background_id, group in gaps.items():
        for gap in group:
            t = gap["relative_ms"]
            if left_ms <= t < right_ms:
                index = math.floor((t - left_ms) / bin_ms)
                per_request[index][background_id].append(gap["gap_ms"])
    return {index: statistics.median(max(values) for values in requests.values())
            for index, requests in per_request.items()}


def stall_gaps(gaps, factor=STALL_FACTOR):
    """Filter each request independently; pre-window membership uses the later receipt."""
    selected = {}
    for background_id, group in gaps.items():
        pre = [gap["gap_ms"] for gap in group if -500 <= gap["relative_ms"] < 0]
        if len(pre) < 2 or statistics.median(pre) <= 0:
            raise ValueError(f"bg{background_id}: insufficient/zero pre-window baseline")
        threshold = factor * statistics.median(pre)
        selected[background_id] = [gap for gap in group if gap["gap_ms"] > threshold]
    return selected


def gap_interval_figure(trials, analyses, trial_gaps, left_ms, right_ms, stall_factor=None):
    """Show every observed interval on a separate background track within its trial."""
    visible = {condition: [] for condition in CONDITIONS}
    durations = []
    for trial, analysis, gaps in zip(trials, analyses, trial_gaps):
        if not analysis["analysis_valid"] or trial["condition"] not in visible:
            continue
        if stall_factor is not None:
            gaps = stall_gaps(gaps, stall_factor)
        visible[trial["condition"]].append((trial, gaps))
        durations.extend(gap["gap_ms"] for group in gaps.values() for gap in group
                         if gap["start_relative_ms"] < right_ms and gap["relative_ms"] > left_ms)
    # One shared color scale; segment length is the directly observed time gap.
    norm = Normalize(vmin=0, vmax=max(durations, default=1) or 1)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    for ax, condition, title_color in zip(axes, CONDITIONS, ("tab:blue", "tab:orange", "tab:green")):
        rows = sorted(visible[condition], key=lambda row: int(row[0]["trial_index"]))
        segments, receipt_ticks, colors, labels = [], [], [], []
        for row_index, (trial, gaps) in enumerate(rows):
            block = trial.get("block_index")
            labels.append(f"B{block} / T{trial['trial_index']}" if block else f"T{trial['trial_index']}")
            for background_id, group in sorted(gaps.items()):
                # Eight distinct thin tracks per trial, bg0 at the top.
                y = row_index + (background_id - 3.5) * 0.075
                for gap in group:
                    start, end = gap["start_relative_ms"], gap["relative_ms"]
                    if start >= right_ms or end <= left_ms:
                        continue
                    segments.append([(max(start, left_ms) / 1000, y),
                                     (min(end, right_ms) / 1000, y)])
                    colors.append(gap["gap_ms"])
                    if stall_factor is not None and left_ms <= start <= right_ms:
                        receipt_ticks.append([(start / 1000, y - 0.025),
                                              (start / 1000, y + 0.025)])
                    if left_ms <= end <= right_ms:
                        receipt_ticks.append([(end / 1000, y - 0.025),
                                              (end / 1000, y + 0.025)])
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.0)
        collection.set_array(colors)
        ax.add_collection(collection)
        # Real receipt boundaries distinguish adjacent gaps even if colors match.
        ax.add_collection(LineCollection(receipt_ticks, colors="black", linewidths=0.4, alpha=0.5))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_xlim(left_ms / 1000, right_ms / 1000)
        ax.set_ylim(max(len(rows) - 0.5, 0.5), -0.5)
        ax.set_yticks(range(len(rows)), labels=labels, fontsize=8)
        ax.set_ylabel("Trial (8 background tracks)")
        ax.set_title(f"{condition.title()} — {len(rows)} valid trials", loc="left",
                     color=title_color, fontsize=11)
        ax.grid(axis="x", alpha=0.18)
        if not rows:
            ax.text(0.5, 0.5, "No analysis-valid trials", transform=ax.transAxes, ha="center")
        elif stall_factor is not None and not segments:
            ax.text(0.5, 0.5, "No gaps above the per-request threshold in this window",
                    transform=ax.transAxes, ha="center", fontsize=10, color="gray")
    axes[-1].set_xlabel("Time relative to alignment (s)")
    if stall_factor is None:
        title = "Client-observed token-bearing event gap intervals"
        subtitle = "Segments span consecutive event receipts; ticks mark receipts. No binning or trial aggregation."
    else:
        title = f"Client-observed stall intervals: gap > {stall_factor:g} × request pre-median"
        subtitle = "Per-request baseline: median gap with later receipt in [-0.5, 0) s. Segments span actual receipts."
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.5, 0.947, subtitle, ha="center", fontsize=10)
    fig.subplots_adjust(left=0.14, right=0.86, top=0.9, bottom=0.09, hspace=0.3)
    color_ax = fig.add_axes([0.89, 0.15, 0.018, 0.67])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
    colorbar.set_label("Full event gap (ms); intervals clipped only at display edges")
    return fig


def plot_results(path, trials, analyses, trial_gaps, bin_ms=20, left_ms=-500, right_ms=1000,
                 timeline_only=False, stall_only=False):
    stem = path.stem.removesuffix("-trials")
    outputs = []

    def save(fig, suffix, tight=True):
        output = path.with_name(f"{stem}-{suffix}.png")
        if tight:
            fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)

    if stall_only:
        fig = gap_interval_figure(trials, analyses, trial_gaps, left_ms, right_ms,
                                  stall_factor=STALL_FACTOR)
        save(fig, "stall-only-timeline", tight=False)
        return outputs

    fig = gap_interval_figure(trials, analyses, trial_gaps, left_ms, right_ms)
    save(fig, "event-gap-timeline", tight=False)
    if timeline_only:
        return outputs

    for suffix, conditions, key, ylabel, title in (
        ("stall-ratio", CONDITIONS, "trial_stall_ratio", "Trial stall ratio", "Median of 8 request stall ratios"),
        ("foreground-ttft", CONDITIONS[1:], "foreground_ttft_s", "Foreground TTFT (s)", "First token-bearing event, output=1"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for position, condition in enumerate(conditions):
            values = [float(row[key]) for trial, row in zip(trials, analyses) if row["condition"] == condition
                      and row[key] not in (None, "") and
                      (row["analysis_valid"] if key == "trial_stall_ratio" else
                       trial["status"] == "success")]
            # Deterministic display jitter only, never change data values.
            xs = [position + (i - (len(values) - 1) / 2) * min(0.04, 0.3 / max(1, len(values)))
                  for i in range(len(values))]
            ax.scatter(xs, values, alpha=0.8, label="Trial" if position == 0 else None)
            if values:
                ax.plot([position - 0.16, position + 0.16], [statistics.median(values)] * 2,
                        color="black", linewidth=2, label="Median" if position == 0 else None)
        if key == "trial_stall_ratio":
            ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(conditions)), labels=[c.title() for c in conditions])
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        save(fig, suffix)
    return outputs


def analyze_file(path, bin_ms=20, left_ms=-500, right_ms=1000, timeline_only=False, stall_only=False):
    with path.open(newline="", encoding="utf-8") as source:
        trials = list(csv.DictReader(source))
    if not trials:
        raise ValueError("No trials in CSV")
    keys = [(t["run_id"], int(t["trial_index"])) for t in trials]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate trial identity")
    groups = defaultdict(list)
    event_path = path.with_name(path.name.removesuffix("-trials.csv") + "-events.jsonl")
    with event_path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            key = (event["run_id"], int(event["trial_index"]))
            if key not in keys:
                raise ValueError(f"Event has unknown trial identity: {key}")
            groups[key].append(event)
    analyses, gaps = [], []
    for trial, key in zip(trials, keys):
        analysis, trial_gaps = analyze_trial(trial, groups[key])
        analyses.append(analysis)
        gaps.append(trial_gaps)
    analysis_path = path.with_name(path.name.removesuffix("-trials.csv") + "-analysis.csv")
    if timeline_only or stall_only:
        return analyses, plot_results(path, trials, analyses, gaps, bin_ms, left_ms, right_ms,
                                      timeline_only=timeline_only, stall_only=stall_only)
    with analysis_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(analyses)
    return analyses, [analysis_path, *plot_results(path, trials, analyses, gaps, bin_ms, left_ms, right_ms)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials_csv", type=Path)
    parser.add_argument("--bin-ms", type=float, default=20,
                        help="Legacy compatibility option; interval timeline uses no bins")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--timeline-only", action="store_true",
                      help="Write only the full-interval timeline PNG; leave CSVs and other plots untouched")
    mode.add_argument("--stall-only", action="store_true",
                      help="Write only a stall-only timeline PNG (gap > 2 × request pre-median)")
    parser.add_argument("--window-ms", nargs=2, type=float, default=(-500, 1000), metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    if not math.isfinite(args.bin_ms) or args.bin_ms <= 0:
        parser.error("--bin-ms must be finite and positive")
    left, right = args.window_ms
    if not all(math.isfinite(x) for x in (left, right)) or left >= right:
        parser.error("--window-ms requires finite LEFT < RIGHT")
    analyses, outputs = analyze_file(args.trials_csv, args.bin_ms, left, right,
                                     args.timeline_only, args.stall_only)
    for row in analyses:
        print(json.dumps({k: v for k, v in row.items() if k != "request_window_summaries"}))
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
