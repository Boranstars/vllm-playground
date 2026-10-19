"""Offline token-budget summaries and actual-interval stall plots."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import plot_prefill_interference as analysis

BUDGETS = (1024, 2048, 4096)
MEASURES = ("pre_median_gap_ms", "post_peak_gap_ms", "max_stall_gap_ms", "cohort_max_gap_ms",
            "stall_ratio", "significant_stall_count", "significant_stall_total_duration_ms", "foreground_ttft_ms")


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def save_csv(path, rows):
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values)) if values else dict(median=None, min=None, max=None)


def summarize_and_plot(directory, rows):
    summary = []
    for budget in BUDGETS:
        for condition in ("baseline", "long"):
            selected = [r for r in rows if int(r["budget"]) == budget and r["condition"] == condition]
            record = dict(budget=budget, condition=condition, trials=len(selected))
            for key in MEASURES:
                values = [float(r[key]) for r in selected if r[key] not in (None, "")]
                record.update({f"{key}_{stat}": value for stat, value in distribution(values).items()})
            summary.append(record)
    save_csv(directory / "summary.csv", summary)
    paired = []
    for budget in BUDGETS:
        for block in range(1, 11):
            pair = {r["condition"]: r for r in rows if int(r["budget"]) == budget and int(r["block_index"]) == block}
            if set(pair) != {"baseline", "long"}:
                raise ValueError(f"Missing baseline/long pair: {budget}, {block}")
            record = dict(budget=budget, block_index=block)
            for key in MEASURES[:-1]:
                baseline, long = (float(pair[c][key]) for c in ("baseline", "long"))
                record[key + "_long_minus_baseline"] = long - baseline
                record[key + "_long_over_baseline"] = long / baseline if baseline else None
            paired.append(record)
    save_csv(directory / "paired-comparisons.csv", paired)
    paired_summary = []
    for budget in BUDGETS:
        record = dict(budget=budget)
        for key in list(paired[0])[2:]:
            values = [r[key] for r in paired if r["budget"] == budget and r[key] is not None]
            record.update({f"{key}_{stat}": value for stat, value in distribution(values).items()})
        paired_summary.append(record)
    save_csv(directory / "paired-summary.csv", paired_summary)

    plots = directory / "plots"
    plots.mkdir(exist_ok=False)
    paths = []
    def save(fig, name):
        fig.tight_layout()
        path = plots / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    for name, measures in (
        ("foreground-ttft.png", [("foreground_ttft_ms", "Long foreground TTFT (ms)")]),
        ("decode-stalls.png", [("max_stall_gap_ms", "Post-peak gap (ms)"), ("stall_ratio", "Stall ratio")]),
        ("significant-stalls.png", [("significant_stall_count", "Abnormal interval count"),
                                    ("significant_stall_total_duration_ms", "Total abnormal interval duration (ms)")]),
        ("baseline-pre-gap.png", [("pre_median_gap_ms", "Pre-window median gap (ms)")]),
    ):
        fig, axes = plt.subplots(1, len(measures), figsize=(6 * len(measures), 4.5), squeeze=False)
        for ax, (key, label) in zip(axes.flat, measures):
            conditions = ("long",) if key == "foreground_ttft_ms" else ("baseline", "long")
            for cindex, condition in enumerate(conditions):
                color = "tab:orange" if condition == "long" else "tab:blue"
                offset = 0 if len(conditions) == 1 else (-0.16 if cindex == 0 else 0.16)
                for x, budget in enumerate(BUDGETS):
                    values = [float(r[key]) for r in rows if int(r["budget"]) == budget and r["condition"] == condition]
                    xs = [x + offset + (i - (len(values)-1)/2)*0.018 for i in range(len(values))]
                    ax.scatter(xs, values, color=color, alpha=0.8, s=22, label=condition if x == 0 else None)
                    ax.plot([x+offset-0.11, x+offset+0.11], [statistics.median(values)]*2, color=color, lw=2.5)
            ax.set_xticks(range(3), BUDGETS)
            ax.set(xlabel="Server max_num_batched_tokens", ylabel=label)
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
        fig.suptitle("All trial points; bars = medians; trial summary = median across 8 backgrounds" +
                     ("\nAbnormal: gap > 2 × request pre median; post endpoint in [0, 0.5] s" if name == "significant-stalls.png" else ""), fontsize=10)
        save(fig, name)

    all_gaps, integrity = {}, {}
    for budget in BUDGETS:
        block_dir = directory / f"budget-{budget}"
        trials = read_csv(block_dir / "trials.csv")
        metrics = read_csv(block_dir / "metrics.csv")
        grouped = defaultdict(list)
        deltas = []
        prompts = []
        with (block_dir / "events.jsonl").open() as source:
            for line in source:
                event = json.loads(line)
                grouped[event["trial_index"]].append(event)
                if event["delta_token_count"]:
                    deltas.append(event["delta_token_count"])
        for trial in trials:
            prompts.extend(s["actual_prompt_tokens"] for s in json.loads(trial["request_summaries"]) if s["role"] == "foreground")
            result, gaps = analysis.analyze_trial(trial, grouped[int(trial["trial_index"])])
            if not result["analysis_valid"]:
                raise ValueError(result["invalid_reason"])
            all_gaps[budget, int(trial["trial_index"])] = (trial, analysis.stall_gaps(gaps))
        integrity[budget] = dict(
            valid_trials=len(trials), token_bearing_events=len(deltas),
            singleton_event_ratio=sum(d == 1 for d in deltas)/len(deltas), max_delta=max(deltas),
            foreground_prompt_min=min(prompts), foreground_prompt_max=max(prompts),
            **{key: sum(float(t[key]) for t in trials) for key in
               ("prompt_tokens_local_cache_hit", "external_kv_transfer", "preemptions_total")},
            **{key+"_max": max(float(m[key]) for m in metrics) for key in
               ("num_requests_waiting", "waiting_capacity", "waiting_deferred", "kv_cache_usage_perc")},
        )
    (directory / "integrity.json").write_text(json.dumps(integrity, indent=2)+"\n")

    # All ten trials, eight tracks per trial, no representative-trial selection.
    for condition in ("long", "baseline"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        durations = [g["gap_ms"] for (trial, gaps) in all_gaps.values() if trial["condition"] == condition
                     for group in gaps.values() for g in group if g["start_relative_ms"] < 500 and g["relative_ms"] > -100]
        norm = Normalize(0, max(durations, default=1))
        for ax, budget in zip(axes, BUDGETS):
            trials = sorted((value for (b, _), value in all_gaps.items() if b == budget and value[0]["condition"] == condition),
                            key=lambda value: int(value[0]["block_index"]))
            segments, colors, ticks = [], [], []
            for index, (trial, gaps) in enumerate(trials):
                for bg, group in gaps.items():
                    y = index + (bg-3.5)*0.075
                    for gap in group:
                        start, end = gap["start_relative_ms"], gap["relative_ms"]
                        if start >= 500 or end <= -100:
                            continue
                        segments.append([(max(start,-100)/1000,y),(min(end,500)/1000,y)])
                        colors.append(gap["gap_ms"])
                        for edge in (start,end):
                            if -100 <= edge <= 500:
                                ticks.append([(edge/1000,y-0.025),(edge/1000,y+0.025)])
            collection = LineCollection(segments, cmap="viridis", norm=norm, linewidths=1.3)
            collection.set_array(colors)
            ax.add_collection(collection)
            ax.add_collection(LineCollection(ticks, colors="black", linewidths=0.5))
            ax.axvline(0, color="gray", ls="--", lw=0.8)
            ax.set(xlim=(-0.1,0.5), ylim=(9.5,-0.5), ylabel="Block (8 background tracks)", title=f"{condition.title()} — budget {budget}")
            ax.set_yticks(range(10), [f"B{t['block_index']} / T{t['trial_index']}" for t,_ in trials], fontsize=8)
            ax.grid(axis="x", alpha=0.2)
        axes[-1].set_xlabel("Alignment-relative time (s)")
        fig.suptitle("Client-observed token-bearing event gap intervals > 2 × request pre median\nAll actual intervals; ticks mark receipt boundaries; no merging/smoothing", fontsize=11)
        # Reserve colorbar space without tight_layout overlap.
        fig.subplots_adjust(left=0.13,right=0.85,top=0.90,bottom=0.06,hspace=0.35)
        cax=fig.add_axes([0.88,0.15,0.015,0.65])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap="viridis"),cax=cax,label="Full event gap (ms)")
        path=plots / f"{condition}-stall-only-timeline.png"
        fig.savefig(path,dpi=160)
        plt.close(fig)
        paths.append(str(path))

    old = directory.parents[1] / "prefill-interference-20261011-213351-40137cab-analysis.csv"
    if old.exists():
        old_rows = [r for r in read_csv(old) if r["condition"] == "long" and r["analysis_valid"] == "True"]
        new_rows = [r for r in rows if int(r["budget"]) == 2048 and r["condition"] == "long"]
        comparison = []
        for label, selected in (("20261011-independent",old_rows),("new-2048",new_rows)):
            comparison.append(dict(run=label, n=len(selected),
                foreground_ttft_ms_median=statistics.median(float(r["foreground_ttft_s"])*1000 for r in selected),
                stall_ratio_median=statistics.median(float(r["trial_stall_ratio"]) for r in selected),
                post_peak_gap_ms_median=statistics.median(float(r["post_peak_gap_ms"]) for r in selected)))
        save_csv(directory / "replication-2048.csv", comparison)
    (directory / "plots.json").write_text(json.dumps(paths,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.run_directory/"manifest.json").read_text())
    if manifest["smoke"] or manifest["status"] != "passed":
        parser.error("Only a fully validated formal run may be plotted")
    rows=[r for b in BUDGETS for r in read_csv(args.run_directory/f"budget-{b}"/"analysis.csv")]
    summarize_and_plot(args.run_directory,rows)
