"""Replay raw evidence and render mechanism timelines or a preemption zoom."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kv_pressure_experiment as k


def read_data(directory):
    trials = [json.loads(line) for line in (directory/"trials.jsonl").read_text().splitlines()]
    metrics, events = defaultdict(list), defaultdict(list)
    with (directory/"metrics.csv").open() as f:
        for r in csv.DictReader(f):
            row = {key: value if key in ("run_id", "condition") else float(value) for key,value in r.items()}
            metrics[int(row["trial_index"])].append(row)
    with (directory/"events.jsonl").open() as f:
        for line in f:
            event=json.loads(line)
            events[event["client_request_id"]].append(event)
    return trials, metrics, events


def replay(trials, metrics, events, pool):
    analysis = []
    all_requests = []
    for t in trials:
        k.validate_trial(t)
        ms = metrics[t["trial_index"]]
        if not ms or any(b["sample_ns"] <= a["sample_ns"] for a,b in zip(ms,ms[1:])):
            raise ValueError("Missing/out-of-order metrics")
        before, after = t["idle_before"], t["idle_after"]
        for name in ("preemptions", "prefix_cache_hits_total", "external_kv_transfer_tokens"):
            values = [before[name]]+[m[name] for m in ms]+[after[name]]
            # Sampler may include snapshots immediately after stored idle_after.
            if any(b<a for a,b in zip(values,values[1:])):
                raise ValueError("Nonmonotone counter")
        if k.counter_delta(before["preemptions"],after["preemptions"]) != t["preemption_delta"]:
            raise ValueError("Counter delta replay mismatch")
        gaps, nearby = [], []
        event_count = multi = total_tokens = 0
        for r in t["requests"]:
            all_requests.append(r["client_request_id"])
            es=events[r["client_request_id"]]
            cumulative=0
            last=r["start_ns"]
            for index,e in enumerate(es):
                cumulative += len(e["token_ids"])
                if (e["trial_index"] != t["trial_index"] or e["condition"] != t["condition"]
                        or e["event_index"] != index or e["delta_token_count"] != len(e["token_ids"])
                        or e["cumulative_token_count"] != cumulative or e["received_ns"] < last
                        or any(type(token) is not int for token in e["token_ids"])):
                    raise ValueError("Raw SSE integrity replay mismatch")
                last=e["received_ns"]
            if cumulative != r["cumulative"] or r["actual_prompt_tokens"]+r["max_tokens"] >= pool.max_model_len:
                raise ValueError("Request length mismatch")
            counts=[e["prompt_token_ids_count"] for e in es if e["prompt_token_ids_count"] is not None]
            if counts and any(c != r["actual_prompt_tokens"] for c in counts):
                raise ValueError("Tokenize versus stream prompt mismatch")
            if r["cancelled"]:
                if not r["start_ns"] < r["cancel_ns"] <= r["closed_ns"] or r["error"]:
                    raise ValueError("Cancellation replay mismatch")
            elif not (r["valid"] and es[-1]["has_usage"] and cumulative == r["max_tokens"]
                      == r["completion_tokens"] == es[-1]["usage_completion_tokens"]
                      and es[-1]["usage_prompt_tokens"] == r["actual_prompt_tokens"]
                      and r["finish_reason"] == "length"):
                raise ValueError("Final usage / finish replay mismatch")
            bearing=[e for e in es if e["delta_token_count"]]
            if not bearing or bearing[0]["received_ns"] != r["first_ns"]:
                raise ValueError("First token replay mismatch")
            event_count+=len(bearing)
            multi+=sum(e["delta_token_count"]>1 for e in bearing)
            total_tokens+=cumulative
            for a,b in zip(bearing,bearing[1:]):
                gap=(b["received_ns"]-a["received_ns"])/1e6
                gaps.append(gap)
                marker=t.get("first_observed_preemption_ns")
                if marker and marker-1e9 <= b["received_ns"] <= marker+1e9:
                    nearby.append(gap)
        if t["condition"] in k.ADMISSION:
            trigger=next(m for m in ms if m["sample_ns"]==t["trigger_ns"])
            if trigger["running"] != t["holder_count"] or trigger["waiting"] != 0 or trigger["preemptions"] != before["preemptions"]:
                raise ValueError("Trigger replay mismatch")
            if abs(pool.free(trigger)-t["trigger_free_blocks"]) > .01:
                raise ValueError("Free blocks replay mismatch")
            if t["condition"] == k.ADMISSION[0]:
                waiting=[m for m in ms if t["first_waiting_ns"]<=m["sample_ns"]<t["release_ns"]]
                if len(waiting)<3 or any(m["waiting"]<1 or m["running"]!=t["holder_count"] for m in waiting):
                    raise ValueError("Blocked plateau replay mismatch")
                if any(e["delta_token_count"] and e["received_ns"]<=t["release_ns"]
                       for r in t["requests"] if r["role"]=="foreground" for e in events[r["client_request_id"]]):
                    raise ValueError("FG token before release")
        else:
            initial=next(m for m in ms if m["sample_ns"]==t["initial_admitted_ns"])
            if initial["running"] != t["cohort_size"] or initial["waiting"] or initial["preemptions"] != before["preemptions"]:
                raise ValueError("Initial admission replay mismatch")
        row={key:t.get(key) for key in (
            "trial_index","block_index","condition","duration_s","trigger_free_blocks",
            "fg_required_blocks","margin_blocks","holder_blocks_estimate","post_release_free_blocks",
            "blocked_samples","blocked_span_ms","client_blocked_interval_ms","release_to_first_ms",
            "fg_ttft_ms","waiting_plateau_ms","max_kv_usage","preemption_delta","prefix_hits_delta",
            "time_to_preemption_s","abort_count_delta","status")}
        row.update(token_events=event_count, delivered_tokens=total_tokens, multi_token_events=multi,
                   max_token_gap_ms=max(gaps), near_preemption_max_gap_ms=max(nearby) if nearby else None,
                   median_scrape_ms=statistics.median(m["scrape_ms"] for m in ms),
                   max_scrape_ms=max(m["scrape_ms"] for m in ms),
                   median_sample_interval_ms=statistics.median((b["sample_ns"]-a["sample_ns"])/1e6 for a,b in zip(ms,ms[1:])))
        initial=t.get("initial_admitted_ns",t.get("trigger_ns"))
        row["max_waiting_after_initial"] = max(m["waiting"] for m in ms if m["sample_ns"]>=initial)
        marker=t.get("first_observed_preemption_ns")
        observed=next((m for m in ms if m["sample_ns"]==marker),{})
        row["running_at_preemption_observation"] = observed.get("running")
        row["waiting_at_preemption_observation"] = observed.get("waiting")
        row["first_observed_kv_drop_ns"] = t.get("first_observed_kv_drop_ns")
        row["first_observed_waiting_zero_after_release_ns"] = t.get("first_observed_waiting_zero_after_release_ns")
        analysis.append(row)
    if len(set(all_requests)) != len(all_requests) or set(all_requests) != set(events):
        raise ValueError("Request identity uniqueness/raw set mismatch")
    return analysis


def admission_plot(directory,trials,metrics,condition):
    selected=[t for t in trials if t["condition"]==condition]
    fig, axes=plt.subplots(3,len(selected),figsize=(5*len(selected),7),squeeze=False,sharex="col")
    colors={"FG sent":"#333333","Holder cancel":"#d97706","FG first token":"#15803d","Cohort cleanup":"#9ca3af"}
    for col,t in enumerate(selected):
        anchor=t["fg_send_ns"]
        left=min(-400,(t["release_ns"]-anchor)/1e6-200)
        right=(t["fg_first_ns"]-anchor)/1e6+200
        cleanup=min(r["cancel_ns"] for r in t["requests"] if r["cancel_ns"] and r["cancel_ns"]>t["release_ns"])
        ms=[m for m in metrics[t["trial_index"]] if left<=(m["sample_ns"]-anchor)/1e6<=right]
        x=[(m["sample_ns"]-anchor)/1e6 for m in ms]
        for row,key in enumerate(("kv_cache_usage_perc","running","waiting")):
            ax=axes[row,col]
            ax.step(x,[m[key]*(100 if row==0 else 1) for m in ms],where="post",color="#2563eb",linewidth=1.8)
            for label,stamp in (("FG sent",anchor),("Holder cancel",t["release_ns"]),("FG first token",t["fg_first_ns"]),("Cohort cleanup",cleanup)):
                ax.axvline((stamp-anchor)/1e6,color=colors[label],linestyle="--",label=label,alpha=.85)
            ax.grid(alpha=.2)
            if col==0:
                ax.set_ylabel(("KV usage (%)","Running","Waiting")[row])
            ax.set_xlim(left,right)
        axes[0,col].set_title(f'Pair {t["block_index"]}: TTFT {t["fg_ttft_ms"]:.0f} ms\nfree {t["trigger_free_blocks"]:.0f} / FG {t["fg_required_blocks"]} blocks')
        axes[2,col].set_xlabel("Time relative to FG send (ms)")
        axes[2,col].set_ylim(-.1,1.2)
        if condition==k.ADMISSION[0]:
            for ax in axes[:,col]:
                ax.axvspan(0,(t["release_ns"]-anchor)/1e6,color="#fbbf24",alpha=.12)
    axes[0,0].legend(fontsize=8,loc="lower left")
    fig.suptitle(condition+"\nClient/scrape timestamps; admission time unobserved",fontsize=12)
    fig.tight_layout()
    fig.savefig(directory/"plots"/(condition.lower()+".png"),dpi=160)
    plt.close(fig)


def growth_plot(directory,trials,metrics):
    pairs=sorted({t["block_index"] for t in trials if t["condition"] in k.GROWTH})
    fig,axes=plt.subplots(3,len(pairs),figsize=(5*len(pairs),7),squeeze=False,sharex="col")
    for col,block in enumerate(pairs):
        for condition,color in zip(k.GROWTH,("#2563eb","#dc2626")):
            t=next(t for t in trials if t["block_index"]==block and t["condition"]==condition)
            ms=metrics[t["trial_index"]]
            x=[(m["sample_ns"]-t["cohort_send_ns"])/1e9 for m in ms]
            axes[0,col].plot(x,[100*m["kv_cache_usage_perc"] for m in ms],color=color,label=condition)
            axes[1,col].step(x,[m["preemptions"]-t["idle_before"]["preemptions"] for m in ms],where="post",color=color)
            axes[2,col].step(x,[m["running"] for m in ms],where="post",color=color,label=condition+" running")
            axes[2,col].step(x,[m["waiting"] for m in ms],where="post",color=color,linestyle=":",label=condition+" waiting")
            if condition==k.GROWTH[1]:
                marker=(t["first_observed_preemption_ns"]-t["cohort_send_ns"])/1e9
                for ax in axes[:,col]:
                    ax.axvline(marker,color="#a16207",linestyle="--",label="First counter observation")
        axes[0,col].set_title(f"Pair {block}")
        axes[2,col].set_xlabel("Time since cohort send (s)")
        for row in range(3):
            axes[row,col].grid(alpha=.2)
            if col==0:
                axes[row,col].set_ylabel(("KV usage (%)","Preemption delta","Requests")[row])
    axes[0,0].legend(fontsize=7)
    axes[2,0].legend(fontsize=6,loc="center left")
    fig.suptitle("RUNNING growth\nCounter observation is not exact scheduler time")
    fig.tight_layout()
    fig.savefig(directory/"plots"/"growth-comparison.png",dpi=160)
    plt.close(fig)


def preemption_zoom(directory, pair=1, half_window_s=1.75):
    """Add one teaching figure from raw scrapes; exclude manual cleanup samples."""
    output=directory/"plots"/"preemption-zoom.png"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    trials=[json.loads(line) for line in (directory/"trials.jsonl").read_text().splitlines()]
    trial=next(t for t in trials if t["condition"]==k.GROWTH[1] and t["block_index"]==pair)
    with (directory/"metrics.csv").open() as f:
        rows=[{key:float(r[key]) for key in (
            "sample_ns","kv_cache_usage_perc","preemptions","running","waiting")}
              for r in csv.DictReader(f) if int(r["trial_index"])==trial["trial_index"]]
    baseline=trial["idle_before"]["preemptions"]
    index=next(i for i,m in enumerate(rows) if m["preemptions"]>baseline)
    if index==0:
        raise ValueError("Missing pre-event scrape")
    before,after=rows[index-1],rows[index]
    anchor=after["sample_ns"]
    if anchor!=trial["first_observed_preemption_ns"]:
        raise ValueError("Raw counter change disagrees with trial summary")
    cancellations=[r["cancel_ns"] for r in trial["requests"] if r["cancel_ns"] is not None]
    cleanup=min(cancellations) if cancellations else float("inf")
    samples=[m for m in rows if abs(m["sample_ns"]-anchor)<=half_window_s*1e9
             and m["sample_ns"]<cleanup]
    x=[(m["sample_ns"]-anchor)/1e9 for m in samples]
    cleanup_s=(cleanup-anchor)/1e9
    fig,axes=plt.subplots(4,1,figsize=(10.5,10),sharex=True)
    panels=(
        ("kv_cache_usage_perc",100,"KV cache usage (%)","#2563eb"),
        ("preemptions",1,"Preemption counter delta","#dc2626"),
        ("running",1,"Running requests","#7c3aed"),
        ("waiting",1,"Waiting requests","#15803d"),
    )
    for i,(key,scale,label,color) in enumerate(panels):
        ax=axes[i]
        values=[(m[key]-(baseline if key=="preemptions" else 0))*scale for m in samples]
        ax.step(x,values,where="post",color=color,linewidth=2.2)
        ax.scatter(x,values,color=color,s=9,zorder=3)
        ax.axvline(0,color="#a16207",linestyle="--",linewidth=1.6,
                   label="First observed Prometheus counter change")
        if cleanup_s<half_window_s:
            ax.axvspan(cleanup_s,half_window_s,color="#e5e7eb",alpha=.85)
        ax.set_ylabel(label,fontsize=11)
        ax.set_title(f"{i+1}. {label}",loc="left",fontsize=11,fontweight="bold",pad=7)
        ax.grid(axis="y",alpha=.25)
        ax.set_xlim(-half_window_s,half_window_s)
        ax.spines[["top","right"]].set_visible(False)
        old=(before[key]-(baseline if key=="preemptions" else 0))*scale
        new=(after[key]-(baseline if key=="preemptions" else 0))*scale
        change=f"{old:.2f}% → {new:.2f}% (≈92%)" if i==0 else f"{old:.0f} → {new:.0f}"
        ax.text(.04,.13 if i==0 else .72,change,transform=ax.transAxes,color=color,
                fontsize=13,fontweight="bold",bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
    axes[0].set_ylim(90,101.8)
    axes[0].set_yticks([92,94,96,98,100])
    axes[0].annotate(f"{before['kv_cache_usage_perc']*100:.2f}%",
                     xy=((before['sample_ns']-anchor)/1e9,before['kv_cache_usage_perc']*100),
                     xytext=(-.65,100.6),fontsize=10,
                     arrowprops=dict(arrowstyle="->",color="#2563eb"))
    axes[0].annotate(f"{after['kv_cache_usage_perc']*100:.2f}%",
                     xy=(0,after['kv_cache_usage_perc']*100),xytext=(.35,94.3),fontsize=10,
                     arrowprops=dict(arrowstyle="->",color="#2563eb"))
    axes[1].set_ylim(-.08,1.25)
    axes[1].set_yticks([0,1])
    axes[2].set_ylim(10.65,12.45)
    axes[2].set_yticks([11,12])
    axes[3].set_ylim(0,1.25)
    axes[3].set_yticks([0,.5,1])
    axes[3].set_xlabel("Time relative to first observed preemption counter change (s)",fontsize=11)
    if cleanup_s<half_window_s:
        axes[0].text((cleanup_s+half_window_s)/2,96,"Client cleanup\nstarted at +"
                     f"{cleanup_s:.3f} s\n\nSamples excluded",ha="center",va="center",fontsize=9,color="#4b5563")
    fig.suptitle("First Preemption: KV Exhaustion Forces One Running Request Back to Waiting",
                 fontsize=13,fontweight="bold",y=.985)
    fig.text(.5,.947,f"PRESSURE_GROWTH · Pair {pair} · Trial {trial['trial_index']} · "
             f"Counter first observed at {(anchor-trial['cohort_send_ns'])/1e9:.3f} s after cohort send",
             ha="center",fontsize=10)
    fig.text(.5,.925,"Dashed line = first Prometheus observation, NOT the exact scheduler preemption timestamp.",
             ha="center",fontsize=10,color="#92400e")
    fig.text(.5,.025,"Dots are raw scrapes; steps show sampled values. No request-level victim identity or allocation-failure time is measured.",
             ha="center",fontsize=9,color="#4b5563")
    fig.tight_layout(rect=(0,.045,1,.90),h_pad=1.3)
    with output.open("xb") as f:
        fig.savefig(f,format="png",dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(output),pair=pair,trial_index=trial["trial_index"],
                         relative_window_s=[-half_window_s,half_window_s],
                         cohort_window_s=[(anchor-trial["cohort_send_ns"])/1e9-half_window_s,
                                          (anchor-trial["cohort_send_ns"])/1e9+half_window_s],
                         last_pre_observation_s=(before["sample_ns"]-anchor)/1e9,
                         cleanup_relative_s=cleanup_s,
                         before=before,first_observed=after),indent=2))


def render_report(directory, metadata, trials, analyses):
    config=metadata.get("actual_scheduler_config",{})
    lines=["# KV pressure mechanism probes", "",f"Run: `{directory.name}`; all {len(trials)} trials passed protocol + raw replay.","",
           "## Actual server and workload", "",
           f"Qwen/Qwen3-0.6B, vLLM {metadata['vllm_version']}, GPU utilization 0.75, max_model_len 4096, batched tokens 2048. Prefix cache and chunked prefill enabled; FCFS; no external KV transfer.",
           f"Pool: {metadata['pool']}; usable tokens excluding null block: {metadata['usable_tokens']}; safety margin {metadata['margin_blocks']} blocks.",
           f"Scheduler: `{json.dumps(config)}`", "",
           f"Admission: {metadata['n_holders']} × 80-token holders, max output 3984; FG 3104 input + 1 output. Growth: N={metadata['n_growth']}, `{metadata['growth_targets']}`.","",
           "## Admission observations", "",
           "| Pair | Condition | Free before release (blocks) | FG blocks | Observed waiting span (ms) | Send→release (ms) | Release→first (ms) | TTFT (ms) |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for a in analyses:
        if a["condition"] in k.ADMISSION:
            trial=next(t for t in trials if t["trial_index"]==a["trial_index"])
            send_release=(trial["release_ns"]-trial["fg_send_ns"])/1e6
            lines.append(f"|{a['block_index']}|{a['condition']}|{a['trigger_free_blocks']:.0f}|{a['fg_required_blocks']}|{a['blocked_span_ms']:.1f}|{send_release:.1f}|{a['release_to_first_ms']:.1f}|{a['fg_ttft_ms']:.1f}|")
    lines += ["", "Both conditions first reached the same block-based high-pressure window. All holders were admitted with Waiting=0 before intervention. A1 showed a foreground waiting plateau with no first token; closing one holder stream dropped KV occupancy, then the FG produced a token. A2 released first, confirmed free blocks above FG requirement plus margin, then injected. No admission trial preempted.","",
              "Direct observation supports the admission hypothesis. Attribution to the full-known-input gate is mechanism-consistent inference from the audited code and joint controls. We did **not** instrument allocate_slots(), observe its return value, or measure exact scheduler admission time. `waiting_capacity` alone is not KV-specific. A1 post-release free headroom includes allowance for FG blocks already consumed before a scrape.","",
              "## Growth observations", "",
              "| Pair | Condition | Max KV (%) | Preemptions | First observed counter change (s) | Running / Waiting at counter change | Largest nearby token-event gap (ms) |",
              "|---|---|---:|---:|---:|---|---:|"]
    for a in analyses:
        if a["condition"] in k.GROWTH:
            rw=f"{a['running_at_preemption_observation']:.0f} / {a['waiting_at_preemption_observation']:.0f}" if a['running_at_preemption_observation'] is not None else "—"
            observed_time=f"{a['time_to_preemption_s']:.3f}" if a['time_to_preemption_s'] is not None else "—"
            nearby_gap=f"{a['near_preemption_max_gap_ms']:.1f}" if a['near_preemption_max_gap_ms'] is not None else "—"
            lines.append(f"|{a['block_index']}|{a['condition']}|{100*a['max_kv_usage']:.2f}|{a['preemption_delta']:.0f}|{observed_time}|{rw}|{nearby_gap}|")
    lines += ["", "All N requests initially ran simultaneously, with Waiting=0 and no preemption. Controls finished at their exact output limits without preemption. Pressure trials naturally increased the preemption counter near full occupancy; remaining requests were cancelled after a one-second observation window. This supports growth-triggered preemption, without directly observing the individual block allocation failure or victim identity. Prefix hits after preemption can be reuse of the same request's cached prefix, not cross-request sharing. No reset was performed. Counter timestamps can lag the scheduler event. Token gaps are client-observed secondary effects and cannot identify the victim by themselves.", "",
              "## Integrity and limits", "",
              "Every trial began/ended at Running=Waiting=0 and idle KV. Cancellation was verified by client stream closure plus server Running/KV recovery and drain. request_success{abort} need not increase on disconnect: the local OutputProcessor removes state before finished-request statistics. Completed streams had exact final usage, length reason and matching prompt count. Unique early nonces and distinct first KV blocks prevent shared prefixes; admission/control prefix hits and external-transfer deltas are zero. Prefix cache remains enabled. No unexpected EOS or unrelated request failures. Raw token IDs were replayed in order, including multi-token SSE events; cancelled streams intentionally have no final usage.",
              f"Raw events: {sum(a['token_events'] for a in analyses)} token-bearing events; {sum(a['multi_token_events'] for a in analyses)} multi-token events. Median scrape latency range: {min(a['median_scrape_ms'] for a in analyses):.2f}–{max(a['median_scrape_ms'] for a in analyses):.2f} ms. Median sampling interval range: {min(a['median_sample_interval_ms'] for a in analyses):.2f}–{max(a['median_sample_interval_ms'] for a in analyses):.2f} ms.",
              "Prometheus values are sampled observations and may be stale relative to scheduler activity. Request-level admission, exact retained block ownership and exact preemption time are not instrumented. Trials are small mechanism probes, not configuration optimization or a throughput benchmark.", "",
              f"Shutdown: `{metadata.get('shutdown')}`", "",
              "## Artifacts", "", "`manifest.json`, `metadata.json`, `server-info.json`, `server.log`, `metrics-start.txt`, `metrics-end.txt`, `environment-start.txt`, `environment-end.txt`, `trials.jsonl`, `trials.csv`, `metrics.csv`, `events.jsonl`, `analysis.csv`, `raw-validation.json`, `plots/`.", "",
              "![Blocked then release](plots/blocked_then_release.png)", "![Release then inject](plots/release_then_inject.png)", "![Growth](plots/growth-comparison.png)"]
    log=(directory/"server.log").read_text()
    before_shutdown=log.split("[shutdown]",1)[0]
    lines += ["", "## Execution audit", "",
              f"Server ERROR lines before shutdown: {sum(' ERROR ' in line for line in before_shutdown.splitlines())}; OOM text present: {'out of memory' in log.lower() or 'OutOfMemoryError' in log}; forced worker kill logged: {'force killing' in log}."]
    if "Numba needs NumPy" in log:
        lines.append("Startup warned that installed Numba cannot import with NumPy 2.5 (Numba requires ≤2.4). Inference ran successfully; dependencies were not changed.")
    if "EngineDeadError" in log.split("[shutdown]",1)[-1]:
        lines.append("After the requested shutdown, the async output handler logged EngineDeadError as EngineCore exited. This is preserved as a shutdown-time error, not claimed to be an error-free log. See lifecycle timestamps and exit code above.")
    if metadata.get("calibration_run"):
        cal_dir=directory.parent/metadata['calibration_run']
        cal_raw=json.loads((cal_dir/'raw-validation.json').read_text())
        lines += [f"Calibration: ../{metadata['calibration_run']}/ ({cal_raw['trials']} trials passed raw replay)."]
        if (cal_dir/'cancellation-smoke.json').exists():
            cancellation=json.loads((cal_dir/'cancellation-smoke.json').read_text())
            cal_meta=json.loads((cal_dir/'metadata.json').read_text())
            cal_pool=k.Pool(**cal_meta['pool'])
            released=cal_pool.free(cancellation['after'])-cal_pool.free(cancellation['pre_cancel'])
            lines.append(f"Cancellation smoke: Running 1→0, {released:.0f} blocks released, stable idle confirmed within {cancellation['cancel_to_idle_ms']:.0f} ms (includes three-scrape confirmation).")
        if directory.name=="formal-20261013":
            lines += ["Earlier attempts are preserved in ../smoke-20261013-1050/ (missing scheduler telemetry, no workload) and ../smoke-20261013-b/ (invalid client-side requirement that disconnect abort must increment request_success{abort}). Neither is formal evidence.",
                      "The formal server added a 30-second shutdown grace period and API-first SIGINT; its inference and scheduler configuration matched calibration. Unit tests: 37 passed (11 new protocol tests), saved in unit-tests.txt."]
        lines += ["Loopback-only dev routing was enabled solely for GET /server_info. No mutation endpoint, cache reset, profiler, package install or installed-vLLM source modification was used.",
                  f"Formal: {len(trials)}/{len(trials)} passed; raw validation: raw-validation.json. Base seed 20261013; growth permutation uses base seed + 1; actual schedule is in manifest.json. Source snapshots and test output are also preserved in this run directory."]
    (directory/"report.md").write_text("\n".join(lines)+"\n")


def main(directory):
    metadata=json.loads((directory/"metadata.json").read_text())
    trials,metrics,events=read_data(directory)
    pool=k.Pool(**metadata["pool"])
    analyses=replay(trials,metrics,events,pool)
    with (directory/"analysis.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(analyses[0]))
        writer.writeheader()
        writer.writerows(analyses)
    k.lifecycle.write_json(directory/"raw-validation.json",dict(status="passed",trials=len(trials),requests=len(events),
                                                               events=sum(map(len,events.values()))))
    (directory/"plots").mkdir(exist_ok=False)
    for condition in k.ADMISSION:
        admission_plot(directory,trials,metrics,condition)
    growth_plot(directory,trials,metrics)
    render_report(directory,metadata,trials,analyses)
    print(json.dumps(analyses,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--preemption-zoom-only",action="store_true",
                        help="Add only preemption-zoom.png; leave existing figures and report intact")
    parser.add_argument("--pair",type=int,default=1,help="Pressure pair for the zoom (default: 1)")
    args=parser.parse_args()
    if args.preemption_zoom_only:
        preemption_zoom(args.directory,pair=args.pair)
    else:
        main(args.directory)
