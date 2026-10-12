"""Pure sweep planning, integrity and interval aggregation; no fake server."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import token_budget_sweep as sweep
import prefill_interference as runner
import plot_token_budget as plotter
from test_prefill_interference import synthetic_events


def complete_baseline_fixture(tokens=508, start_ms=-900, finish_ms=700, end_ms=None,
                              first_token_ms=-800):
    """Complete raw records for eight successful requests; no server/mock system."""
    alignment = 2_000_000_000
    end_ms = finish_ms + 1 if end_ms is None else end_ms
    identity = dict(run_id="synthetic", block_index=1, trial_index=1, condition="baseline")
    summaries, events = [], []
    for bg in range(8):
        request_id = f"bg{bg}"
        summaries.append(dict(client_request_id=request_id, role="background", background_id=bg,
                              completion_tokens=tokens, generated_ids_count=tokens, actual_prompt_tokens=577,
                              request_start_ns=alignment+int(start_ms*1e6),
                              request_end_ns=alignment+int(end_ms*1e6),
                              valid=True, error="", finish_reason="stop"))
        for index in range(tokens + 1):
            usage = index == tokens
            ms = end_ms if usage else first_token_ms + (finish_ms-first_token_ms)*index/(tokens-1)
            events.append(dict(**identity, client_request_id=request_id, role="background", background_id=bg,
                               alignment_ns=alignment, received_ns=alignment+round(ms*1e6),
                               relative_to_alignment_ms=ms, event_index=index,
                               token_ids=[] if usage else [index], delta_token_count=0 if usage else 1,
                               cumulative_token_count=min(index+1,tokens), has_usage=usage,
                               usage_completion_tokens=tokens if usage else None,
                               usage_prompt_tokens=577 if usage else None,
                               finish_reason="stop" if index == tokens-1 else None))
    row = dict(**identity, status="success", alignment_ns=alignment, trigger_count=1,
               **{f"trigger_count_bg{i}": 64 for i in range(8)}, request_summaries=json.dumps(summaries),
               prompt_tokens_local_cache_hit=0, external_kv_transfer=0, preemptions_total=0,
               generation_tokens_total=8*tokens, prompt_tokens_total=8*577,
               idle_running=0, idle_waiting=0, idle_kv_cache_usage_perc=0)
    metrics = [dict(num_requests_waiting=0, waiting_capacity=0, waiting_deferred=0, kv_cache_usage_perc=0.2)]
    return row, events, metrics


class SweepTests(unittest.TestCase):
    def test_508_tokens_covering_window_is_valid(self):
        sweep.validate_trial(*complete_baseline_fixture(tokens=508))

    def test_natural_stop_inside_post_window_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "lifecycle does not cover"):
            sweep.validate_trial(*complete_baseline_fixture(tokens=508, finish_ms=400))

    def test_delayed_final_usage_cannot_hide_early_stop(self):
        with self.assertRaisesRegex(ValueError, "finishes before post window ends"):
            sweep.validate_trial(*complete_baseline_fixture(finish_ms=400, end_ms=700))

    def test_pre_window_coverage_is_required(self):
        with self.assertRaisesRegex(ValueError, "lifecycle does not cover"):
            sweep.validate_trial(*complete_baseline_fixture(start_ms=-400, first_token_ms=-390))

    def test_stop_at_post_window_boundary_is_valid(self):
        sweep.validate_trial(*complete_baseline_fixture(finish_ms=500))

    def test_raw_trigger_progress_and_upper_bound_are_required(self):
        with self.assertRaisesRegex(ValueError, "raw trigger progress"):
            sweep.validate_trial(*complete_baseline_fixture(first_token_ms=1))
        with self.assertRaisesRegex(ValueError, "max_tokens upper bound"):
            sweep.validate_trial(*complete_baseline_fixture(tokens=513))

    def test_uv_sigint_exit_encoding_requires_owned_signal(self):
        self.assertTrue(sweep.expected_shutdown_returncode(130, ["SIGINT"]))
        self.assertTrue(sweep.expected_shutdown_returncode(-2, ["SIGINT"]))
        self.assertTrue(sweep.expected_shutdown_returncode(0, ["SIGINT"]))
        self.assertFalse(sweep.expected_shutdown_returncode(130, []))
        self.assertFalse(sweep.expected_shutdown_returncode(1, ["SIGINT"]))
        self.assertFalse(sweep.expected_shutdown_returncode(0, ["SIGINT", "SIGTERM", "SIGKILL"]))

    def test_synthetic_formal_summaries_and_interval_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "results/token-budget-sweep/synthetic"
            directory.mkdir(parents=True)
            rows = []
            for budget in plotter.BUDGETS:
                block_dir = directory / f"budget-{budget}"
                block_dir.mkdir()
                trials, raw = [], []
                for index, (block, condition) in enumerate(runner.make_schedule(10, sweep.SEED, sweep.CONDITIONS), 1):
                    trial = dict(run_id="synthetic", block_index=block, trial_index=index,
                                 condition=condition, status="success", alignment_ns=2_000_000_000,
                                 foreground_ttft_s=0.2 if condition == "long" else None,
                                 request_summaries=json.dumps([dict(role="foreground", actual_prompt_tokens=3070)]
                                                              if condition == "long" else []),
                                 prompt_tokens_local_cache_hit=0, external_kv_transfer=0, preemptions_total=0)
                    events = synthetic_events(index)
                    for event in events:
                        event.update(condition=condition, block_index=block)
                    raw.extend(events)
                    trials.append(trial)
                    rows.append(sweep.analyze_extended(trial, events, budget))
                sweep.write_csv(block_dir / "trials.csv", trials)
                sweep.write_csv(block_dir / "metrics.csv", [dict(num_requests_waiting=0,
                                waiting_capacity=0, waiting_deferred=0, kv_cache_usage_perc=0.25)])
                (block_dir / "events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in raw))
            plotter.summarize_and_plot(directory, rows)
            self.assertEqual(len(plotter.read_csv(directory / "summary.csv")), 6)
            self.assertEqual(len(plotter.read_csv(directory / "paired-comparisons.csv")), 30)
            self.assertEqual(len(plotter.read_csv(directory / "paired-summary.csv")), 3)
            paths = json.loads((directory / "plots.json").read_text())
            self.assertEqual(len(paths), 6)
            for path in paths:
                self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_plan_and_reproducible_paired_blocks(self):
        self.assertEqual(sweep.budget_plan(),(2048,4096,1024))
        self.assertEqual(sweep.budget_plan(True),(2048,))
        schedule=runner.make_schedule(10,sweep.SEED,sweep.CONDITIONS)
        self.assertEqual(schedule,runner.make_schedule(10,sweep.SEED,sweep.CONDITIONS))
        self.assertEqual(len(schedule),20)
        for block in range(1,11):
            self.assertCountEqual([c for b,c in schedule if b==block],sweep.CONDITIONS)
        self.assertEqual({tuple(c for b,c in schedule if b==i) for i in range(1,11)},
                         {("baseline","long"),("long","baseline")})
        with self.assertRaises(ValueError):
            runner.make_schedule(0,sweep.SEED,sweep.CONDITIONS)

    def test_output_paths_and_no_overwrite(self):
        self.assertEqual(sweep.output_directory(Path("results"),"run-1",True),Path("results/smoke-run-1"))
        self.assertEqual(sweep.output_directory(Path("results"),"run-1"),Path("results/run-1"))
        with self.assertRaises(ValueError):
            sweep.output_directory(Path("results"),"../old")
        with tempfile.TemporaryDirectory() as tmp:
            path=sweep.output_directory(tmp,"run")
            path.mkdir(exist_ok=False)
            with self.assertRaises(FileExistsError):
                path.mkdir(exist_ok=False)

    def test_metadata_serialization_and_unknowns(self):
        log="""non-default args: {'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.75}
Using max model len 4096
Initializing a V1 LLM engine (v0.31.0) enable_prefix_caching=True, enable_chunked_prefill=True
GPU KV cache size: 37,200 tokens, Maximum concurrency for 4,096 tokens per request: 9.08x
Available KV cache memory: 3.97 GiB"""
        data=sweep.startup_metadata("run",4096,123,["serve","4096"],Path("server.log"),log,"now")
        self.assertEqual(data["kv_cache_capacity_tokens"],37200)
        self.assertEqual(data["observed_max_num_batched_tokens"],4096)
        self.assertTrue(data["prefix_cache_enabled"])
        self.assertIsNone(data["max_num_seqs"])
        self.assertIsNone(data["async_scheduling"])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"metadata.json"
            sweep.write_json(path,data)
            self.assertEqual(json.loads(path.read_text()),data)
        unknown=sweep.startup_metadata("run",1024,1,[],Path("server.log"),"","now")
        self.assertIsNone(unknown["observed_max_num_batched_tokens"])
        self.assertIsNone(unknown["kv_cache_capacity_tokens"])

    def test_strict_threshold_interval_count_and_duration(self):
        row=dict(run_id="synthetic",block_index=1,trial_index=1,condition="baseline",
                 status="success",alignment_ns=2_000_000_000,foreground_ttft_s=None)
        events=synthetic_events()
        result=sweep.analyze_extended(row,events,1024)
        self.assertEqual(result["significant_stall_count"],0) # exactly 2x is excluded
        # Move the 250 ms receipt to 251: actual interval [50,251] is 201 ms.
        for event in events:
            if event["relative_to_alignment_ms"]==250:
                event["relative_to_alignment_ms"]=251
                event["received_ns"]+=1_000_000
        result=sweep.analyze_extended(row,events,1024)
        self.assertEqual(result["significant_stall_count"],1)
        self.assertEqual(result["significant_stall_total_duration_ms"],201)
        self.assertEqual(result["max_stall_gap_ms"],201)
        self.assertEqual(result["stall_ratio"],2.01)

    def test_failure_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError,"bad stream"):
            sweep.validate_trial(dict(status="failed",error="bad stream"),[],[])


if __name__=="__main__":
    unittest.main()
