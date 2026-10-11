"""Offline protocol tests: no inference endpoint or GPU required."""

import asyncio
import csv
import inspect
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from experiments import common, concurrency
from experiments import prefill_interference as runner
from experiments import plot_prefill_interference as plotter


def chunk(ids=None, *, content=None, role=None, finish=None, usage=None, prompt=None):
    return NS(choices=[] if usage else [NS(token_ids=ids, finish_reason=finish,
                                           delta=NS(content=content, role=role))],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None,
              prompt_token_ids=list(range(prompt)) if prompt is not None else None)


class FakeStream:
    def __init__(self, events, pause=0):
        self.events = iter(events)
        self.closed = False
        self.pause = pause

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self.pause)
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def fake_client(create):
    return NS(chat=NS(completions=NS(create=create)))


def snapshot():
    return {"timestamp": 1.0, "counters": {k: 0 for k in common.COUNTER_SELECTORS},
            "gauges": {k: 0 for k in common.GAUGE_SELECTORS}}


def synthetic_events(trial_index=1):
    events = []
    for bg in range(8):
        for index, ms in enumerate((-450, -350, -250, -150, -50, 50, 250, 350, 450)):
            events.append(dict(run_id="synthetic", block_index=1, trial_index=trial_index,
                               condition="baseline", role="background", background_id=bg,
                               event_index=index, delta_token_count=1, token_ids=[index],
                               received_ns=int((2000 + ms) * 1e6),
                               relative_to_alignment_ms=ms, alignment_ns=2_000_000_000))
    return events


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_event_types_and_incremental_ids(self):
        request = runner.Request("background", 0, 577, 512)
        events = [chunk(role="assistant", prompt=577), chunk(content=""),
                  chunk([1], content=""), chunk(list(range(61))), chunk([62, 63, 64]),
                  chunk(finish="length"), chunk(usage=(577, 65))]
        stream = FakeStream(events)
        create = AsyncMock(return_value=stream)
        await runner.stream_request(fake_client(create), request)
        self.assertTrue(stream.closed)
        self.assertTrue(request.valid)
        self.assertEqual(len(request.events), 7)
        self.assertEqual([e["cumulative_token_count"] for e in request.events], [0, 0, 1, 62, 65, 65, 65])
        self.assertTrue(request.ready.is_set())
        self.assertEqual(request.prompt_count, 577)
        self.assertEqual(request.events[2]["has_content"], False)
        self.assertGreaterEqual(request.ttft_s, 0)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"return_token_ids": True})
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertNotIn("logprobs", kwargs)

    async def test_usage_and_prompt_failures_close_stream(self):
        cases = [([chunk([1]), chunk(finish="stop")], 577, "Final usage"),
                 ([chunk([1]), chunk(finish="stop"), chunk(usage=(577, 2))], 577, "mismatch"),
                 ([chunk([1]), chunk(finish="stop"), chunk(usage=(2048, 1))], 3072, "Invalid Long"),
                 ([chunk([1], prompt=576), chunk(finish="stop"), chunk(usage=(577, 1))], 577, "disagrees")]
        for events, target, message in cases:
            with self.subTest(message=message):
                stream = FakeStream(events)
                request = runner.Request("foreground", None, target, 1)
                with self.assertRaisesRegex(ValueError, message):
                    await runner.stream_request(fake_client(AsyncMock(return_value=stream)), request)
                self.assertTrue(stream.closed)
                self.assertFalse(request.valid)
                self.assertTrue(request.error)

    async def test_stream_cancellation_closes(self):
        stream = FakeStream([chunk([1])], pause=10)
        request = runner.Request("foreground", None, 577, 1)
        task = asyncio.create_task(runner.stream_request(fake_client(AsyncMock(return_value=stream)), request))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        self.assertIn("CancelledError", request.error)

    async def test_trigger_once_and_alignment_for_all_conditions(self):
        for condition in runner.CONDITIONS:
            with self.subTest(condition=condition):
                bgs = [runner.Request("background", i, 577, 512) for i in range(8)]
                counts = [65, 68, 64, 66, 64, 67, 65, 69]
                fg = None if condition == "baseline" else runner.Request(
                    "foreground", None, 3072 if condition == "long" else 577, 1)
                create = AsyncMock(return_value=FakeStream([
                    chunk([1], finish="length"), chunk(usage=(3072 if condition == "long" else 577, 1))]))
                alignment = {}
                task = asyncio.create_task(runner.coordinate(fake_client(create), bgs, fg, alignment))
                for r, count in zip(bgs[:-1], counts[:-1]):
                    runner.record_event(r, chunk(list(range(count))), 1)
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                create.assert_not_called()
                runner.record_event(bgs[-1], chunk(list(range(counts[-1]))), 1)
                await task
                self.assertEqual(alignment["trigger_counts"], counts)
                runner.record_event(bgs[-1], chunk([70]), 2)
                self.assertEqual(alignment["trigger_count"], 1)
                if fg is None:
                    create.assert_not_called()
                    self.assertNotIn("foreground_request_start_ns", alignment)
                    self.assertEqual(alignment["trigger_ns"], alignment["alignment_ns"])
                else:
                    create.assert_awaited_once()
                    self.assertGreaterEqual(alignment["foreground_request_start_ns"], alignment["trigger_ns"])
                    self.assertEqual(fg.start_ns, alignment["alignment_ns"])

    async def test_premature_background_cancels_cohort(self):
        streams = []
        async def create(**kwargs):
            events = [chunk([1], finish="stop"), chunk(usage=(577, 1))]
            stream = FakeStream(events, pause=0 if not streams else 10)
            streams.append(stream)
            return stream
        bgs = [runner.Request("background", i, 577, 512) for i in range(8)]
        with patch.object(runner, "get_metrics_snapshot", AsyncMock(return_value=snapshot())):
            with self.assertRaises(ExceptionGroup) as caught:
                await asyncio.wait_for(runner.run_cohort(fake_client(create), bgs, None, {}, None, []), 1)
        self.assertIn("before trigger", runner.error_text(caught.exception))
        self.assertTrue(all(s.closed for s in streams))

    async def test_foreground_failure_propagates(self):
        bgs = [runner.Request("background", i, 577, 512) for i in range(8)]
        for r in bgs:
            r.cumulative = 64
            r.ready.set()
        fg = runner.Request("foreground", None, 577, 1)
        with self.assertRaisesRegex(RuntimeError, "foreground failure"):
            await runner.coordinate(fake_client(AsyncMock(side_effect=RuntimeError("foreground failure"))), bgs, fg, {})
        self.assertIn("foreground failure", fg.error)

    async def test_failed_trial_records_metrics_failure(self):
        identity = dict(run_id="synthetic", block_index=1, trial_index=1, condition="baseline", seed=1, smoke=True)
        with patch.object(runner, "wait_for_idle", AsyncMock(side_effect=RuntimeError("metrics unreachable"))):
            row, events, metrics = await runner.run_trial(None, None, identity, 1)
        self.assertEqual(row["status"], "failed")
        self.assertIn("metrics unreachable", row["error"])
        self.assertFalse(row["token_ids_usage_all_match"])
        self.assertEqual(events, [])
        self.assertEqual(metrics, [])
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=runner.TRIAL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    async def test_timeout_and_sampler_failure_leave_no_tasks(self):
        bgs = [runner.Request("background", i, 577, 512) for i in range(8)]
        streams = []
        async def create(**kwargs):
            stream = FakeStream([chunk([1])], pause=10)
            streams.append(stream)
            return stream
        for fail_metrics in (False, True):
            with self.subTest(fail_metrics=fail_metrics):
                scrape = AsyncMock(side_effect=RuntimeError("scrape failed")) if fail_metrics else AsyncMock(return_value=snapshot())
                with patch.object(runner, "get_metrics_snapshot", scrape):
                    with self.assertRaises((TimeoutError, ExceptionGroup)):
                        await asyncio.wait_for(runner.run_cohort(fake_client(create), bgs, None, {}, None, []),
                                               1 if fail_metrics else 0.02)
                self.assertTrue(all(s.closed for s in streams))
                self.assertFalse([t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()])


class AnalysisTests(unittest.TestCase):
    def test_gap_and_singleton_pairs(self):
        events = [dict(event_index=i, received_ns=ns, relative_to_alignment_ms=ms, delta_token_count=d)
                  for i, (ns, ms, d) in enumerate([(0, -1, 0), (1_000_000, 0, 1),
                                                  (4_000_000, 3, 3), (6_000_000, 5, 1),
                                                  (11_000_000, 10, 1)])]
        gaps = plotter.event_gaps(events)
        self.assertEqual([g["gap_ms"] for g in gaps], [3, 2, 5])
        self.assertEqual([g["relative_ms"] for g in gaps], [3, 5, 10])
        self.assertEqual([g["singleton_pair"] for g in gaps], [False, False, True])
        distribution = runner.event_distribution(events)
        self.assertEqual(distribution["token_bearing_events"], 4)
        self.assertEqual(distribution["singleton_event_ratio"], 0.75)
        self.assertEqual(distribution["ge3_token_event_ratio"], 0.25)

    def test_stall_windows_and_invalid_samples(self):
        trial = dict(run_id="synthetic", trial_index=1, condition="baseline", status="success", alignment_ns=2_000_000_000)
        row, _ = plotter.analyze_trial(trial, synthetic_events())
        self.assertTrue(row["analysis_valid"])
        self.assertEqual(row["pre_median_gap_ms"], 100)
        self.assertEqual(row["post_peak_gap_ms"], 200)
        self.assertEqual(row["trial_stall_ratio"], 2)
        # A single-request outlier must not become the trial's summary.
        events = synthetic_events()
        events[7]["received_ns"] += 100_000_000
        self.assertEqual(plotter.analyze_trial(trial, events)[0]["trial_stall_ratio"], 2)
        for bad in (events[:9], [e for e in events if e["relative_to_alignment_ms"] < 0]):
            invalid, _ = plotter.analyze_trial(trial, bad)
            self.assertFalse(invalid["analysis_valid"])
            self.assertIsNone(invalid["trial_stall_ratio"])
            self.assertTrue(invalid["invalid_reason"])

    def test_bin_preserves_request_peak(self):
        gaps = {i: [{"relative_ms": 1, "gap_ms": 5}, {"relative_ms": 2, "gap_ms": 80}] for i in range(8)}
        self.assertEqual(plotter.timeline_bins(gaps, 20, 0, 100), {0: 80})

    def test_block_schedule_and_common_api(self):
        schedule = runner.make_schedule(10, 20261011)
        self.assertEqual(schedule, runner.make_schedule(10, 20261011))
        self.assertEqual(len(schedule), 30)
        for block in range(1, 11):
            self.assertCountEqual([c for b, c in schedule if b == block], runner.CONDITIONS)
        self.assertEqual(len(runner.make_schedule(1, 20261011)), 3)
        for name in ("measure_streaming_request", "measure_streaming_request_async"):
            self.assertEqual(list(inspect.signature(getattr(common, name)).parameters), ["client", "model", "prompt", "max_tokens"])
        self.assertEqual(runner.Request("background", 0, 577, 512, client_request_id="abc").prompt,
                         f"request-abc\n{concurrency.FIXED_INSTRUCTION}\n{concurrency.FIXED_BODY}")
        subprocess.run(["git", "check-ignore", "results/prefill-interference-synthetic-events.jsonl"],
                       cwd=ROOT, check=True, capture_output=True)

    def test_jsonl_csv_and_three_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefill-interference-synthetic-trials.csv"
            events_path = path.with_name(path.name.replace("-trials.csv", "-events.jsonl"))
            metrics_path = path.with_name(path.name.replace("-trials.csv", "-metrics.csv"))
            with path.open("w", newline="") as tf, events_path.open("w") as ef, metrics_path.open("w", newline="") as mf:
                tw = csv.DictWriter(tf, fieldnames=runner.TRIAL_COLUMNS)
                mw = csv.DictWriter(mf, fieldnames=runner.METRIC_COLUMNS)
                tw.writeheader()
                mw.writeheader()
                for index, condition in enumerate(runner.CONDITIONS, 1):
                    row = dict(run_id="synthetic", block_index=1, trial_index=index, condition=condition,
                               status="success", alignment_ns=2_000_000_000,
                               foreground_ttft_s=None if condition == "baseline" else 0.1)
                    events = synthetic_events(index)
                    for event in events:
                        event["condition"] = condition
                    metric = {k: 0 for k in runner.METRIC_COLUMNS}
                    metric.update(run_id="synthetic", trial_index=index, condition=condition)
                    runner.write_trial(row, events, [metric], tw, ef, mw)
            rows, outputs = plotter.analyze_file(path)
            self.assertTrue(all(r["analysis_valid"] for r in rows))
            self.assertEqual(len(outputs), 4)
            for output in outputs[1:]:
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with outputs[0].open() as source:
                self.assertEqual(csv.DictReader(source).fieldnames, plotter.ANALYSIS_COLUMNS)
            # Sparse/failed smoke data still produces all plots and invalid CSV rows.
            events_path.write_text("")
            rows, outputs = plotter.analyze_file(path)
            self.assertTrue(all(not r["analysis_valid"] for r in rows))
            self.assertTrue(all(p.exists() for p in outputs))


if __name__ == "__main__":
    unittest.main()
