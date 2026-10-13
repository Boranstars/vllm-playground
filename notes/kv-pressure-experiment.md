# KV pressure mechanism probes — protocol

Fixed Qwen/Qwen3-0.6B server: GPU utilization .75, model length 4096,
2048 batched tokens, FCFS, prefix cache and chunked prefill enabled. No source
or dependency changes. Owned smoke server is shut down before a fresh formal
server; all trials drain to idle. Seed 20261013, three randomized paired blocks
per probe. Each prompt has an early fresh nonce and is counted by `/tokenize`.

Read capacity from this server's cache_config_info and startup log, cross-check
block count × block size. Usable pool excludes the null block. Sample metrics
every 50 ms; save timestamps and scrape latency, and use counter deltas.

Admission: derive holder count from usable capacity and model length. Short
prompts target 80 tokens; holders can grow to 4064 context tokens. Foreground
has exactly 3104 input tokens and one output token. Trigger only when all
holders run, waiting=0, no preemption, free blocks < FG blocks − margin, and
free blocks + estimated designated-holder blocks > FG blocks + margin.
Margin is 1% of pool, clamped to 16–32 blocks. BLOCKED_THEN_RELEASE requires
at least three consecutive waiting scrapes over ≥100 ms before cancellation;
stop if not observed within 600 ms. RELEASE_THEN_INJECT first verifies restored
headroom and one fewer running holder. A1 also verifies restored headroom
allowing for foreground allocation, since the foreground can consume freed
blocks before the next scrape. Abort all remaining holders after foreground
completion; verify Running/Waiting/KV idle drain. Record abort counters but do not
require increments: local OutputProcessor.abort_requests removes state before
finished-request statistics, so disconnect aborts need not increment this metric.
Stop on invalid trials.

Growth: start at N=12; increase minimally only if 110% capacity demand cannot
fit under the per-request limit. CONTROL uses block-rounded aggregate target
≤70%; PRESSURE uses ≥110%. Both require initial Running=N, Waiting=0 and zero
preemption. Pressure stops one second after first observed counter increment.
Control must finish with length reason and exact max_tokens. No reset or swap.

Calibration runs cancellation, both admission conditions, and both growth
conditions before formal execution. Unit tests cover arithmetic, boundaries,
validity, event counts, cancellation and reproducible scheduling. Preserve raw
data on failure. Plot three mechanism timelines and report all trials, with
raw-event integrity replay.

Interpretation: waiting_capacity is not KV-specific. Evidence is the joint
headroom bound, waiting plateau, lack of sequence/token-budget interference,
release response and zero preemption. Neither admission time nor allocate_slots
return values are directly measured. Preemption time is first counter observation.

Configuration telemetry uses `VLLM_SERVER_DEV_MODE=1` with `--host 127.0.0.1`,
only to GET `/server_info?config_format=json`; no modifying dev endpoint is used.
This is fixed across the successful smoke and formal server launches.

## 2026-10-13 completed run

See `results/kv-pressure/formal-20261013/report.md` and its three plots. Formal:
2311 blocks × 16 = 36976 tokens (36960 usable), margin 24 blocks, max_num_seqs
256, async scheduling on. All twelve trials and 384316 raw SSE events passed.
All six admission triggers had 168 free blocks versus 194 required by FG.
A1 send→release: 203–260 ms; release→first: 269–282 ms. A2 TTFT: 264–293 ms,
no sustained waiting plateau. All three growth controls completed without
preemption, at peak KV 69.61%; all three pressure trials had one preemption at
peak sampled KV 99.74–99.78%. Running/Waiting at first counter observation was
11/1, after all twelve initially ran. Formal growth output limits were 2064 and
3312 tokens, recalculated from fresh capacity. No prefix hits or external KV
transfer occurred in formal trials.

Calibration `smoke-20261013-c` passed four trials and cancellation. Earlier
`smoke-20261013-1050` and `smoke-20261013-b` preserve telemetry/validation
implementation failures. No workload adjustment was needed. Default zero
shutdown timeout caused forced worker termination in smoke; formal used 30 s
grace plus SIGINT to the API process first, and exited 0 without force kill.
The inference/scheduler configuration was identical. A shutdown-time
EngineDeadError and startup Numba/NumPy warning remain in the logs; no ERROR
occurred during formal measurement. No dependencies were changed.

Reproduction (use the IDE-configured interpreter; each run ID must be new):

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/kv_pressure_experiment.py --phase smoke --blocks 1 --run-id NEW_SMOKE
.venv/bin/python experiments/plot_kv_pressure.py results/kv-pressure/NEW_SMOKE
.venv/bin/python experiments/kv_pressure_experiment.py --phase formal --blocks 3 --calibration-run NEW_SMOKE --run-id NEW_FORMAL
.venv/bin/python experiments/plot_kv_pressure.py results/kv-pressure/NEW_FORMAL
```

Local source checked: vllm/v1/core/sched/scheduler.py (RUNNING allocate_slots,
WAITING full_sequence_must_fit, _preempt_request), kv_cache_manager.py
(full known sequence check then chunk allocation), block_pool.py (null block
and usage denominator), config/scheduler.py (reserve_full_isl=True, watermark=0).
