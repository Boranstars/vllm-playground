"""Offline mechanism protocol tests, no mocked scheduler behaviour."""
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
import kv_pressure_experiment as k


def chunk(ids=(), usage=None, finish=None):
    return NS(choices=[] if usage else [NS(token_ids=list(ids), finish_reason=finish,
                                         delta=NS(content="", role=None))],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None,
              prompt_token_ids=None)


class FakeStream:
    def __init__(self, events, delay=0):
        self.events, self.delay, self.closed = iter(events), delay, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self.delay)
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def client(stream):
    return NS(chat=NS(completions=NS(create=AsyncMock(return_value=stream))))


class ArithmeticTests(unittest.TestCase):
    def test_capacity_null_block_and_rounding(self):
        self.assertEqual(k.free_blocks(0, 101), 100)
        self.assertEqual(k.free_blocks(.5, 101), 50)
        self.assertEqual(k.free_blocks(1, 101), 0)
        self.assertEqual([k.required_blocks(t, 16) for t in (0, 1, 16, 17)], [0, 1, 1, 2])
        self.assertEqual(k.Pool(101,16,1616).usable_tokens,1600)
        with self.assertRaises(ValueError):
            k.free_blocks(1.1, 101)

    def test_margin_strict_boundaries(self):
        self.assertTrue(k.admission_window(160, 194, 240, 24))
        self.assertFalse(k.admission_window(170, 194, 240, 24))
        self.assertFalse(k.admission_window(160, 194, 58, 24))

    def test_growth_targets_derived_and_bounded(self):
        for capacity in (32768, 36864, 38912):
            t=k.growth_targets(capacity,16,12,[80]*12)
            self.assertLessEqual(t["safe_aggregate"], .7*capacity)
            self.assertGreaterEqual(t["pressure_aggregate"],1.1*capacity)
            self.assertLess(t["pressure_context"],4096)
            self.assertEqual(t["safe_context"]%16,0)
            self.assertEqual(t["pressure_max_tokens"][0]+80,t["pressure_context"])
        with self.assertRaises(ValueError):
            k.growth_targets(60000,16,12,[80]*12)

    def test_counter_reset_rejected(self):
        self.assertEqual(k.counter_delta(7,9),2)
        with self.assertRaises(ValueError):
            k.counter_delta(9,7)

    def test_schedule_reproducible_pairs(self):
        a=k.make_schedule(3,k.ADMISSION)
        self.assertEqual(a,k.make_schedule(3,k.ADMISSION))
        for block in (1,2,3):
            self.assertEqual({c for b,c in a if b==block},set(k.ADMISSION))

    def test_validity_requires_joint_evidence(self):
        row=dict(initial_admitted=True,drained=True,stream_integrity=True,abort_verified=True,
                 external_kv_transfer_delta=0,condition=k.ADMISSION[0],preemption_delta=0,
                 prefix_hits_delta=0,trigger_free_blocks=160,fg_required_blocks=194,
                 holder_blocks_estimate=240,margin_blocks=24,release_headroom_verified=True,
                 fg_send_ns=100,release_ns=200,fg_first_ns=300,blocked_samples=3,blocked_span_ms=101)
        k.validate_trial(row)
        for field, value in (("preemption_delta",1),("blocked_samples",2),("initial_admitted",False),
                             ("abort_verified",False),("release_headroom_verified",False),
                             ("trigger_free_blocks",190),("fg_first_ns",150)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                k.validate_trial(row | {field:value})

    def test_growth_validity_no_waiting_requirement(self):
        row=dict(initial_admitted=True,drained=True,stream_integrity=True,abort_verified=True,
                 external_kv_transfer_delta=0,condition=k.GROWTH[1],preemption_delta=1,
                 max_kv_usage=.99,prefix_hits_before_preemption=0)
        k.validate_trial(row)
        with self.assertRaises(ValueError):
            k.validate_trial(row | {"preemption_delta":0})


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_multitoken_and_empty_events(self):
        r=k.Request("test","nonce",80,3)
        s=FakeStream([chunk(),chunk([1,2]),chunk([3],finish="length"),chunk(usage=(80,3))])
        await k.stream_request(client(s),r)
        self.assertEqual([e["delta_token_count"] for e in r.events],[0,2,1,0])
        self.assertEqual(r.cumulative,3)
        self.assertTrue(r.valid and s.closed)

    async def test_integrity_mismatch(self):
        r=k.Request("test","nonce",80,3)
        s=FakeStream([chunk([1],finish="length"),chunk(usage=(80,3))])
        with self.assertRaises(ValueError):
            await k.stream_request(client(s),r)
        self.assertTrue(s.closed)

    async def test_cancel_closes_without_fake_final_usage(self):
        r=k.Request("test","nonce",80,100)
        s=FakeStream([chunk([1])]*100,delay=.01)
        task=asyncio.create_task(k.stream_request(client(s),r))
        await asyncio.sleep(.035)
        await k.cancel(r,task)
        self.assertTrue(r.cancelled and s.closed)
        self.assertIsNone(r.completion_tokens)
        self.assertGreaterEqual(r.closed_ns,r.cancel_ns)
        self.assertFalse(r.valid)

    async def test_finished_request_cannot_be_release(self):
        r=k.Request("test","nonce",80,1)
        task=asyncio.create_task(k.stream_request(client(FakeStream([
            chunk([1],finish="length"),chunk(usage=(80,1))])),r))
        await task
        with self.assertRaises(ValueError):
            await k.cancel(r,task)


if __name__ == "__main__":
    unittest.main()
