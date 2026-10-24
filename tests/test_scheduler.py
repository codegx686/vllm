"""Integration tests against the installed vLLM scheduler, using its real KV allocator.

Fixture construction follows vLLM v0.22.1 tests/v1/core/utils.py (Apache-2.0).
No model execution/GPU memory allocation is needed for these scheduler tests.
"""

import json
import os

import pytest
import torch
from examples.features.adaptive_budget.scheduler import (
    AdaptiveBudgetScheduler as ExampleScheduler,
)
from examples.features.adaptive_budget.scheduler import (
    BudgetConfig,
)
from examples.features.adaptive_budget.scheduler import (
    BudgetController as ExampleController,
)
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from adaptive_budget.scheduler import AdaptiveBudgetScheduler


@pytest.fixture
def make_scheduler(monkeypatch, tmp_path):
    schedulers = []

    def make(
        mode="static",
        budget=1024,
        cls=AdaptiveBudgetScheduler,
        blocks=4096,
        policy=None,
        prefix_caching=False,
        additional=False,
        async_scheduling=False,
        policy_only=False,
    ):
        path = tmp_path / f"{mode}.json"
        config = dict(mode=mode, static_budget=budget, **(policy or {}))
        if policy_only:
            config = policy or {}
        path.write_text(json.dumps(config))
        monkeypatch.delenv("AB_CONFIG", raising=False)
        if not additional:
            monkeypatch.setenv("AB_CONFIG", str(path))
        model = ModelConfig(
            model=os.environ["AB_TEST_MODEL"], dtype="bfloat16", max_model_len=16384
        )
        sc = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            max_model_len=16384,
            enable_chunked_prefill=True,
            async_scheduling=async_scheduling,
            is_encoder_decoder=False,
        )
        cache = CacheConfig(block_size=16, enable_prefix_caching=prefix_caching)
        cache.num_gpu_blocks = blocks
        cfg = VllmConfig(
            model_config=model,
            scheduler_config=sc,
            cache_config=cache,
            additional_config={"adaptive_budget": config} if additional else {},
        )
        kv = KVCacheConfig(
            num_blocks=blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                    ),
                )
            ],
        )
        scheduler = cls(
            vllm_config=cfg,
            kv_cache_config=kv,
            block_size=16,
            structured_output_manager=StructuredOutputManager(cfg),
            log_stats=False,
        )
        schedulers.append(scheduler)
        return scheduler

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()


def request(rid, n):
    return Request(
        request_id=rid,
        prompt_token_ids=[42] * n,
        sampling_params=SamplingParams(max_tokens=32, ignore_eos=True),
        pooling_params=None,
        mm_features=None,
    )


def test_off_has_native_schedule_and_cancellation(make_scheduler):
    a, b = make_scheduler(mode="off"), make_scheduler(cls=Scheduler)
    for s in [a, b]:
        for i, n in enumerate([100, 3000, 7000]):
            s.add_request(request(str(i), n))
    assert a.schedule().num_scheduled_tokens == b.schedule().num_scheduled_tokens
    for s in [a, b]:
        s.finish_requests(["0", "1", "2"], RequestStatus.FINISHED_ABORTED)
        assert s.get_num_unfinished_requests() == 0


@pytest.mark.parametrize("budget", [512, 1024, 2048, 4096, 8192])
def test_real_allocator_respects_budget_and_restores_capacity(make_scheduler, budget):
    s = make_scheduler(budget=budget)
    s.add_request(request("long", 12000))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"long": budget}
    assert s.max_num_scheduled_tokens == 8192
    s.finish_requests("long", RequestStatus.FINISHED_ABORTED)
    s._ab_observe(1)
    assert not s.ab_progress


def test_capacity_restored_when_native_schedule_raises(make_scheduler, monkeypatch):
    s = make_scheduler()

    def fail(_):
        raise RuntimeError("simulated allocator failure")

    monkeypatch.setattr(Scheduler, "schedule", fail)
    with pytest.raises(RuntimeError):
        s.schedule()
    assert s.max_num_scheduled_tokens == 8192


def test_stall_progress_and_blocked_request_filter(make_scheduler):
    s = make_scheduler()
    r = request("r", 4000)
    s.add_request(r)
    assert s._ab_observe(10).oldest_stall_ms == 0
    assert s._ab_observe(11).oldest_stall_ms == 1000
    r.num_computed_tokens = 100
    assert s._ab_observe(12).oldest_stall_ms == 0
    r.num_computed_tokens = 0
    r.status = RequestStatus.PREEMPTED
    assert s._ab_observe(13).oldest_stall_ms == 1000
    r.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert s._ab_observe(14).prefills == 0
    assert not s.ab_progress


@pytest.mark.parametrize("mode", ["off", "adaptive"])
def test_kv_pressure_preempts_and_resumes_without_losing_request(make_scheduler, mode):
    s = make_scheduler(mode=mode, blocks=11)
    a, b = request("a", 80), request("b", 80)
    s.add_request(a)
    first = s.schedule()
    s.add_request(b)
    s.schedule()
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    model_out = ModelRunnerOutput(
        req_ids=["a"],
        req_id_to_index={"a": 0},
        sampled_token_ids=[[100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    s.update_from_output(first, model_out)
    s.schedule()
    assert b.status == RequestStatus.PREEMPTED
    assert b.request_id in s.requests
    s.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    resumed = s.schedule()
    assert resumed.num_scheduled_tokens["b"] == 80
    assert b.status == RequestStatus.RUNNING
    s.finish_requests("b", RequestStatus.FINISHED_ABORTED)
    s._ab_observe(20)
    assert s.get_num_unfinished_requests() == 0
    assert not s.ab_progress
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 10


def test_additional_config_and_trace_record_actual_budget(make_scheduler, tmp_path):
    trace_path = tmp_path / "schedule.jsonl"
    s = make_scheduler(
        additional=True, budget=512, policy={"trace_path": str(trace_path)}
    )
    s.add_request(request("long", 2000))
    out = s.schedule()
    s.shutdown()
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[0]["capacity"] == 8192
    assert records[1]["budget"] == records[1]["scheduled_tokens"] == 512
    assert out.num_scheduled_tokens == {"long": 512}
    assert s.ab_trace is None


def test_reused_request_id_does_not_inherit_stall(make_scheduler):
    s = make_scheduler(mode="adaptive")
    s.add_request(request("reused", 100))
    s._ab_observe(1)
    s.finish_requests("reused", RequestStatus.FINISHED_ABORTED)
    s.add_request(request("reused", 100))
    assert s._ab_observe(10).oldest_stall_ms == 0


def test_off_rejects_async_scheduler_mismatch(make_scheduler):
    with pytest.raises(ValueError, match="async_scheduling=False"):
        make_scheduler(mode="off", async_scheduling=True)


def test_mixed_work_selects_low_budget_through_real_scheduler(make_scheduler):
    s = make_scheduler(
        mode="adaptive",
        additional=True,
        policy={"enter_prefills": 1, "guard_enabled": False},
    )
    s.add_request(request("decode", 32))
    out = s.schedule()
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=["decode"],
            req_id_to_index={"decode": 0},
            sampled_token_ids=[[100]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    s.add_request(request("prefill", 8000))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"decode": 1, "prefill": 1023}
    assert s.max_num_scheduled_tokens == 8192
    assert s.scheduler_config.max_num_batched_tokens == 8192


def test_upstream_example_mixed_work_and_cancellation(make_scheduler):
    s = make_scheduler(
        cls=ExampleScheduler,
        additional=True,
        policy_only=True,
        policy={"enter_prefills": 1},
    )
    s.add_request(request("decode", 32))
    out = s.schedule()
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=["decode"],
            req_id_to_index={"decode": 0},
            sampled_token_ids=[[100]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    s.add_request(request("prefill", 8000))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"decode": 1, "prefill": 1023}
    assert s.max_num_scheduled_tokens == 8192
    s.finish_requests(["decode", "prefill"], RequestStatus.FINISHED_ABORTED)
    assert s.schedule().total_num_scheduled_tokens == 0
    assert not s.prefill_progress


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"enter_prefills": 3, "exit_prefills": 1},
        {"guard_enabled": False},
        {"min_dwell_steps": 0},
    ],
)
def test_upstream_example_matches_experiment_controller(config):
    """The reduced upstream example must retain the measured adaptive policy."""
    import random

    from adaptive_budget.controller import BudgetController, Observation, PolicyConfig

    lab = BudgetController(PolicyConfig(**config), 8192)
    example = ExampleController(BudgetConfig(**config))
    rng = random.Random(20261024)
    for _ in range(1000):
        p, d, stall = rng.randrange(5), rng.randrange(5), rng.random() * 500
        assert (
            example.decide(p, d, stall) == lab.decide(Observation(p, d, stall)).budget
        )


def test_trace_io_failure_does_not_discard_allocated_step(
    make_scheduler, tmp_path, monkeypatch
):
    """An external diagnostics failure must not turn a committed schedule into a retry."""
    from pathlib import Path

    trace_path = tmp_path / "failing-trace.jsonl"
    open_file = Path.open

    class FailingTrace:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def write(self, data):
            self.writes += 1
            if self.writes > 1:
                raise OSError("simulated full trace filesystem")
            return self.stream.write(data)

        def flush(self):
            self.stream.flush()

        def close(self):
            self.stream.close()

    def open_trace(path, *args, **kwargs):
        stream = open_file(path, *args, **kwargs)
        return FailingTrace(stream) if path == trace_path else stream

    monkeypatch.setattr(Path, "open", open_trace)
    s = make_scheduler(
        budget=512, additional=True, policy={"trace_path": str(trace_path)}
    )
    s.add_request(request("long", 2000))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"long": 512}
    assert s.requests["long"].num_computed_tokens == 512
    assert s.ab_trace is None
    assert s.max_num_scheduled_tokens == 8192
