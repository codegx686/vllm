# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in token budgets for the synchronous, text-only V1 scheduler.

Pass configuration in additional_config["adaptive_budget"]. AB_CONFIG remains
available for reproducing the original experiments, but cannot be combined
with the additional_config entry.
"""

import json
import os
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .controller import BudgetController, Observation, PolicyConfig

logger = init_logger(__name__)


class AdaptiveBudgetScheduler(Scheduler):
    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        additional = vllm_config.additional_config
        raw = (
            additional.get("adaptive_budget") if isinstance(additional, dict) else None
        )
        config_path = os.environ.get("AB_CONFIG")
        if config_path:
            if raw is not None:
                raise ValueError(
                    "Use either AB_CONFIG or additional_config.adaptive_budget"
                )
            raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("adaptive_budget must be a JSON object")
        raw = dict(raw)
        trace_path = raw.pop("trace_path", None)
        self.ab_config = PolicyConfig(**raw)
        sc = vllm_config.scheduler_config
        self.ab_capacity = sc.max_num_scheduled_tokens or sc.max_num_batched_tokens
        self.ab_controller = BudgetController(self.ab_config, self.ab_capacity)
        # This class inherits the synchronous scheduler even in off mode.
        if sc.async_scheduling:
            raise ValueError("AdaptiveBudgetScheduler requires async_scheduling=False")
        if self.ab_config.mode != "off":
            if vllm_config.speculative_config is not None:
                raise ValueError("Speculative decoding is not supported")
            if not sc.enable_chunked_prefill:
                raise ValueError("Adaptive budgets require chunked prefill")
            if (
                vllm_config.parallel_config.world_size != 1
                or vllm_config.parallel_config.data_parallel_size != 1
            ):
                raise ValueError("Adaptive budgets support one GPU only")
            if sc.is_multimodal_model or vllm_config.model_config.is_encoder_decoder:
                raise ValueError("Adaptive budgets support text-only decoder models")
            if sc.runner_type != "generate":
                raise ValueError("Adaptive budgets require the generate runner")
            if (
                vllm_config.kv_transfer_config is not None
                or vllm_config.ec_transfer_config is not None
            ):
                raise ValueError("KV/encoder transfer is not supported")
        super().__init__(vllm_config, *args, **kwargs)
        self.ab_progress: dict[str, tuple[Request, int, float]] = {}
        self.ab_trace: TextIO | None = None
        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.ab_trace = path.open("w", encoding="utf-8", buffering=65536)
            self.ab_trace.write(
                json.dumps(
                    {
                        "type": "config",
                        "pid": os.getpid(),
                        "capacity": self.ab_capacity,
                        "config": asdict(self.ab_config),
                    }
                )
                + "\n"
            )
            self.ab_trace.flush()

    def _ab_observe(self, now: float) -> Observation:
        prefills = decodes = remaining = 0
        oldest_stall = 0.0
        live: set[str] = set()
        for req_id, req in self.requests.items():
            if req.status not in (
                RequestStatus.WAITING,
                RequestStatus.RUNNING,
                RequestStatus.PREEMPTED,
            ):
                continue
            # Include replay of generated tokens after preemption. The last
            # generated token is the normal single-token decode input.
            prefill_end = max(req.num_prompt_tokens, req.num_tokens - 1)
            if req.num_computed_tokens < prefill_end:
                prefills += 1
                remaining += prefill_end - req.num_computed_tokens
                live.add(req_id)
                old_req, old_count, last_progress = self.ab_progress.get(
                    req_id, (req, req.num_computed_tokens, now)
                )
                if old_req is not req or req.num_computed_tokens > old_count:
                    last_progress = now
                self.ab_progress[req_id] = (req, req.num_computed_tokens, last_progress)
                oldest_stall = max(oldest_stall, (now - last_progress) * 1000)
            elif req.status == RequestStatus.RUNNING:
                decodes += 1
        for req_id in self.ab_progress.keys() - live:
            del self.ab_progress[req_id]
        return Observation(prefills, decodes, oldest_stall, remaining)

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self.ab_config.mode == "off":
            return super().schedule(*args, **kwargs)
        started = time.perf_counter_ns() if self.ab_trace is not None else 0
        obs = (
            Observation(0, 0)
            if self.ab_config.mode == "static"
            else self._ab_observe(time.monotonic())
        )
        decision = self.ab_controller.decide(obs)
        decision_us = (time.perf_counter_ns() - started) / 1000 if started else 0
        saved_budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = decision.budget
        try:
            result = super().schedule(*args, **kwargs)
        finally:
            self.max_num_scheduled_tokens = saved_budget
        if self.ab_trace is not None:
            record = {
                "type": "step",
                "step": self.ab_controller.step,
                "at": time.monotonic(),
                **asdict(obs),
                **asdict(decision),
                "scheduled_tokens": result.total_num_scheduled_tokens,
                "scheduled_requests": len(result.num_scheduled_tokens),
                "preempted_requests": len(result.preempted_req_ids or ()),
                "switches_total": self.ab_controller.switches,
                "guards_total": self.ab_controller.guard_fires,
                "decision_us": decision_us,
            }
            try:
                self.ab_trace.write(json.dumps(record, separators=(",", ":")) + "\n")
                if not obs.prefills or self.ab_controller.step % 128 == 0:
                    self.ab_trace.flush()
            except OSError:
                # Scheduling has already mutated request/KV state. An optional
                # diagnostic sink must not discard that successful result.
                logger.warning(
                    "Disabling adaptive budget trace after an I/O error", exc_info=True
                )
                try:
                    self.ab_trace.close()
                except OSError:
                    pass
                self.ab_trace = None
        return result

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if self.ab_trace is not None:
                with suppress(OSError):
                    self.ab_trace.close()
                self.ab_trace = None
            self.ab_progress.clear()
