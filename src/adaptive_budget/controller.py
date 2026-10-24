# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pure state machine: no GPU calls, wall-clock assumptions, or queue mutation.

P-PAS pressure switching is an attributed comparator, not an original algorithm.
The adaptive policy adds hysteresis, minimum residence, and a rate-limited
budget boost for prefills that have not made scheduling progress.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "adaptive"
    high_budget: int = 8192
    low_budget: int = 1024
    static_budget: int = 8192
    enter_prefills: int = 2
    exit_prefills: int = 0
    min_dwell_steps: int = 4
    guard_enabled: bool = True
    max_stall_ms: float = 250.0
    guard_budget: int = 4096
    guard_interval_steps: int = 8

    def validate(self, engine_capacity: int) -> None:
        if not isinstance(self.mode, str) or self.mode not in {
            "off",
            "static",
            "ppas",
            "adaptive",
        }:
            raise ValueError(f"Unknown policy mode: {self.mode}")
        if type(engine_capacity) is not int or engine_capacity < 1:
            raise ValueError("Engine capacity must be positive")
        if self.mode == "off":
            return
        if self.mode == "static":
            if (
                type(self.static_budget) is not int
                or not 1 <= self.static_budget <= engine_capacity
            ):
                raise ValueError(
                    "Static budget exceeds engine capacity or is nonpositive"
                )
            return
        for name in ("high_budget", "low_budget", "enter_prefills"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.low_budget <= self.high_budget <= engine_capacity:
            raise ValueError(
                "Require 1 <= low_budget <= high_budget <= engine capacity"
            )
        if self.enter_prefills < 1:
            raise ValueError("enter_prefills must be positive")
        if self.mode == "ppas":
            return
        for name in (
            "exit_prefills",
            "min_dwell_steps",
            "guard_budget",
            "guard_interval_steps",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if type(self.guard_enabled) is not bool:
            raise ValueError("guard_enabled must be a boolean")
        if not self.low_budget <= self.guard_budget <= self.high_budget:
            raise ValueError("Guard budget must lie between low and high budgets")
        if not 0 <= self.exit_prefills < self.enter_prefills:
            raise ValueError("Require 0 <= exit_prefills < enter_prefills")
        if self.min_dwell_steps < 0 or self.guard_interval_steps < 1:
            raise ValueError("Invalid residence/cooldown step count")
        if (
            type(self.max_stall_ms) not in (int, float)
            or not math.isfinite(self.max_stall_ms)
            or self.max_stall_ms <= 0
        ):
            raise ValueError("Stall threshold must be finite and positive")


@dataclass(frozen=True)
class Observation:
    prefills: int
    decodes: int
    oldest_stall_ms: float = 0.0
    pending_prefill_tokens: int = 0


@dataclass(frozen=True)
class Decision:
    budget: int
    state: str
    reason: str
    switched: bool = False
    guard_fired: bool = False


class BudgetController:
    def __init__(self, config: PolicyConfig, engine_capacity: int):
        config.validate(engine_capacity)
        self.config = config
        self.engine_capacity = engine_capacity
        self.limited = False
        self.step = 0
        self.last_switch = -config.min_dwell_steps if config.mode == "adaptive" else 0
        self.last_guard = (
            -config.guard_interval_steps if config.mode == "adaptive" else 0
        )
        self.switches = 0
        self.guard_fires = 0

    def decide(self, obs: Observation) -> Decision:
        counts = (obs.prefills, obs.decodes, obs.pending_prefill_tokens)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Observation counts must be nonnegative integers")
        if (
            type(obs.oldest_stall_ms) not in (int, float)
            or not math.isfinite(obs.oldest_stall_ms)
            or obs.oldest_stall_ms < 0
        ):
            raise ValueError("Stall observation must be finite and nonnegative")
        cfg = self.config
        self.step += 1
        if cfg.mode == "off":
            return Decision(self.engine_capacity, "OFF", "native")
        if cfg.mode == "static":
            return Decision(cfg.static_budget, "STATIC", "fixed")

        previous = self.limited
        if cfg.mode == "ppas":
            self.limited = obs.prefills >= cfg.enter_prefills and obs.decodes > 0
            reason = "pressure" if self.limited else "open"
        elif obs.decodes == 0 or obs.prefills == 0:
            # No mixed-phase interference: bypass residence and release capacity.
            self.limited = False
            reason = "no_mixed_work"
        elif self.step - self.last_switch >= cfg.min_dwell_steps:
            if not self.limited and obs.prefills >= cfg.enter_prefills:
                self.limited = True
                reason = "enter_pressure"
            elif self.limited and obs.prefills <= cfg.exit_prefills:
                self.limited = False
                reason = "exit_pressure"
            else:
                reason = "hysteresis_hold"
        else:
            reason = "residence_hold"

        switched = previous != self.limited
        if switched:
            self.last_switch = self.step
            self.switches += 1
        budget = cfg.low_budget if self.limited else cfg.high_budget
        guard = (
            cfg.mode == "adaptive"
            and cfg.guard_enabled
            and self.limited
            and obs.prefills > 0
            and obs.oldest_stall_ms >= cfg.max_stall_ms
            and self.step - self.last_guard >= cfg.guard_interval_steps
        )
        if guard:
            budget = max(budget, cfg.guard_budget)
            self.last_guard = self.step
            self.guard_fires += 1
            reason = "stalled_prefill_boost"
        # A boost creates an opportunity; it is not an admission/fairness guarantee.
        return Decision(
            budget, "LIMITED" if self.limited else "OPEN", reason, switched, guard
        )
