import random
from dataclasses import replace

import pytest

from adaptive_budget.controller import BudgetController, Observation, PolicyConfig


def test_off_accepts_small_native_capacity():
    c = BudgetController(PolicyConfig(mode="off"), 256)
    assert c.decide(Observation(100, 100, 10000)).budget == 256


def test_static_ignores_unused_adaptive_parameters():
    c = BudgetController(PolicyConfig(mode="static", static_budget=256), 512)
    assert {c.decide(Observation(p, d)).budget for p, d in [(0, 0), (4, 8)]} == {256}


def test_ppas_matches_pressure_rule():
    c = BudgetController(PolicyConfig(mode="ppas"), 8192)
    for p in range(5):
        for d in range(5):
            assert c.decide(Observation(p, d)).budget == (
                1024 if p >= 2 and d > 0 else 8192
            )


def test_hysteresis_reduces_threshold_chatter():
    adaptive = BudgetController(PolicyConfig(guard_enabled=False), 8192)
    ppas = BudgetController(PolicyConfig(mode="ppas"), 8192)
    for p in [2, 1] * 20:
        adaptive.decide(Observation(p, 4))
        ppas.decide(Observation(p, 4))
    assert adaptive.switches == 1
    assert ppas.switches == 40


def test_dwell_prevents_immediate_reentry():
    c = BudgetController(PolicyConfig(min_dwell_steps=4), 8192)
    assert c.decide(Observation(2, 1)).state == "LIMITED"
    assert c.decide(Observation(0, 1)).state == "OPEN"
    assert c.decide(Observation(2, 1)).reason == "residence_hold"
    c.decide(Observation(2, 1))
    c.decide(Observation(2, 1))
    assert c.decide(Observation(2, 1)).state == "LIMITED"


@pytest.mark.parametrize("obs", [Observation(0, 8), Observation(8, 0)])
def test_unmixed_work_releases_capacity_immediately(obs):
    c = BudgetController(PolicyConfig(min_dwell_steps=100), 8192)
    c.decide(Observation(3, 8))
    assert c.decide(obs).budget == 8192


def test_guard_is_rate_limited_and_does_not_change_state():
    c = BudgetController(PolicyConfig(guard_interval_steps=3), 8192)
    decisions = [c.decide(Observation(3, 4, 500)) for _ in range(7)]
    assert [i for i, d in enumerate(decisions) if d.guard_fired] == [0, 3, 6]
    assert all(d.state == "LIMITED" for d in decisions)
    assert c.switches == 1
    assert c.guard_fires == 3


def test_guard_ablation():
    c = BudgetController(PolicyConfig(guard_enabled=False), 8192)
    assert c.decide(Observation(4, 8, 10000)).budget == 1024
    assert c.guard_fires == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "bad"},
        {"low_budget": 0},
        {"high_budget": 9000},
        {"exit_prefills": 2},
        {"guard_budget": 9999},
        {"min_dwell_steps": -1},
        {"guard_interval_steps": 0},
        {"max_stall_ms": 0},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        BudgetController(replace(PolicyConfig(), **changes), 8192)


def test_negative_observation_rejected():
    with pytest.raises(ValueError):
        BudgetController(PolicyConfig(), 8192).decide(Observation(-1, 0))


def test_random_trace_safety_and_guard_cooldown():
    rng = random.Random(101)
    c = BudgetController(PolicyConfig(), 8192)
    last_guard = -100
    for step in range(10000):
        o = Observation(rng.randrange(20), rng.randrange(20), rng.random() * 1000)
        d = c.decide(o)
        assert 1024 <= d.budget <= 8192
        if not o.prefills or not o.decodes:
            assert d.budget == 8192
        if d.guard_fired:
            assert step - last_guard >= 8
            last_guard = step


@pytest.mark.parametrize(
    "changes",
    [
        {"low_budget": 1.5},
        {"high_budget": True},
        {"enter_prefills": 2.5},
        {"exit_prefills": False},
        {"min_dwell_steps": 1.5},
        {"guard_interval_steps": True},
        {"guard_enabled": "false"},
        {"max_stall_ms": float("nan")},
        {"max_stall_ms": float("inf")},
        {"mode": "static", "static_budget": True},
    ],
)
def test_rejects_non_integral_budgets_and_nonfinite_thresholds(changes):
    """JSON values must not leak floats or booleans into native token accounting."""
    with pytest.raises(ValueError):
        BudgetController(replace(PolicyConfig(), **changes), 8192)


def test_dwell_holds_both_transitions_during_mixed_work():
    c = BudgetController(
        PolicyConfig(
            enter_prefills=3, exit_prefills=1, min_dwell_steps=3, guard_enabled=False
        ),
        8192,
    )
    states = [c.decide(Observation(p, 4)).state for p in [3, 1, 1, 1, 3, 3, 3]]
    assert states == ["LIMITED"] * 3 + ["OPEN"] * 3 + ["LIMITED"]
    assert c.switches == 3


@pytest.mark.parametrize(
    "obs",
    [
        Observation(True, 1),
        Observation(1, 1.5),
        Observation(1, 1, float("nan")),
        Observation(1, 1, float("inf")),
    ],
)
def test_invalid_observations_do_not_advance_controller(obs):
    c = BudgetController(PolicyConfig(), 8192)
    with pytest.raises(ValueError):
        c.decide(obs)
    assert c.step == 0
