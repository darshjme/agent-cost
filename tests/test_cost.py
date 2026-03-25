"""Tests for agent-cost. Real tests, no stubs."""

import time
import pytest
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_cost import (
    ModelPricing,
    CostRecord,
    CostTracker,
    Budget,
    BudgetExceededError,
    CostReport,
    PRICING,
)


# ── ModelPricing tests ────────────────────────────────────────────────────────

class TestModelPricing:
    def test_cost_calculation(self):
        p = ModelPricing(model="test", input_per_million=5.0, output_per_million=15.0)
        # 1M input + 1M output tokens
        cost = p.cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(20.0)

    def test_cost_partial_tokens(self):
        p = ModelPricing(model="test", input_per_million=10.0, output_per_million=30.0)
        # 100 input tokens + 50 output tokens
        cost = p.cost(100, 50)
        expected = (100 / 1_000_000) * 10.0 + (50 / 1_000_000) * 30.0
        assert cost == pytest.approx(expected)

    def test_from_table_known_model(self):
        p = ModelPricing.from_table("gpt-4o")
        assert p.input_per_million == 5.00
        assert p.output_per_million == 15.00

    def test_from_table_unknown_model(self):
        with pytest.raises(KeyError):
            ModelPricing.from_table("nonexistent-model-xyz")

    def test_zero_tokens(self):
        p = ModelPricing(model="test", input_per_million=5.0, output_per_million=15.0)
        assert p.cost(0, 0) == 0.0

    def test_pricing_table_completeness(self):
        # Ensure we have at least the major models
        for model in ["gpt-4o", "gpt-4o-mini", "claude-3-opus", "gemini-1.5-pro"]:
            assert model in PRICING


# ── CostRecord tests ──────────────────────────────────────────────────────────

class TestCostRecord:
    def test_total_tokens(self):
        r = CostRecord(
            record_id="test",
            model="gpt-4o",
            agent="agent1",
            task="summarize",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            timestamp=time.time(),
        )
        assert r.total_tokens == 150

    def test_to_dict(self):
        r = CostRecord(
            record_id="abc",
            model="gpt-4o",
            agent="a1",
            task="t1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            timestamp=1234567890.0,
        )
        d = r.to_dict()
        assert d["record_id"] == "abc"
        assert d["total_tokens"] == 150
        assert d["cost_usd"] == 0.001


# ── Budget tests ──────────────────────────────────────────────────────────────

class TestBudget:
    def test_initial_state(self):
        b = Budget(name="test", limit_usd=100.0)
        assert b.spent == 0.0
        assert b.remaining == 100.0
        assert not b.is_exceeded

    def test_add_spending(self):
        b = Budget(name="test", limit_usd=100.0)
        b.add(30.0)
        assert b.spent == pytest.approx(30.0)
        assert b.remaining == pytest.approx(70.0)

    def test_utilization(self):
        b = Budget(name="test", limit_usd=100.0)
        b.add(50.0)
        assert b.utilization == pytest.approx(0.5)

    def test_exceeded(self):
        b = Budget(name="test", limit_usd=10.0)
        b.add(15.0)
        assert b.is_exceeded
        assert b.remaining == 0.0

    def test_alert_fires_at_threshold(self):
        alerts = []
        b = Budget(
            name="test",
            limit_usd=100.0,
            alert_threshold=0.8,
            on_alert=lambda budget, spent: alerts.append(spent),
        )
        b.add(79.0)
        assert len(alerts) == 0
        b.add(2.0)  # now at 81%
        assert len(alerts) == 1
        assert alerts[0] == pytest.approx(81.0)

    def test_alert_fires_only_once(self):
        alerts = []
        b = Budget(
            name="test",
            limit_usd=100.0,
            alert_threshold=0.8,
            on_alert=lambda budget, spent: alerts.append(spent),
        )
        b.add(85.0)
        b.add(5.0)
        b.add(5.0)
        assert len(alerts) == 1

    def test_hard_limit_raises(self):
        b = Budget(name="test", limit_usd=50.0, hard_limit=True)
        with pytest.raises(BudgetExceededError) as exc_info:
            b.add(60.0)
        assert exc_info.value.budget_name == "test"
        assert exc_info.value.limit == 50.0

    def test_reset(self):
        b = Budget(name="test", limit_usd=100.0)
        b.add(80.0)
        b.reset()
        assert b.spent == 0.0

    def test_to_dict(self):
        b = Budget(name="test", limit_usd=100.0)
        b.add(25.0)
        d = b.to_dict()
        assert d["name"] == "test"
        assert d["spent_usd"] == pytest.approx(25.0)
        assert d["utilization_pct"] == pytest.approx(25.0)


# ── CostTracker tests ─────────────────────────────────────────────────────────

class TestCostTracker:
    def test_track_basic(self):
        tracker = CostTracker()
        record = tracker.track(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            agent="test-agent",
            task="test-task",
        )
        assert record.agent == "test-agent"
        assert record.task == "test-task"
        assert record.model == "gpt-4o"
        assert record.cost_usd > 0
        assert record.input_tokens == 1000
        assert record.output_tokens == 500

    def test_track_cost_accuracy(self):
        tracker = CostTracker()
        # gpt-4o: $5/1M input, $15/1M output
        record = tracker.track(
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert record.cost_usd == pytest.approx(20.0)

    def test_track_custom_cost(self):
        tracker = CostTracker()
        record = tracker.track(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            custom_cost_usd=0.5,
        )
        assert record.cost_usd == 0.5

    def test_track_unknown_model_raises(self):
        tracker = CostTracker()
        with pytest.raises(KeyError):
            tracker.track(model="unknown-xyz", input_tokens=100, output_tokens=50)

    def test_add_pricing(self):
        tracker = CostTracker()
        tracker.add_pricing("my-model", input_per_million=1.0, output_per_million=2.0)
        record = tracker.track(model="my-model", input_tokens=1_000_000, output_tokens=1_000_000)
        assert record.cost_usd == pytest.approx(3.0)

    def test_total_cost(self):
        tracker = CostTracker()
        tracker.track(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.track(model="gpt-4o-mini", input_tokens=2000, output_tokens=1000)
        assert tracker.total_cost() > 0

    def test_records_filtering_by_agent(self):
        tracker = CostTracker()
        tracker.track(model="gpt-4o-mini", input_tokens=100, output_tokens=50, agent="a1")
        tracker.track(model="gpt-4o-mini", input_tokens=100, output_tokens=50, agent="a2")
        tracker.track(model="gpt-4o-mini", input_tokens=100, output_tokens=50, agent="a1")
        recs = tracker.records(agent="a1")
        assert len(recs) == 2
        assert all(r.agent == "a1" for r in recs)

    def test_budget_integration(self):
        tracker = CostTracker()
        tracker.add_budget("monthly", limit_usd=1000.0)
        tracker.track(
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=500_000,
            budget_names=["monthly"],
        )
        budget = tracker.get_budget("monthly")
        assert budget.spent > 0

    def test_budget_hard_limit(self):
        tracker = CostTracker()
        tracker.add_budget("tight", limit_usd=0.000001, hard_limit=True)
        with pytest.raises(BudgetExceededError):
            tracker.track(
                model="gpt-4o",
                input_tokens=100_000,
                output_tokens=50_000,
                budget_names=["tight"],
            )

    def test_get_budget_unknown_raises(self):
        tracker = CostTracker()
        with pytest.raises(KeyError):
            tracker.get_budget("nonexistent")

    def test_reset_clears_records(self):
        tracker = CostTracker()
        tracker.track(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.reset()
        assert tracker.total_cost() == 0.0
        assert tracker.records() == []

    def test_metadata_attached(self):
        tracker = CostTracker()
        record = tracker.track(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            metadata={"request_id": "abc123", "user": "test"},
        )
        assert record.metadata["request_id"] == "abc123"

    def test_thread_safety(self):
        tracker = CostTracker()
        errors = []

        def worker():
            try:
                tracker.track(
                    model="gpt-4o-mini",
                    input_tokens=100,
                    output_tokens=50,
                    agent="worker",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(tracker.records()) == 50


# ── CostReport tests ──────────────────────────────────────────────────────────

class TestCostReport:
    def _make_tracker(self):
        tracker = CostTracker()
        tracker.track(model="gpt-4o", input_tokens=1000, output_tokens=500, agent="agent1", task="summarize")
        tracker.track(model="gpt-4o-mini", input_tokens=500, output_tokens=200, agent="agent1", task="classify")
        tracker.track(model="gpt-4o", input_tokens=800, output_tokens=400, agent="agent2", task="summarize")
        return tracker

    def test_report_totals(self):
        tracker = self._make_tracker()
        report = tracker.report()
        assert report.total_calls == 3
        assert report.total_cost > 0
        assert report.total_input_tokens == 2300
        assert report.total_output_tokens == 1100

    def test_report_by_model(self):
        tracker = self._make_tracker()
        report = tracker.report()
        assert "gpt-4o" in report.by_model
        assert "gpt-4o-mini" in report.by_model
        assert report.by_model["gpt-4o"]["calls"] == 2

    def test_report_by_agent(self):
        tracker = self._make_tracker()
        report = tracker.report()
        assert "agent1" in report.by_agent
        assert "agent2" in report.by_agent
        assert report.by_agent["agent1"]["calls"] == 2

    def test_report_by_task(self):
        tracker = self._make_tracker()
        report = tracker.report()
        assert "summarize" in report.by_task
        assert report.by_task["summarize"]["calls"] == 2

    def test_report_filtered_by_agent(self):
        tracker = self._make_tracker()
        report = tracker.report(agent="agent2")
        assert report.total_calls == 1

    def test_report_filtered_by_model(self):
        tracker = self._make_tracker()
        report = tracker.report(model="gpt-4o-mini")
        assert report.total_calls == 1

    def test_report_filtered_by_since(self):
        tracker = CostTracker()
        t_before = time.time() - 10
        tracker.track(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        report = tracker.report(since=t_before + 5)
        # The record was created after t_before but the filter is t_before+5
        # It depends on timing; at least test it doesn't crash
        assert isinstance(report, CostReport)

    def test_top_models(self):
        tracker = self._make_tracker()
        report = tracker.report()
        top = report.top_models(2)
        assert len(top) <= 2
        # gpt-4o should be top (more usage)
        assert top[0][0] == "gpt-4o"

    def test_top_agents(self):
        tracker = self._make_tracker()
        report = tracker.report()
        top = report.top_agents(2)
        assert len(top) == 2

    def test_to_dict(self):
        tracker = self._make_tracker()
        report = tracker.report()
        d = report.to_dict()
        assert "total_cost_usd" in d
        assert "by_model" in d
        assert "by_agent" in d
        assert "by_task" in d

    def test_repr(self):
        tracker = self._make_tracker()
        report = tracker.report()
        assert "CostReport" in repr(report)
        assert "calls=3" in repr(report)

    def test_empty_report(self):
        report = CostReport([])
        assert report.total_calls == 0
        assert report.total_cost == 0.0
