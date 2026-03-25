"""
agent-cost: Per-call cost tracking and attribution for LLM agents.

Track costs by agent, task, and model. Set budgets, receive alerts,
generate cost reports. Zero dependencies. Pure Python 3.8+.

Components:
  - ModelPricing: Price table per model (input/output $/1M tokens)
  - CostRecord: Immutable record of a single LLM call's cost
  - CostTracker: Main tracking engine (thread-safe)
  - Budget: Named budget with optional alerts
  - CostReport: Aggregated summary view
"""

from __future__ import annotations

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from collections import defaultdict


__version__ = "0.1.0"
__all__ = [
    "ModelPricing",
    "CostRecord",
    "CostTracker",
    "Budget",
    "BudgetExceededError",
    "CostReport",
    "PRICING",
]


# Default pricing table (USD per 1M tokens, as of 2025)
# Format: model_name -> (input_price, output_price)
PRICING: Dict[str, tuple] = {
    # OpenAI
    "gpt-4o":              (5.00,   15.00),
    "gpt-4o-mini":         (0.15,    0.60),
    "gpt-4-turbo":         (10.00,  30.00),
    "gpt-4":               (30.00,  60.00),
    "gpt-3.5-turbo":       (0.50,   1.50),
    "o1":                  (15.00,  60.00),
    "o1-mini":             (3.00,   12.00),
    # Anthropic
    "claude-opus-4":       (15.00,  75.00),
    "claude-sonnet-4":     (3.00,   15.00),
    "claude-haiku-3-5":    (0.80,    4.00),
    "claude-3-opus":       (15.00,  75.00),
    "claude-3-sonnet":     (3.00,   15.00),
    "claude-3-haiku":      (0.25,   1.25),
    # Google
    "gemini-1.5-pro":      (3.50,   10.50),
    "gemini-1.5-flash":    (0.075,   0.30),
    "gemini-2.0-flash":    (0.10,    0.40),
    # Meta / Open Source (hosted pricing estimate)
    "llama-3.1-70b":       (0.52,   0.75),
    "llama-3.1-8b":        (0.20,   0.20),
    # Mistral
    "mistral-large":       (4.00,   12.00),
    "mistral-small":       (2.00,    6.00),
}


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded and hard_limit=True."""

    def __init__(self, budget_name: str, limit: float, current: float):
        self.budget_name = budget_name
        self.limit = limit
        self.current = current
        super().__init__(
            f"Budget '{budget_name}' exceeded: ${current:.6f} > ${limit:.6f}"
        )


@dataclass
class ModelPricing:
    """Price definition for a model."""
    model: str
    input_per_million: float   # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute total cost in USD."""
        return (
            (input_tokens / 1_000_000) * self.input_per_million
            + (output_tokens / 1_000_000) * self.output_per_million
        )

    @classmethod
    def from_table(cls, model: str) -> "ModelPricing":
        """Look up pricing from the built-in PRICING table."""
        if model not in PRICING:
            raise KeyError(f"Unknown model '{model}'. Add it to PRICING or pass custom ModelPricing.")
        inp, out = PRICING[model]
        return cls(model=model, input_per_million=inp, output_per_million=out)


@dataclass
class CostRecord:
    """Immutable record of a single LLM API call."""
    record_id: str
    model: str
    agent: str
    task: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "model": self.model,
            "agent": self.agent,
            "task": self.task,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Budget:
    """Named spending budget with optional alert and hard limit."""
    name: str
    limit_usd: float
    alert_threshold: float = 0.8  # alert at 80% of limit
    hard_limit: bool = False       # raise BudgetExceededError if True
    on_alert: Optional[Callable[["Budget", float], None]] = None
    _spent: float = field(default=0.0, init=False, repr=False)
    _alert_fired: bool = field(default=False, init=False, repr=False)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self._spent)

    @property
    def utilization(self) -> float:
        if self.limit_usd <= 0:
            return 1.0
        return self._spent / self.limit_usd

    @property
    def is_exceeded(self) -> bool:
        return self._spent > self.limit_usd

    def add(self, amount: float) -> None:
        """Add spending to this budget. Fires alerts and raises if configured."""
        self._spent += amount
        # Check alert threshold
        if (
            not self._alert_fired
            and self.utilization >= self.alert_threshold
            and self.on_alert is not None
        ):
            self._alert_fired = True
            self.on_alert(self, self._spent)
        # Check hard limit
        if self.hard_limit and self.is_exceeded:
            raise BudgetExceededError(self.name, self.limit_usd, self._spent)

    def reset(self) -> None:
        """Reset spending to zero."""
        self._spent = 0.0
        self._alert_fired = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "limit_usd": self.limit_usd,
            "spent_usd": self._spent,
            "remaining_usd": self.remaining,
            "utilization_pct": round(self.utilization * 100, 2),
            "is_exceeded": self.is_exceeded,
            "hard_limit": self.hard_limit,
        }


class CostReport:
    """Aggregated cost summary from a set of CostRecords."""

    def __init__(self, records: List[CostRecord]):
        self.records = records
        self._build()

    def _build(self):
        self.total_cost = sum(r.cost_usd for r in self.records)
        self.total_input_tokens = sum(r.input_tokens for r in self.records)
        self.total_output_tokens = sum(r.output_tokens for r in self.records)
        self.total_calls = len(self.records)

        # By model
        self.by_model: Dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
        )
        # By agent
        self.by_agent: Dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
        )
        # By task
        self.by_task: Dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
        )

        for r in self.records:
            for bucket, key in [(self.by_model, r.model), (self.by_agent, r.agent), (self.by_task, r.task)]:
                bucket[key]["calls"] += 1
                bucket[key]["cost_usd"] += r.cost_usd
                bucket[key]["input_tokens"] += r.input_tokens
                bucket[key]["output_tokens"] += r.output_tokens

        self.by_model = dict(self.by_model)
        self.by_agent = dict(self.by_agent)
        self.by_task = dict(self.by_task)

    def top_models(self, n: int = 5) -> List[tuple]:
        """Return top-N models by cost."""
        return sorted(self.by_model.items(), key=lambda x: x[1]["cost_usd"], reverse=True)[:n]

    def top_agents(self, n: int = 5) -> List[tuple]:
        """Return top-N agents by cost."""
        return sorted(self.by_agent.items(), key=lambda x: x[1]["cost_usd"], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 8),
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": self.by_model,
            "by_agent": self.by_agent,
            "by_task": self.by_task,
        }

    def __repr__(self) -> str:
        return (
            f"CostReport(calls={self.total_calls}, "
            f"total=${self.total_cost:.6f}, "
            f"models={list(self.by_model.keys())})"
        )


class CostTracker:
    """
    Thread-safe per-call LLM cost tracker with attribution and budget support.

    Usage:
        tracker = CostTracker()
        tracker.add_budget("monthly", limit_usd=100.0)

        record = tracker.track(
            model="gpt-4o",
            agent="researcher",
            task="summarize",
            input_tokens=1200,
            output_tokens=400,
        )
        print(tracker.report())
    """

    def __init__(self, pricing: Optional[Dict[str, tuple]] = None):
        """
        Args:
            pricing: Custom pricing table. Falls back to built-in PRICING.
                     Format: {model: (input_per_million, output_per_million)}
        """
        self._pricing_table = {**PRICING, **(pricing or {})}
        self._records: List[CostRecord] = []
        self._budgets: Dict[str, Budget] = {}
        self._lock = threading.Lock()

    def add_pricing(self, model: str, input_per_million: float, output_per_million: float) -> None:
        """Register custom pricing for a model."""
        self._pricing_table[model] = (input_per_million, output_per_million)

    def add_budget(
        self,
        name: str,
        limit_usd: float,
        alert_threshold: float = 0.8,
        hard_limit: bool = False,
        on_alert: Optional[Callable[["Budget", float], None]] = None,
    ) -> Budget:
        """Register a named budget."""
        budget = Budget(
            name=name,
            limit_usd=limit_usd,
            alert_threshold=alert_threshold,
            hard_limit=hard_limit,
            on_alert=on_alert,
        )
        with self._lock:
            self._budgets[name] = budget
        return budget

    def get_budget(self, name: str) -> Budget:
        """Retrieve a budget by name."""
        with self._lock:
            if name not in self._budgets:
                raise KeyError(f"No budget named '{name}'")
            return self._budgets[name]

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: str = "default",
        task: str = "default",
        budget_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_cost_usd: Optional[float] = None,
    ) -> CostRecord:
        """
        Record a single LLM call and its cost.

        Args:
            model: Model identifier (must be in pricing table or add via add_pricing).
            input_tokens: Number of input/prompt tokens consumed.
            output_tokens: Number of output/completion tokens consumed.
            agent: Agent identifier for attribution.
            task: Task identifier for attribution.
            budget_names: List of budget names to charge this call against.
            metadata: Arbitrary key-value pairs attached to the record.
            custom_cost_usd: Override auto-computed cost (useful for non-token pricing).

        Returns:
            CostRecord with computed cost.

        Raises:
            KeyError: If model is not in pricing table and no custom_cost_usd given.
            BudgetExceededError: If a hard-limit budget is exceeded.
        """
        if custom_cost_usd is not None:
            cost = custom_cost_usd
        else:
            if model not in self._pricing_table:
                raise KeyError(
                    f"Unknown model '{model}'. Use add_pricing() or pass custom_cost_usd."
                )
            inp_price, out_price = self._pricing_table[model]
            pricing = ModelPricing(
                model=model,
                input_per_million=inp_price,
                output_per_million=out_price,
            )
            cost = pricing.cost(input_tokens, output_tokens)

        record = CostRecord(
            record_id=str(uuid.uuid4()),
            model=model,
            agent=agent,
            task=task,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            for bname in (budget_names or []):
                if bname in self._budgets:
                    self._budgets[bname].add(cost)

        return record

    def report(
        self,
        agent: Optional[str] = None,
        task: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[float] = None,
    ) -> CostReport:
        """
        Generate a cost report, optionally filtered.

        Args:
            agent: Filter by agent name.
            task: Filter by task name.
            model: Filter by model name.
            since: Unix timestamp — only include records after this time.
        """
        with self._lock:
            records = list(self._records)

        if agent:
            records = [r for r in records if r.agent == agent]
        if task:
            records = [r for r in records if r.task == task]
        if model:
            records = [r for r in records if r.model == model]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]

        return CostReport(records)

    def total_cost(self) -> float:
        """Quick accessor for total cost across all records."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def reset(self) -> None:
        """Clear all records (budgets remain but are reset)."""
        with self._lock:
            self._records.clear()
            for b in self._budgets.values():
                b.reset()

    def records(
        self,
        agent: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[CostRecord]:
        """Return raw records, optionally filtered."""
        with self._lock:
            recs = list(self._records)
        if agent:
            recs = [r for r in recs if r.agent == agent]
        if task:
            recs = [r for r in recs if r.task == task]
        return recs
